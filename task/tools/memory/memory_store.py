import os
os.environ['OMP_NUM_THREADS'] = '1'

import json
from datetime import datetime, UTC, timedelta
import numpy as np
import faiss
from aidial_client import AsyncDial
from sentence_transformers import SentenceTransformer

from task.tools.memory._models import Memory, MemoryData, MemoryCollection


class LongTermMemoryStore:
    """
    Manages long-term memory storage for users.

    Storage format: Single JSON file per user in DIAL bucket
    - File: {user_id}/long-memories.json
    - Caching: In-memory cache with conversation_id as key
    - Deduplication: O(n log n) using FAISS batch search
    """

    DEDUP_INTERVAL_HOURS = 24

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache: dict[str, MemoryCollection] = {}
        faiss.omp_set_num_threads(1)

    async def _get_memory_file_path(self, dial_client: AsyncDial) -> str:
        """Get the path to the memory file in DIAL bucket."""
        bucket_with_app_home = await dial_client.get_dial_home()
        return f"files/{bucket_with_app_home}/__long-memories/data.json"

    async def _load_memories(self, api_key: str) -> MemoryCollection:
        async with AsyncDial(base_url=self.endpoint, api_key=api_key, api_version="2025-01-01-preview") as dial_client:
            memory_file_path = await self._get_memory_file_path(dial_client)
            
            if memory_file_path in self.cache:
                return self.cache[memory_file_path]
            
            try:
                response = await dial_client.download_file(memory_file_path)
                content = response.content.decode('utf-8')
                data = json.loads(content)
                collection = MemoryCollection.model_validate(data)
            except:
                collection = MemoryCollection()
            
            self.cache[memory_file_path] = collection
            return collection

    async def _save_memories(self, api_key: str, memories: MemoryCollection):
        """Save memories to DIAL bucket and update cache."""
        async with AsyncDial(base_url=self.endpoint, api_key=api_key, api_version="2025-01-01-preview") as dial_client:
            memory_file_path = await self._get_memory_file_path(dial_client)
            memories.updated_at = datetime.now(UTC)
            json_content = memories.model_dump_json(indent=None)
            await dial_client.upload_file(memory_file_path, json_content.encode('utf-8'))
            self.cache[memory_file_path] = memories

    async def add_memory(self, api_key: str, content: str, importance: float, category: str, topics: list[str]) -> str:
        """Add a new memory to storage."""
        memories = await self._load_memories(api_key)
        
        embedding = self.model.encode([content])[0].tolist()
        
        memory = Memory(
            data=MemoryData(
                id=int(datetime.now(UTC).timestamp()),
                content=content,
                importance=importance,
                category=category,
                topics=topics
            ),
            embedding=embedding
        )
        
        memories.memories.append(memory)
        await self._save_memories(api_key, memories)
        
        return f"Successfully stored memory: {content}"

    async def search_memories(self, api_key: str, query: str, top_k: int = 5) -> list[MemoryData]:
        """
        Search memories using semantic similarity.

        Returns:
            List of MemoryData objects (without embeddings)
        """
        collection = await self._load_memories(api_key)
        
        if not collection.memories:
            return []
        
        if self._needs_deduplication(collection):
            collection = await self._deduplicate_and_save(api_key, collection)
        
        # Perform vector search
        query_embedding = self.model.encode([query])[0]
        
        # Create FAISS index for search
        embeddings_matrix = np.array([mem.embedding for mem in collection.memories], dtype=np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_matrix)
        
        index = faiss.IndexFlatIP(embeddings_matrix.shape[1])
        index.add(embeddings_matrix)
        
        # Normalize query
        query_vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)
        
        # Search
        k = min(top_k, len(collection.memories))
        distances, indices = index.search(query_vector, k)
        
        # Return top_k results
        results = [collection.memories[idx].data for idx in indices[0]]
        return results

    def _needs_deduplication(self, collection: MemoryCollection) -> bool:
        """Check if deduplication is needed (>24 hours since last deduplication)."""
        if len(collection.memories) <= 10:
            return False
        
        if collection.last_deduplicated_at is None:
            return True
        
        time_since_last_dedup = datetime.now(UTC) - collection.last_deduplicated_at
        return time_since_last_dedup > timedelta(hours=self.DEDUP_INTERVAL_HOURS)

    async def _deduplicate_and_save(self, api_key: str, collection: MemoryCollection) -> MemoryCollection:
        """
        Deduplicate memories synchronously and save the result.
        Returns the updated collection.
        """
        collection.memories = self._deduplicate_fast(collection.memories)
        collection.last_deduplicated_at = datetime.now(UTC)
        await self._save_memories(api_key, collection)
        return collection

    def _deduplicate_fast(self, memories: list[Memory]) -> list[Memory]:
        """
        Fast deduplication using FAISS batch search with cosine similarity.

        Strategy:
        - Find k nearest neighbors for each memory using cosine similarity
        - Mark duplicates based on similarity threshold (cosine similarity > 0.75)
        - Keep memory with higher importance
        """
        if len(memories) <= 1:
            return memories
        
        # Convert embeddings to numpy array
        embeddings_matrix = np.array([mem.embedding for mem in memories], dtype=np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_matrix)
        
        # Create FAISS index
        dimension = embeddings_matrix.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity after normalization)
        index.add(embeddings_matrix)
        
        # Find k nearest neighbors for each memory
        k = min(10, len(memories))  # Check up to 10 neighbors
        distances, indices = index.search(embeddings_matrix, k)
        
        # Mark duplicates
        to_remove = set()
        similarity_threshold = 0.75
        
        for i in range(len(memories)):
            if i in to_remove:
                continue
            
            for j_idx in range(1, k):  # Skip first result (self)
                j = indices[i][j_idx]
                similarity = distances[i][j_idx]
                
                if similarity > similarity_threshold and j not in to_remove:
                    # Keep the one with higher importance
                    if memories[i].data.importance >= memories[j].data.importance:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break
        
        # Return deduplicated memories
        return [mem for idx, mem in enumerate(memories) if idx not in to_remove]

    async def delete_all_memories(self, api_key: str, ) -> str:
        """
        Delete all memories for the user.

        Removes the memory file from DIAL bucket and clears the cache
        for the current conversation.
        """
        async with AsyncDial(base_url=self.endpoint, api_key=api_key, api_version="2025-01-01-preview") as dial_client:
            memory_file_path = await self._get_memory_file_path(dial_client)
            await dial_client.delete_file(memory_file_path)
            if memory_file_path in self.cache:
                del self.cache[memory_file_path]
            return "All memories have been successfully deleted."
