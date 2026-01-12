import json
from typing import Any

from task.tools.base import BaseTool
from task.tools.memory._models import MemoryData
from task.tools.memory.memory_store import LongTermMemoryStore
from task.tools.models import ToolCallParams


class SearchMemoryTool(BaseTool):
    """
    Tool for searching long-term memories about the user.

    Performs semantic search over stored memories to find relevant information.
    """

    def __init__(self, memory_store: LongTermMemoryStore):
        self.memory_store = memory_store


    @property
    def name(self) -> str:
        return "search_memory"

    @property
    def description(self) -> str:
        return ("Search long-term memories about the user using semantic similarity. Use this to recall relevant information "
                "from previous conversations. Search when you need context about the user's preferences, background, goals, "
                "or any previously stored information. This helps provide personalized and contextually aware responses.")

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Can be a question or keywords to find relevant memories"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of most relevant memories to return.",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5
                }
            },
            "required": ["query"]
        }


    async def _execute(self, tool_call_params: ToolCallParams) -> str:
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        query = arguments["query"]
        top_k = arguments.get("top_k", 5)
        
        results = await self.memory_store.search_memories(
            api_key=tool_call_params.api_key,
            query=query,
            top_k=top_k
        )
        
        if not results:
            final_result = "No memories found."
        else:
            final_result = "## Found Memories\n\n"
            for idx, memory_data in enumerate(results, 1):
                final_result += f"### Memory {idx}\n"
                final_result += f"**Content**: {memory_data.content}\n"
                final_result += f"**Category**: {memory_data.category}\n"
                if memory_data.topics:
                    final_result += f"**Topics**: {', '.join(memory_data.topics)}\n"
                final_result += "\n"
        
        stage = tool_call_params.stage
        stage.append_content(final_result)
        
        return final_result
