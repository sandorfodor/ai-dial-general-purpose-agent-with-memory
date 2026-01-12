FROM python:3.11
ADD . .
RUN pip install aidial-sdk aidial-client mcp pydantic faiss-cpu sentence-transformers beautifulsoup4 pdfplumber numpy pandas tabulate langchain langchain-text-splitters
CMD ["python", "./run-app.py"] 

