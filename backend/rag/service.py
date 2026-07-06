from typing import List, Dict, Any
from backend.rag.loader import DocumentLoader
from backend.rag.retriever import RAGRetriever

class RAGService:
    """Unified service entrypoint for knowledge indexing and retrieval operations."""
    def __init__(self):
        self.loader = DocumentLoader()
        self.retriever = RAGRetriever()

    def index_document(self, file_path: str, metadata: Dict[str, Any] = None) -> str:
        """Parses, embeds, and indexes a file into vector tables."""
        return self.loader.load_and_index(file_path, metadata)

    async def get_context(self, query: str, limit: int = None) -> str:
        """Retrieves and formats matched document context segments."""
        chunks = await self.retriever.retrieve_context(query, limit)
        if not chunks:
            return ""
        
        parts = ["--- KNOWLEDGE BASE REFERENCES ---"]
        for i, chunk in enumerate(chunks):
            parts.append(f"Source Reference [{i+1}]:\n{chunk}")
        parts.append("---------------------------------")
        return "\n\n".join(parts)
