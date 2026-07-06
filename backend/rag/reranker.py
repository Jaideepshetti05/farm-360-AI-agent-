from typing import List
from backend.rag.interfaces import RerankerInterface

class CosineReranker(RerankerInterface):
    """Simple Cosine Relevance ranking sorting utility."""
    def rerank(self, query: str, chunks: List[dict]) -> List[dict]:
        # Chunks already contain similarity scores from retrieval query execution.
        return sorted(chunks, key=lambda x: x.get("score", 0.0), reverse=True)
