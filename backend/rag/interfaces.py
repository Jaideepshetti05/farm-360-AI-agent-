from typing import List

class EmbeddingProviderInterface:
    """Interface for embedding vector calculation providers."""
    async def get_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for a single string segment."""
        raise NotImplementedError

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings for a list of string segments."""
        raise NotImplementedError

class RerankerInterface:
    """Interface for contextual relevance ranking controllers."""
    def rerank(self, query: str, chunks: List[dict]) -> List[dict]:
        """Rerank chunks based on relevance to the query."""
        raise NotImplementedError
