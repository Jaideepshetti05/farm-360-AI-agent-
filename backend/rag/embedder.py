import os
from typing import List
from google import genai
from backend.rag.interfaces import EmbeddingProviderInterface
from backend.provider_manager import provider_manager
from backend.rag.config import RAGConfig

class GeminiEmbedder(EmbeddingProviderInterface):
    def __init__(self):
        self.model = RAGConfig.EMBEDDING_MODEL

    def _get_client(self) -> genai.Client:
        """Retrieves genai.Client initialized with active Gemini key."""
        for entry in provider_manager.keys:
            if entry.provider == "gemini" and entry.is_available:
                return genai.Client(api_key=entry.key)
        key = os.environ.get("GOOGLE_API_KEY")
        return genai.Client(api_key=key)

    async def get_embedding(self, text: str) -> List[float]:
        try:
            client = self._get_client()
            response = client.models.embed_content(
                model=self.model,
                contents=text
            )
            if response.embeddings:
                return response.embeddings[0].values
        except Exception as e:
            # Fallback to zero embeddings vector
            return [0.0] * 768
        return [0.0] * 768

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            emb = await self.get_embedding(text)
            results.append(emb)
        return results
