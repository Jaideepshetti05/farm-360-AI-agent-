from typing import List, Dict, Any
import numpy as np
from sqlalchemy import select
from backend.services.database_service import UnitOfWork
from backend.models.database import DocumentChunk, HAS_PGVECTOR
from backend.rag.embedder import GeminiEmbedder
from backend.rag.reranker import CosineReranker
from backend.rag.config import RAGConfig

class RAGRetriever:
    def __init__(self):
        self.embedder = GeminiEmbedder()
        self.reranker = CosineReranker()

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Manually calculate cosine similarity for fallback search."""
        arr_a = np.array(a)
        arr_b = np.array(b)
        norm_a = np.linalg.norm(arr_a)
        norm_b = np.linalg.norm(arr_b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(arr_a, arr_b) / (norm_a * norm_b))

    async def retrieve_context(self, query: str, limit: int = None) -> List[str]:
        """
        Retrieves matching chunks using pgvector if available,
        or falling back to Python-based similarity matching on SQLite.
        """
        if limit is None:
            limit = RAGConfig.TOP_K

        query_vector = await self.embedder.get_embedding(query)
        
        async def _search():
            async with UnitOfWork() as uow:
                if HAS_PGVECTOR:
                    stmt = select(DocumentChunk).order_by(
                        DocumentChunk.embedding.cosine_distance(query_vector)
                    ).limit(limit)
                    res = await uow.session.execute(stmt)
                    chunks = res.scalars().all()
                    
                    results = []
                    for c in chunks:
                        results.append({
                            "content": c.content,
                            "score": 1.0
                        })
                    return results
                else:
                    stmt = select(DocumentChunk)
                    res = await uow.session.execute(stmt)
                    chunks = res.scalars().all()
                    
                    scored = []
                    is_zero_query = (np.linalg.norm(np.array(query_vector)) == 0.0)
                    
                    for c in chunks:
                        vector_list = c.embedding
                        score = 0.0
                        
                        # Full Text Search fallback for zero embeddings (mock/offline runs)
                        if is_zero_query or (vector_list and np.linalg.norm(np.array(vector_list)) == 0.0):
                            q_words = set(query.lower().replace("?", "").replace(",", "").split())
                            c_words = set(c.content.lower().replace(".", "").replace(",", "").split())
                            if q_words & c_words:
                                score = float(len(q_words & c_words) / len(q_words))
                            else:
                                score = 0.0
                        else:
                            if vector_list and isinstance(vector_list, list):
                                score = self._cosine_similarity(query_vector, vector_list)
                                
                        scored.append({
                            "content": c.content,
                            "score": score
                        })
                    
                    scored = [s for s in scored if s["score"] >= RAGConfig.SIMILARITY_THRESHOLD or (is_zero_query and s["score"] > 0.0)]
                    return scored[:limit]

        try:
            matched_chunks = await _search()
        except Exception as e:
            return []
            
        reranked = self.reranker.rerank(query, matched_chunks)
        return [c["content"] for c in reranked]
