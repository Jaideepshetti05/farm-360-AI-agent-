import math
from typing import List, Optional, Dict, Any
from loguru import logger

from backend.memory_v2.memory_models import MemoryRecordModel
from backend.memory_v2.memory_store import MemoryStore
from backend.memory_v2.memory_ranker import MemoryRanker
from backend.rag.embedder import GeminiEmbedder

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes the cosine similarity between two float vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    # Guard against zero vectors
    if all(x == 0.0 for x in v1) or all(x == 0.0 for x in v2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)

class MemoryRetriever:
    """Orchestrates candidate retrieval, hybrid matching, scoring, and budget limiting."""
    def __init__(self, store: MemoryStore, embedder: Optional[GeminiEmbedder] = None):
        self.store = store
        self.embedder = embedder or GeminiEmbedder()

    async def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        token_budget: int = 1000,
        min_score: float = 0.05
    ) -> List[MemoryRecordModel]:
        """
        Performs hybrid retrieval:
        1. Generates query embedding.
        2. Filters candidate list by status, category, and tags.
        3. Computes vector similarities and keyword fit.
        4. Re-ranks candidates and returns those matching the token budget.
        """
        # Generate query embedding
        query_vector = await self.embedder.get_embedding(query)
        has_vector = query_vector and not all(x == 0.0 for x in query_vector)
        
        # Get active candidates
        candidates = await self.store.get_all_active_memories(category)
        
        scored_candidates = []
        
        # Tokenize query keywords for keyword search fallback
        query_words = set(query.lower().split())
        
        for c in candidates:
            # Tag filter check
            if tags:
                if not any(t in c.tags for t in tags):
                    continue
            
            # Vector similarity
            similarity = 0.5  # default baseline
            if has_vector and c.embedding:
                similarity = cosine_similarity(query_vector, c.embedding)
            
            # Keyword score boost (if keywords overlap)
            content_lower = c.content.lower()
            keyword_match_count = sum(1 for w in query_words if w in content_lower)
            keyword_boost = 0.0
            if query_words:
                keyword_boost = 0.3 * (keyword_match_count / len(query_words))
            
            # Combine similarity with keyword weight
            combined_match_score = max(similarity, keyword_boost)
            
            # Rank
            final_score = MemoryRanker.score_memory(c, semantic_similarity=combined_match_score)
            
            if final_score >= min_score:
                scored_candidates.append((final_score, c))
                
        # Sort by final score descending
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Apply token budget
        budget_used = 0
        accepted_memories = []
        
        for score, m in scored_candidates:
            # Estimate token count (chars // 4 fallback)
            est_tokens = max(10, len(m.content) // 4)
            if budget_used + est_tokens > token_budget:
                continue
                
            budget_used += est_tokens
            
            # Increment access frequency
            m.access_frequency += 1
            await self.store.update_record(m)
            
            accepted_memories.append(m)
            
        logger.info(f"[MemoryRetriever] Retrieved {len(accepted_memories)} memories (budget: {budget_used}/{token_budget} tokens).")
        return accepted_memories
