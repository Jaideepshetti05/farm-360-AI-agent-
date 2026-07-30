from typing import List, Tuple
from loguru import logger
import uuid
import datetime

from backend.memory_v2.memory_models import MemoryRecordModel
from backend.memory_v2.memory_store import MemoryStore
from backend.memory_v2.memory_retriever import cosine_similarity
from backend.provider_manager import provider_manager

class MemorySummarizer:
    """Handles consolidation of duplicate records and summarizing older conversation sessions."""
    def __init__(self, store: MemoryStore):
        self.store = store

    async def merge_duplicates(self, threshold: float = 0.92) -> int:
        """
        Scans all active memories and merges those that are semantically identical (similarity > threshold).
        Deduplicates by keeping the record with higher importance / access count, and tags union.
        """
        memories = await self.store.get_all_active_memories()
        merged_count = 0
        
        # Sort by length of content to process larger content first
        memories.sort(key=lambda x: len(x.content), reverse=True)
        
        to_delete = set()
        
        for i in range(len(memories)):
            if memories[i].id in to_delete:
                continue
            
            for j in range(i + 1, len(memories)):
                if memories[j].id in to_delete:
                    continue
                
                # Check vector similarity if both have embeddings
                sim = 0.0
                if memories[i].embedding and memories[j].embedding:
                    sim = cosine_similarity(memories[i].embedding, memories[j].embedding)
                elif memories[i].content.lower() == memories[j].content.lower():
                    sim = 1.0
                
                if sim >= threshold:
                    # Merge j into i
                    logger.info(f"[Summarizer] Merging duplicate memory {memories[j].id} into {memories[i].id} (similarity: {sim:.2f})")
                    memories[i].access_frequency += memories[j].access_frequency
                    memories[i].importance = max(memories[i].importance, memories[j].importance)
                    memories[i].tags = list(set(memories[i].tags + memories[j].tags))
                    
                    # Update record i
                    await self.store.update_record(memories[i])
                    
                    # Mark j as deleted
                    to_delete.add(memories[j].id)
                    await self.store.update_status(memories[j].id, "deleted")
                    merged_count += 1
                    
        return merged_count

    async def generate_summary(self, memories: List[MemoryRecordModel]) -> str:
        """Asynchronously summarizes a list of memories using the active LLM provider."""
        if not memories:
            return ""
            
        content_block = "\n".join([f"- {m.content}" for m in memories])
        
        prompt_messages = [
            {"role": "system", "content": "You are a concise agricultural data compiler. Summarize the user's farm profile highlights and notes into a single clear, actionable paragraph (max 3 sentences). Output PROSE only."},
            {"role": "user", "content": f"Summarize these logs:\n{content_block}"}
        ]
        
        req_id = f"consolidate-{uuid.uuid4().hex[:8]}"
        
        try:
            tokens = []
            # We use stream_completion_async and accumulate the tokens
            async for token in provider_manager.stream_completion_async(prompt_messages, req_id):
                tokens.append(token)
            
            summary = "".join(tokens).strip()
            if summary:
                return summary
        except Exception as e:
            logger.warning(f"[Summarizer] LLM summary failed: {e}. Falling back to list combination.")
            
        # Fallback summary
        fallback = "; ".join([m.content[:50] + "..." for m in memories[:3]])
        return f"Summary of logs: {fallback}"
