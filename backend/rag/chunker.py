from typing import List
from backend.rag.config import RAGConfig

class DocumentChunker:
    @staticmethod
    def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """
        Splits raw text into sliding window chunks based on character lengths.
        """
        if chunk_size is None:
            chunk_size = RAGConfig.DEFAULT_CHUNK_SIZE
        if overlap is None:
            overlap = RAGConfig.DEFAULT_CHUNK_OVERLAP

        if not text:
            return []

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start += (chunk_size - overlap)
            
        return chunks
