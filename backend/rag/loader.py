import os
from typing import Dict, Any
from backend.services.database_service import UnitOfWork
from backend.models.database import Document, DocumentChunk
from backend.rag.parser import DocumentParser
from backend.rag.chunker import DocumentChunker
from backend.rag.embedder import GeminiEmbedder
from backend.memory.session import run_async_sync

class DocumentLoader:
    def __init__(self):
        self.embedder = GeminiEmbedder()

    def load_and_index(self, file_path: str, metadata: Dict[str, Any] = None) -> str:
        """
        Synchronous pipeline wrapper that parses, chunks, generates embeddings,
        and saves documents/chunks into the database.
        """
        async def _run():
            raw_text = DocumentParser.parse_file(file_path)
            chunks = DocumentChunker.chunk_text(raw_text)
            embeddings = await self.embedder.get_embeddings(chunks)
            
            async with UnitOfWork() as uow:
                meta = metadata or {}
                doc = Document(
                    filename=os.path.basename(file_path),
                    source=meta.get("source"),
                    license=meta.get("license"),
                    version=meta.get("version", "1.0.0"),
                    language=meta.get("language", "en"),
                    region=meta.get("region"),
                    status="active"
                )
                uow.session.add(doc)
                await uow.session.flush()  # get doc.id
                
                for i, (content, embedding) in enumerate(zip(chunks, embeddings)):
                    chunk = DocumentChunk(
                        document_id=doc.id,
                        content=content,
                        embedding=embedding,
                        chunk_index=i
                    )
                    uow.session.add(chunk)
                await uow.session.commit()
                return doc.id
                
        return run_async_sync(_run())
