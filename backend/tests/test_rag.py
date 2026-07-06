import sys
import os
import asyncio
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.core.database import engine
from backend.models.database import Base
from backend.rag.service import RAGService
from backend.rag.chunker import DocumentChunker
from backend.rag.parser import DocumentParser

class TestRAGSystem(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Create schemas in clean memory database
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def asyncTearDown(self):
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    def test_chunker_sliding_window(self):
        text = "This is a dummy reference string that needs to be chunked into parts."
        chunks = DocumentChunker.chunk_text(text, chunk_size=20, overlap=5)
        self.assertTrue(len(chunks) > 0)
        self.assertIn("This is a dummy", chunks[0])

    def test_parser_markdown(self):
        dummy_path = "test_manual.md"
        with open(dummy_path, "w", encoding="utf-8") as f:
            f.write("# Rice Sowing Manual\nUse Tricyclazole 75WP for Leaf Blast.")
            
        try:
            parsed = DocumentParser.parse_file(dummy_path)
            self.assertIn("Rice Sowing Manual", parsed)
        finally:
            if os.path.exists(dummy_path):
                os.remove(dummy_path)

    async def test_indexing_and_retrieval(self):
        dummy_path = "test_manual.txt"
        with open(dummy_path, "w", encoding="utf-8") as f:
            f.write("Tricyclazole is highly recommended for Leaf Blast treatment.")
            
        try:
            rag_service = RAGService()
            doc_id = rag_service.index_document(dummy_path, {"source": "ICAR", "language": "en"})
            self.assertIsNotNone(doc_id)
            
            context = await rag_service.get_context("How to treat Leaf Blast?", limit=1)
            self.assertIn("Tricyclazole", context)
            self.assertIn("KNOWLEDGE BASE REFERENCES", context)
        finally:
            if os.path.exists(dummy_path):
                os.remove(dummy_path)

if __name__ == "__main__":
    unittest.main()
