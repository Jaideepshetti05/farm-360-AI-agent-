import os

class RAGConfig:
    EMBEDDING_PROVIDER = os.environ.get("RAG_EMBEDDING_PROVIDER", "gemini")
    EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "models/text-embedding-004")
    
    # Chunking properties
    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_CHUNK_OVERLAP = 100
    
    # Retrieval properties
    TOP_K = int(os.environ.get("RAG_TOP_K", 3))
    SIMILARITY_THRESHOLD = float(os.environ.get("RAG_SIMILARITY_THRESHOLD", 0.50))
    
    # Cache settings
    ENABLE_CACHE = True
