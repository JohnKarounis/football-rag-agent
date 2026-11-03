from pathlib import Path

# Paths 
ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "data" / "corpus"
CHROMA_DIR = ROOT / "chroma_db"

# LM Studio
LMSTUDIO_URL = "http://localhost:1234/v1"
LMSTUDIO_API_KEY = "lm-studio"
CHAT_MODEL = "qwen/qwen3.8-27b"
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"

# Ingestion 
COLLECTION = "football_nomic"
CHUNK_SIZE = 100                # in words, not characters 
CHUNK_OVERLAP = 20
SKIP_SECTIONS = ["See also", "References", "External links", "Notes"]

# Retrieval 
TOP_K = 10
