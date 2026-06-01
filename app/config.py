from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

DATA_DIR = os.getenv("DATA_DIR", "data")
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "vector_store")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "healthcare_docs")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_TOP_K = 4
MAX_CHUNK_SIZE = 900
CHUNK_OVERLAP = 180

# Lower distance means better match for cosine distance in Chroma.
HIGH_CONFIDENCE_MAX_DISTANCE = 0.35
MEDIUM_CONFIDENCE_MAX_DISTANCE = 0.55
MAX_RETRIEVAL_DISTANCE = 0.72
