from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

DATA_DIR = os.getenv("DATA_DIR", "data")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag").strip()
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "healthcare-docs").strip()

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

DEFAULT_TOP_K = 3
MAX_CHUNK_SIZE = 900
CHUNK_OVERLAP = 180

# Pinecone cosine similarity is converted to distance with 1 - score.
HIGH_CONFIDENCE_MAX_DISTANCE = 0.35
MEDIUM_CONFIDENCE_MAX_DISTANCE = 0.55
MAX_RETRIEVAL_DISTANCE = 0.72
