from __future__ import annotations

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL_NAME

_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_embedding(text: str):
    model = get_embedding_model()
    return model.encode(text).tolist()
