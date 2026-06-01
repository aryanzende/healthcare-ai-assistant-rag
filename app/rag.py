from __future__ import annotations

import re
import uuid
from pathlib import Path

import chromadb
from pypdf import PdfReader

from app.config import (
    CHUNK_OVERLAP,
    COLLECTION_NAME,
    DATA_DIR,
    MAX_CHUNK_SIZE,
    VECTOR_DB_DIR,
)
from app.embeddings import get_embedding

client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _infer_document_type(filename: str) -> str:
    name = filename.lower()
    if "policy" in name:
        return "policy"
    if "insurance" in name:
        return "insurance"
    if "discharge" in name:
        return "discharge"
    if "privacy" in name:
        return "privacy"
    if "appointment" in name:
        return "appointment"
    return "general"


def read_pdf(file_path: Path) -> str:
    try:
        reader = PdfReader(str(file_path))
    except Exception as exc:
        raise RuntimeError(f"Failed PDF extraction for {file_path.name}: {exc}") from exc

    pages = []
    for page in reader.pages:
        page_text = _normalize_text(page.extract_text() or "")
        if page_text:
            pages.append(page_text)

    return "\n\n".join(pages)


def split_text(text: str, chunk_size: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paragraphs:
        return []

    units = []
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
            continue

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    units.append(current)
                current = sentence
        if current:
            units.append(current)

    chunks = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = unit
    if current:
        chunks.append(current)

    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for chunk in chunks[1:]:
        prefix = overlapped[-1][-overlap:]
        overlapped.append(f"{prefix} {chunk}".strip())
    return overlapped


def reset_collection():
    global collection
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(name=COLLECTION_NAME)


def ingest_documents():
    data_path = Path(DATA_DIR)
    if not data_path.exists() or not data_path.is_dir():
        raise FileNotFoundError("Data folder not found.")

    pdf_files = [p for p in data_path.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
    if not pdf_files:
        raise FileNotFoundError("No PDF files found in data folder.")

    reset_collection()

    total_files = 0
    total_chunks = 0

    for pdf_file in pdf_files:
        text = read_pdf(pdf_file)
        if not text.strip():
            continue

        chunks = split_text(text)
        if not chunks:
            continue

        total_files += 1
        for chunk_index, chunk in enumerate(chunks):
            collection.add(
                ids=[str(uuid.uuid4())],
                embeddings=[get_embedding(chunk)],
                documents=[chunk],
                metadatas=[
                    {
                        "source": pdf_file.name,
                        "chunk_index": chunk_index,
                        "document_type": _infer_document_type(pdf_file.name),
                    }
                ],
            )
            total_chunks += 1

    if total_files == 0:
        raise ValueError("All PDFs were empty or unreadable.")

    return {
        "message": "Documents ingested successfully",
        "total_files": total_files,
        "total_chunks": total_chunks,
    }


def confidence_from_distance(distance: float) -> str:
    if distance <= 0.8:
        return "high"
    elif distance <= 1.4:
        return "medium"
    return "low"


def retrieve_context(question: str, top_k: int = 5):
    try:
        question_embedding = get_embedding(question)
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved_chunks = []

        for doc, meta, distance in zip(documents, metadatas, distances):
            if distance > 2.0:
                continue

            retrieved_chunks.append({
                "document": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", 0),
                "chunk": doc,
                "distance": distance,
            })

        return retrieved_chunks
    except Exception as exc:
        raise RuntimeError(f"ChromaDB retrieval error: {exc}") from exc
