from __future__ import annotations

import re
import logging
from pathlib import Path

from pinecone import Pinecone
from pypdf import PdfReader

from app.config import (
    CHUNK_OVERLAP,
    DATA_DIR,
    DEFAULT_TOP_K,
    EMBEDDING_DIMENSION,
    HIGH_CONFIDENCE_MAX_DISTANCE,
    MAX_CHUNK_SIZE,
    MAX_RETRIEVAL_DISTANCE,
    MEDIUM_CONFIDENCE_MAX_DISTANCE,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
)
from app.embeddings import get_embedding

logger = logging.getLogger(__name__)

_pinecone_index = None
MIN_PAGE_TEXT_LENGTH = 20


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


def read_pdf(file_path: Path) -> tuple[str, list[str]]:
    warnings = []

    try:
        reader = PdfReader(str(file_path))
    except Exception as exc:
        return "", [f"Failed PDF extraction for {file_path.name}, skipped: {exc}"]

    pages = []
    for page_index, page in enumerate(reader.pages):
        try:
            page_text = _normalize_text(page.extract_text() or "")
        except Exception as exc:
            page_text = ""
            warnings.append(
                f"pypdf extraction failed for {file_path.name} page {page_index + 1}, skipped: {exc}"
            )

        if len(page_text) < MIN_PAGE_TEXT_LENGTH:
            page_text = ""
            warnings.append(f"No readable text found in {file_path.name} page {page_index + 1}, skipped")

        if page_text.strip():
            pages.append(page_text)

    if not pages:
        warnings.append(f"No readable text found in {file_path.name}, skipped")

    return "\n\n".join(pages), warnings


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


def _get_pinecone_index():
    global _pinecone_index

    if _pinecone_index is None:
        if not PINECONE_API_KEY:
            raise ValueError("Missing PINECONE_API_KEY.")
        if not PINECONE_INDEX_NAME:
            raise ValueError("Missing PINECONE_INDEX_NAME.")
        if not PINECONE_NAMESPACE:
            raise ValueError("Missing PINECONE_NAMESPACE.")

        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            _pinecone_index = pc.Index(PINECONE_INDEX_NAME)
            logger.info(
                "Connected to Pinecone index '%s' namespace '%s'.",
                PINECONE_INDEX_NAME,
                PINECONE_NAMESPACE,
            )
        except Exception as exc:
            raise RuntimeError(f"Pinecone connection error: {exc}") from exc

    return _pinecone_index


def _get_namespace_vector_count(index) -> int:
    stats = index.describe_index_stats()
    namespaces = getattr(stats, "namespaces", None)
    if namespaces is None and isinstance(stats, dict):
        namespaces = stats.get("namespaces", {})
    namespaces = namespaces or {}

    namespace_stats = namespaces.get(PINECONE_NAMESPACE)
    if not namespace_stats:
        return 0

    vector_count = getattr(namespace_stats, "vector_count", None)
    if vector_count is None and isinstance(namespace_stats, dict):
        vector_count = namespace_stats.get("vector_count", 0)

    return int(vector_count or 0)


def _batched(items: list, batch_size: int = 100):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _vector_id_for(pdf_file: Path, chunk_index: int) -> str:
    return f"{pdf_file.stem}-{chunk_index}"


def _get_fetched_vectors(fetch_response) -> dict:
    vectors = getattr(fetch_response, "vectors", None)
    if vectors is None and isinstance(fetch_response, dict):
        vectors = fetch_response.get("vectors", {})
    return vectors or {}


def _is_pdf_already_ingested(index, pdf_file: Path) -> bool:
    first_chunk_id = _vector_id_for(pdf_file, 0)
    response = index.fetch(ids=[first_chunk_id], namespace=PINECONE_NAMESPACE)
    return first_chunk_id in _get_fetched_vectors(response)


def ingest_documents():
    data_path = Path(DATA_DIR)
    if not data_path.exists() or not data_path.is_dir():
        raise FileNotFoundError("Data folder not found.")

    pdf_files = sorted(
        [p for p in data_path.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"],
        key=lambda p: p.name.lower(),
    )
    if not pdf_files:
        raise FileNotFoundError("No PDF files found in data folder.")

    index = _get_pinecone_index()
    existing_vector_count = _get_namespace_vector_count(index)
    new_files_ingested = 0
    skipped_files = 0
    new_chunks_added = 0
    warnings = []
    vectors = []

    for pdf_file in pdf_files:
        if _is_pdf_already_ingested(index, pdf_file):
            skipped_files += 1
            logger.info("Skipping already ingested PDF '%s'.", pdf_file.name)
            continue

        text, pdf_warnings = read_pdf(pdf_file)
        warnings.extend(pdf_warnings)
        if not text.strip():
            continue

        chunks = split_text(text)
        if not chunks:
            warnings.append(f"No chunks created for {pdf_file.name}, skipped")
            continue

        new_files_ingested += 1
        for chunk_index, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            if len(embedding) != EMBEDDING_DIMENSION:
                raise RuntimeError(
                    f"Embedding dimension mismatch: expected {EMBEDDING_DIMENSION}, got {len(embedding)}."
                )

            vectors.append(
                {
                    "id": _vector_id_for(pdf_file, chunk_index),
                    "values": embedding,
                    "metadata": {
                        "document": pdf_file.name,
                        "chunk_index": chunk_index,
                        "chunk": chunk,
                        "document_type": _infer_document_type(pdf_file.name),
                    },
                }
            )
            new_chunks_added += 1

    if new_files_ingested == 0:
        if skipped_files == len(pdf_files):
            response = {
                "message": "All documents already ingested in Pinecone",
                "total_files": len(pdf_files),
                "new_files_ingested": 0,
                "skipped_files": skipped_files,
                "new_chunks_added": 0,
                "total_chunks": existing_vector_count,
            }
            if warnings:
                response["warnings"] = warnings
            return response

        response = {
            "message": "No new readable documents ingested",
            "total_files": len(pdf_files),
            "new_files_ingested": 0,
            "skipped_files": skipped_files,
            "new_chunks_added": 0,
            "total_chunks": existing_vector_count,
        }
        if warnings:
            response["message"] = "No new readable documents ingested; warnings generated"
            response["warnings"] = warnings
        return response

    for batch in _batched(vectors):
        index.upsert(vectors=batch, namespace=PINECONE_NAMESPACE)

    total_chunks = max(
        _get_namespace_vector_count(index),
        existing_vector_count + new_chunks_added,
    )

    logger.info(
        "Upserted %s new vectors from %s new PDF files to Pinecone namespace '%s'.",
        new_chunks_added,
        new_files_ingested,
        PINECONE_NAMESPACE,
    )

    response = {
        "message": "Documents ingested with warnings" if warnings else "Documents ingested successfully",
        "total_files": len(pdf_files),
        "new_files_ingested": new_files_ingested,
        "skipped_files": skipped_files,
        "new_chunks_added": new_chunks_added,
        "total_chunks": total_chunks,
    }
    if warnings:
        response["warnings"] = warnings

    return response


def confidence_from_distance(distance: float) -> str:
    if distance <= HIGH_CONFIDENCE_MAX_DISTANCE:
        return "high"
    elif distance <= MEDIUM_CONFIDENCE_MAX_DISTANCE:
        return "medium"
    return "low"


def retrieve_context(question: str, top_k: int = DEFAULT_TOP_K):
    try:
        index = _get_pinecone_index()
        question_embedding = get_embedding(question)
        results = index.query(
            vector=question_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=PINECONE_NAMESPACE,
        )

        matches = getattr(results, "matches", None)
        if matches is None and isinstance(results, dict):
            matches = results.get("matches", [])
        matches = matches or []

        retrieved_chunks = []

        for match in matches:
            metadata = getattr(match, "metadata", None)
            if metadata is None and isinstance(match, dict):
                metadata = match.get("metadata", {})
            metadata = metadata or {}

            score = getattr(match, "score", None)
            if score is None and isinstance(match, dict):
                score = match.get("score", 0.0)
            score = float(score or 0.0)
            distance = 1 - score

            if distance > MAX_RETRIEVAL_DISTANCE:
                continue

            retrieved_chunks.append({
                "document": metadata.get("document", "unknown"),
                "chunk_index": metadata.get("chunk_index", 0),
                "chunk": metadata.get("chunk", ""),
                "distance": distance,
            })

        logger.info(
            "Pinecone query returned %s matches; %s passed retrieval threshold.",
            len(matches),
            len(retrieved_chunks),
        )

        return retrieved_chunks
    except Exception as exc:
        raise RuntimeError(f"Pinecone retrieval error: {exc}") from exc
