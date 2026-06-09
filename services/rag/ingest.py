"""
services/rag/ingest.py

Chunk policy documents and upsert their embeddings into Qdrant.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from services.rag.config import settings
from services.rag.embedder import embed_texts

_MIN_CHUNK_CHARS = 30
_MAX_CHUNK_TOKENS = 300
_TOKEN_SPLIT = re.compile(r"\s+")
_SECTION_HEADER_RE = re.compile(r"^(Section|SECTION|Article|ARTICLE|Part|PART|Chapter|CHAPTER)\b")
_TXT_SUFFIX = ".txt"


def _split_long_paragraph(paragraph: str) -> list[str]:
    """Split an oversized paragraph on single newlines as a fallback."""
    if not paragraph.strip():
        return []
    lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for line in lines:
        line_tokens = len(_TOKEN_SPLIT.split(line))
        if buffer and buffer_tokens + line_tokens > _MAX_CHUNK_TOKENS:
            chunks.append("\n".join(buffer))
            buffer = [line]
            buffer_tokens = line_tokens
        else:
            buffer.append(line)
            buffer_tokens += line_tokens
    if buffer:
        chunks.append("\n".join(buffer))
    return chunks


def _looks_like_section_header(paragraph: str) -> bool:
    """Detect standalone section titles that should stay with the next paragraph."""
    cleaned = paragraph.strip()
    if not cleaned or "\n" in cleaned or len(cleaned) > 120:
        return False
    if cleaned.endswith((".", ":", ";", "?", "!")):
        return False
    return bool(_SECTION_HEADER_RE.match(cleaned) or cleaned.isupper())


def chunk_document(text: str, doc_id: str, doc_title: str) -> list[dict]:
    """Split a policy document into retrieval chunks.

    Strategy:
      1. Split on blank lines (double newlines) to recover paragraphs.
      2. Merge any paragraph shorter than 60 characters with the paragraph that
         follows it.
      3. Drop paragraphs shorter than _MIN_CHUNK_CHARS characters.
      4. If a paragraph still exceeds _MAX_CHUNK_TOKENS, fall back to
         splitting on single newlines.
    """
    chunks: list[dict] = []
    chunk_idx = 0
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    merged_paragraphs: list[str] = []
    paragraph_idx = 0
    while paragraph_idx < len(paragraphs):
        paragraph = paragraphs[paragraph_idx]
        if len(paragraph) < 60 and paragraph_idx + 1 < len(paragraphs):
            merged_paragraphs.append(f"{paragraph}\n\n{paragraphs[paragraph_idx + 1]}")
            paragraph_idx += 2
            continue
        merged_paragraphs.append(paragraph)
        paragraph_idx += 1

    for cleaned in merged_paragraphs:
        token_count = len(_TOKEN_SPLIT.split(cleaned))
        if token_count > _MAX_CHUNK_TOKENS:
            pieces = _split_long_paragraph(cleaned)
        else:
            pieces = [cleaned]
        for piece in pieces:
            if len(piece) < _MIN_CHUNK_CHARS:
                continue
            chunks.append(
                {
                    "text": piece,
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "chunk_idx": chunk_idx,
                }
            )
            chunk_idx += 1
    return chunks


def _point_id(doc_id: str, chunk_idx: int) -> int:
    """Deterministic 63-bit positive integer ID derived from doc + index."""
    digest = hashlib.sha1(f"{doc_id}:{chunk_idx}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def _ensure_collection(client: QdrantClient, name: str) -> None:
    try:
        client.get_collection(collection_name=name)
        return
    except UnexpectedResponse:
        pass

    client.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(
            size=settings.VECTOR_SIZE,
            distance=qmodels.Distance.COSINE,
        ),
    )
    logger.info("Created Qdrant collection {}", name)


def _read_policy_file(path: Path) -> tuple[str, str]:
    """Return (doc_id, doc_title, ...) — the caller reads the body separately."""
    text = path.read_text(encoding="utf-8")
    doc_id = path.stem
    doc_title = next(
        (line.strip() for line in text.splitlines() if line.strip()),
        doc_id,
    )
    return text, doc_id, doc_title


def ingest_policies(policies_dir: str, collection_name: str) -> int:
    """Chunk, embed, and upsert every .txt file under `policies_dir`.

    Idempotent: point IDs are derived from (doc_id, chunk_idx), so a second
    run replaces the same points and produces the same collection state.
    Returns the total number of chunks ingested.
    """
    client = QdrantClient(url=settings.QDRANT_URL)
    _ensure_collection(client, collection_name)

    base = Path(policies_dir)
    if not base.exists():
        raise FileNotFoundError(f"policies directory not found: {base}")

    txt_files = sorted(base.glob(f"*{_TXT_SUFFIX}"))
    if not txt_files:
        logger.warning("No .txt files found in {}", base)
        return 0

    total_chunks = 0
    documents_processed = 0

    for path in txt_files:
        text, doc_id, doc_title = _read_policy_file(path)
        chunks = chunk_document(text, doc_id, doc_title)
        if not chunks:
            logger.warning("No chunks produced for {}", path.name)
            continue

        vectors = embed_texts([c["text"] for c in chunks])
        points = [
            qmodels.PointStruct(
                id=_point_id(doc_id, chunk["chunk_idx"]),
                vector=vector,
                payload={
                    "text": chunk["text"],
                    "doc_id": chunk["doc_id"],
                    "doc_title": chunk["doc_title"],
                    "chunk_idx": chunk["chunk_idx"],
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        client.upsert(collection_name=collection_name, points=points)
        documents_processed += 1
        total_chunks += len(chunks)
        logger.info(
            "Ingested {} chunks from {} ({} total so far)",
            len(chunks),
            path.name,
            total_chunks,
        )

    logger.info(
        "Finished: {} chunks from {} documents into '{}'",
        total_chunks,
        documents_processed,
        collection_name,
    )
    return total_chunks
