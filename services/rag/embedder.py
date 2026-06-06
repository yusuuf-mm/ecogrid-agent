"""
services/rag/embedder.py

Embedding backend: fastembed with BAAI/bge-small-en-v1.5.

Rationale (per services/rag/CLAUDE.md "Known Environment Constraint"):
  - sentence-transformers requires torch (~2GB). The deployment machine has
    limited disk; the CPU-only torch wheel still costs ~500MB.
  - fastembed is ~50MB, no torch dependency, and ships ONNX runtime.
  - BAAI/bge-small-en-v1.5 emits 384-dim vectors — identical schema to
    all-MiniLM-L6-v2, so the Qdrant collection spec is unchanged.
"""
from __future__ import annotations

from functools import lru_cache

from fastembed import TextEmbedding

from services.rag.config import settings


@lru_cache(maxsize=1)
def _get_model() -> TextEmbedding:
    """Load the embedding model once and cache it for the process lifetime."""
    return TextEmbedding(model_name=settings.EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns a list of 384-dim float vectors.

    The first call downloads the model into the local fastembed cache
    (typically ~/.cache/fastembed). Subsequent calls are in-process.
    """
    model = _get_model()
    return [vec.tolist() for vec in model.embed(texts)]
