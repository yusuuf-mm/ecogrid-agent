"""
services/rag/retriever.py

Semantic policy retrieval. The retriever embeds a natural-language query,
finds the top-k closest chunks in Qdrant, parses each chunk's buffer
constraint, and returns a list of `PolicyResult` objects (the contract
defined in `shared/contracts.py`).
"""
from __future__ import annotations

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from services.rag.config import settings
from services.rag.embedder import embed_texts
from services.rag.parser import extract_buffer_constraint

# Re-export the shared contract type under its task-spec name so callers
# that expect `PolicyChunk` continue to work. The actual contract type is
# `PolicyResult` (see shared/contracts.py).
from shared.contracts import PolicyResult as PolicyChunk  # noqa: F401  (re-export)


class PolicyRetriever:
    """Top-k semantic retriever over the Qdrant `grid_policies` collection."""

    def __init__(self, collection_name: str | None = None) -> None:
        self.collection = collection_name or settings.QDRANT_COLLECTION
        self.client = QdrantClient(url=settings.QDRANT_URL)

    def retrieve(self, query: str, top_k: int = 1) -> list[PolicyChunk]:
        """Embed `query` and return the top-k `PolicyChunk` results.

        Raises:
            ConnectionError: if Qdrant cannot be reached.
        """
        try:
            vectors = embed_texts([query])
        except Exception as exc:  # noqa: BLE001 — boundary translation
            raise ConnectionError(
                f"Failed to embed query: {exc}"
            ) from exc

        try:
            raw_hits = self.client.search(
                collection_name=self.collection,
                query_vector=vectors[0],
                limit=top_k,
                with_payload=True,
            )
        except UnexpectedResponse as exc:
            raise ConnectionError(
                f"Qdrant at {settings.QDRANT_URL} returned an error: {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 — qdrant-client raises bare exceptions on transport failure
            raise ConnectionError(
                f"Could not reach Qdrant at {settings.QDRANT_URL}: {exc}"
            ) from exc

        results: list[PolicyChunk] = []
        for hit in raw_hits:
            payload = hit.payload or {}
            raw_text = str(payload.get("text", ""))
            parsed = extract_buffer_constraint(raw_text)
            if parsed is None:
                logger.warning(
                    "Could not parse buffer constraint from chunk of {}; defaulting to 0.10",
                    payload.get("doc_id", "<unknown>"),
                )
                constraint = 0.10
                confidence = "fallback"
            else:
                constraint = parsed
                confidence = "parsed"
            results.append(
                PolicyChunk(
                    doc_id=str(payload.get("doc_id", "")),
                    doc_title=str(payload.get("doc_title", "")),
                    raw_chunk=raw_text,
                    constraint_float=constraint,
                    parse_confidence=confidence,
                )
            )
        return results


__all__ = ["PolicyRetriever", "PolicyChunk"]
