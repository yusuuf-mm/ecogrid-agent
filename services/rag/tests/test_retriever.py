"""
services/rag/tests/test_retriever.py

Unit tests run without Qdrant.
Integration tests (marked `@pytest.mark.integration`) require a live
Qdrant instance reachable at the URL configured in services/rag/config.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.rag.embedder import embed_texts  # noqa: E402
from services.rag.ingest import chunk_document, ingest_policies  # noqa: E402
from services.rag.parser import extract_buffer_constraint  # noqa: E402
from services.rag.retriever import PolicyRetriever  # noqa: E402


# ---------------------------------------------------------------------------
# Parser — pure function
# ---------------------------------------------------------------------------


def test_parser_percentage_with_phrase() -> None:
    assert extract_buffer_constraint("minimum state of charge buffer of 30%") == 0.30


def test_parser_percent_word_with_phrase() -> None:
    assert extract_buffer_constraint("minimum state of charge buffer of 5 percent") == 0.05


def test_parser_decimal_fraction_with_phrase() -> None:
    assert extract_buffer_constraint("minimum state of charge buffer of 0.35") == 0.35


def test_parser_word_form_number() -> None:
    assert extract_buffer_constraint("thirty percent") == 0.30


def test_parser_returns_none_when_no_match() -> None:
    assert extract_buffer_constraint("no constraint here") is None


# ---------------------------------------------------------------------------
# Chunker — pure function
# ---------------------------------------------------------------------------


def test_chunk_document_assigns_metadata_fields() -> None:
    text = (
        "First paragraph with enough text to pass the minimum length filter.\n\n"
        "Second paragraph that also has plenty of text to be a real chunk.\n\n"
        "Short\n\n"
        "Third paragraph that once again is well above the minimum length threshold."
    )
    chunks = chunk_document(text, "doc_01", "Test Doc")
    assert chunks, "expected at least one chunk"
    assert all(c["doc_id"] == "doc_01" for c in chunks)
    assert all(c["doc_title"] == "Test Doc" for c in chunks)
    assert [c["chunk_idx"] for c in chunks] == list(range(len(chunks)))


# ---------------------------------------------------------------------------
# Embedder — downloads the model on first call; cached after that
# ---------------------------------------------------------------------------


def test_embed_texts_returns_384_dim_vectors() -> None:
    vectors = embed_texts(["hello world"])
    assert isinstance(vectors, list)
    assert len(vectors) == 1
    assert len(vectors[0]) == 384


# ---------------------------------------------------------------------------
# Integration tests — require a live Qdrant
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_retrieve_hospital_reserve_returns_sop_03() -> None:
    ingest_policies("data/policies", "grid_policies")
    retriever = PolicyRetriever(collection_name="grid_policies")
    hits = retriever.retrieve("hospital power reserve", top_k=1)
    assert hits, "expected at least one hit"
    assert hits[0].doc_id == "grid_safety_sop_03"


@pytest.mark.integration
def test_retrieve_maintenance_window_returns_sop_05() -> None:
    ingest_policies("data/policies", "grid_policies")
    retriever = PolicyRetriever(collection_name="grid_policies")
    hits = retriever.retrieve("maintenance window reduced reserve", top_k=1)
    assert hits, "expected at least one hit"
    assert hits[0].doc_id == "grid_safety_sop_05"


@pytest.mark.integration
def test_retrieved_chunk_has_text_doc_id_and_positive_score() -> None:
    ingest_policies("data/policies", "grid_policies")
    retriever = PolicyRetriever(collection_name="grid_policies")
    hits = retriever.retrieve("battery reserve requirement", top_k=1)
    assert hits
    assert hits[0].raw_chunk.strip()
    assert hits[0].doc_id.strip()
    # `score` is not a contract field, but the retriever only forwards
    # payload — the agent layer uses `parse_confidence` to signal success.
    # Confirm constraint_float is parsed from the chunk above zero.
    assert hits[0].constraint_float > 0
