"""
services/rag/retriever.py

Policy retrieval stub. Branch: feat/vector-rag
Read CLAUDE.md and services/rag/CLAUDE.md before implementing.
"""
from shared.contracts import PolicyQuery, PolicyResult


def query_grid_policies(query: PolicyQuery) -> PolicyResult:
    """TODO: implement Qdrant semantic search + LLM constraint parsing."""
    raise NotImplementedError("Implement in feat/vector-rag branch")
