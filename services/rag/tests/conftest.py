"""
services/rag/tests/conftest.py

Register the `integration` marker so pytest doesn't emit
"PytestUnknownMarkWarning" on integration-marked tests.
"""
from __future__ import annotations


def pytest_configure(config: object) -> None:
    config.addinivalue_line(
        "markers",
        "integration: tests that require a live Qdrant instance (deselect with '-m \"not integration\"')",
    )
