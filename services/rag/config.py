"""
services/rag/config.py

Settings for the RAG service. All values are env-overridable via pydantic-settings.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Default to the internal docker hostname so containers use the in-network service
    # when an env var isn't provided. Override with RAG_QDRANT_URL in .env if needed.
    QDRANT_URL: str = "http://qdrant:6333"
    QDRANT_COLLECTION: str = "grid_policies"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    VECTOR_SIZE: int = 384
    POLICIES_DIR: str = "data/policies"
    TOP_K_RETRIEVAL: int = 1


settings = Settings()
