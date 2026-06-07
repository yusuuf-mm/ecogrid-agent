"""services/agent/config.py

Settings for the LangChain agent. Loaded from environment variables or a
.env file in the project root.

OPENAI_API_KEY is required. The agent cannot run without it.
get_settings() is cached and lazy so tests can import without a key set.
"""
from __future__ import annotations
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_API_BASE: str = ""
    AGENT_MAX_ITERATIONS: int = 5
    AGENT_TEMPERATURE: float = 0.0


@lru_cache(maxsize=1)
def get_settings() -> AgentSettings:
    return AgentSettings()
