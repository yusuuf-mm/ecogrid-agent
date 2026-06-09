"""services/agent/config.py

Settings for the Gemini-native agent. Loaded from environment variables
(or a `.env` file in the project root if present).

`GEMINI_API_KEY` is required — the agent cannot run without it.
`get_settings()` is cached and lazy, so test code can import this module
without an API key and patch the value before building an agent.
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

    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.5-flash"
    AGENT_MAX_ITERATIONS: int = 5
    AGENT_TEMPERATURE: float = 0.0


@lru_cache(maxsize=1)
def get_settings() -> AgentSettings:
    return AgentSettings()
