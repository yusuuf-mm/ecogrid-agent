"""services/agent/config.py

Settings for the LangChain agent. Loaded from environment variables (or a
`.env` file in the project root if present).

`OPENAI_API_KEY` is required — there is no default. The agent cannot run
without it. The other fields have safe defaults for an engineering system
that must behave deterministically.

`get_settings()` is cached and lazy. The settings object is only built on
the first call, so test code can import this module without an
OPENAI_API_KEY in the environment and then patch the value (or call
`get_settings.cache_clear()` + re-set the env) before any LLM is built.
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
    AGENT_MAX_ITERATIONS: int = 5
    AGENT_TEMPERATURE: float = 0.0


@lru_cache(maxsize=1)
def get_settings() -> AgentSettings:
    return AgentSettings()
