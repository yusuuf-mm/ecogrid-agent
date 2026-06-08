"""services/agent/prompts.py

Module-level system prompt for the Gemini-native orchestrator.
Kept concise — Gemini follows structured-output instructions reliably
with shorter prompts than chat-based LangChain agents.

The prompt is split into two parts:
1. PARSE_INTENT — guides Gemini's response_schema extraction of the
   policy query and target date from the user's natural-language prompt.
2. SYNTHESIZE_SUMMARY — guides Gemini's one-sentence summary generation
   from the solver results.
"""
from __future__ import annotations


PARSE_INTENT_PROMPT: str = """You are the intent extractor for the EcoGrid VPP orchestrator.
Given a user's natural-language grid optimization request, extract:
1. A short policy query phrase describing the operational context
   (e.g. "hospital reserve during heatwave", "critical infrastructure minimum SoC").
2. The target ISO date string for the schedule.

Respond only with the structured schema. No extra text.
""".strip()

SYNTHESIZE_SUMMARY_PROMPT: str = """You are a grid-operations summariser for a Virtual Power Plant.
Given the solver status, applied policy, and safety buffer, produce exactly
one plain-language sentence that a grid operator can read aloud.

- If the solver returned OPTIMAL, state the policy and buffer applied.
- If the solver returned INFEASIBLE, explain why no schedule was possible
  and which constraint could not be satisfied.
- If the solver returned ERROR, describe the failure concisely.

Respond only with the structured schema. No extra text.
""".strip()


# Legacy — kept for backward compat with any remaining LangChain references.
# Both the intent parser and summary synthesizer use their own dedicated prompts above.
SYSTEM_PROMPT: str = PARSE_INTENT_PROMPT
