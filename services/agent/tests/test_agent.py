"""Tests for services/agent.

All tests in this file MUST run without real LLM calls, real RAG, real
solar forecasting, or real solver execution. Mocks are mandatory.
"""
from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# GEMINI_API_KEY must be set before AgentSettings resolves.
# The settings module is lazy; the env var only needs to be present
# when GridOptimizationAgent is constructed in a test.
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")


# ---------------------------------------------------------------------------
# Tool tests — exercise the three tool functions in isolation.
# Tools are plain functions now (not @tool-decorated LangChain tools).
# ---------------------------------------------------------------------------


def test_tool_query_policies_returns_dict_with_float_buffer():
    """When the RAG service is unreachable, the tool returns a fallback dict
    whose `min_battery_buffer` is a float in [0.0, 1.0].
    """
    from services.agent.tools import tool_query_policies

    with patch(
        "services.rag.retriever.PolicyRetriever.retrieve",
        side_effect=ConnectionError("qdrant not running"),
    ):
        result = tool_query_policies(query="hospital reserve")

    assert isinstance(result, dict)
    assert "min_battery_buffer" in result
    assert isinstance(result["min_battery_buffer"], float)
    assert 0.0 <= result["min_battery_buffer"] <= 1.0
    assert result["source"] == "fallback"
    assert "doc_id" in result
    assert "doc_title" in result
    assert "policy_text" in result


def test_tool_query_policies_uses_rag_output_when_available():
    """When the RAG service returns a real PolicyResult, its fields are
    surfaced in the tool output without modification.
    """
    from shared.contracts import PolicyResult
    from services.agent.tools import tool_query_policies

    rag_result = PolicyResult(
        doc_id="grid_safety_sop_03",
        doc_title="Grid Safety SOP 03",
        raw_chunk="Critical infrastructure: maintain 30% SoC reserve during declared anomalies.",
        constraint_float=0.30,
        parse_confidence="parsed",
    )

    with patch(
        "services.rag.retriever.PolicyRetriever.retrieve",
        return_value=[rag_result],
    ):
        result = tool_query_policies(query="hospital reserve heatwave")

    assert result["doc_id"] == "grid_safety_sop_03"
    assert result["doc_title"] == "Grid Safety SOP 03"
    assert result["policy_text"] == rag_result.raw_chunk
    assert result["min_battery_buffer"] == 0.30
    assert result["source"] == "rag"


def test_tool_forecast_solar_returns_dict_with_24_hour_forecast():
    """The tool returns hourly_forecast_kw of length 24, even when the ML
    service is stubbed (returns zeros in fallback mode).
    """
    from services.agent.tools import tool_forecast_solar

    with patch(
        "services.ml.inference.predictor.forecast_solar_generation",
        side_effect=NotImplementedError,
    ):
        result = tool_forecast_solar(date="2026-07-15")

    assert isinstance(result, dict)
    assert "hourly_forecast_kw" in result
    assert isinstance(result["hourly_forecast_kw"], list)
    assert len(result["hourly_forecast_kw"]) == 24
    for value in result["hourly_forecast_kw"]:
        assert isinstance(value, float)


def test_tool_optimize_grid_with_valid_input_returns_dict_with_status_key():
    """A valid SolverConstraints-shaped dict is deserialised and passed to
    the solver. The returned dict contains a `status` key.
    """
    from services.agent.tools import tool_optimize_grid

    fake_solver_result: dict[str, Any] = {
        "status": "OPTIMAL",
        "schedule": [],
        "total_profit_usd": 0.0,
        "carbon_saved_kg": 0.0,
        "safety_constraints_passed": True,
        "solver_time_ms": 0.5,
        "reason": None,
    }

    solver_input = {
        "solar_forecast_kw": [0.0] * 24,
        "market_prices_kwh": [0.05] * 24,
        "min_battery_buffer": 0.10,
        "battery_capacity_kwh": 1000.0,
        "max_charge_rate_kw": 250.0,
        "initial_soc_kwh": 500.0,
        "objective": "MAXIMIZE_PROFIT",
    }

    with patch(
        "services.solver.engine.optimize_battery_schedule",
        return_value=MagicMock(model_dump=MagicMock(return_value=fake_solver_result)),
    ):
        result = tool_optimize_grid(solver_input=solver_input)

    assert isinstance(result, dict)
    assert "status" in result
    assert result["status"] in {"OPTIMAL", "INFEASIBLE", "ERROR"}


# ---------------------------------------------------------------------------
# GridOptimizationAgent.run() — end-to-end with Gemini + tools mocked.
# ---------------------------------------------------------------------------


def _fake_genai_response(parsed: Any) -> MagicMock:
    """Build a MagicMock that mimics google.genai's response object
    with a populated `.parsed` attribute."""
    resp = MagicMock()
    resp.parsed = parsed
    return resp


def _assert_tool_call_order(response, expected_tools: list[str]) -> None:
    """Assert that audit.agent_tool_calls contains the expected tool names
    in order."""
    tools = [c["tool"] for c in response.audit.agent_tool_calls]
    assert tools == expected_tools, f"Expected {expected_tools}, got {tools}"


def _make_mock_genai_client(monkeypatch, intent_parsed, summary_parsed):
    """Patch GridOptimizationAgent._generate_structured to return
    pre-built _AgentIntent / _AgentSummary instances instead of
    calling the real Gemini API."""
    from services.agent import agent as agent_module

    original = agent_module.GridOptimizationAgent._generate_structured

    def fake_generate_structured(self, schema, system_instruction, user_content):
        if schema is agent_module._AgentIntent:
            return intent_parsed
        if schema is agent_module._AgentSummary:
            return summary_parsed
        return original(self, schema, system_instruction, user_content)

    monkeypatch.setattr(
        agent_module.GridOptimizationAgent,
        "_generate_structured",
        fake_generate_structured,
    )


def test_agent_run_returns_optimization_response_with_non_empty_audit_calls(monkeypatch):
    """agent.run() returns a fully-populated OptimizationResponse whose
    audit.agent_tool_calls records every tool invocation in order.
    """
    from services.agent import agent as agent_module
    from services.agent.agent import GridOptimizationAgent
    from shared.contracts import OptimizationResponse

    # Mock Gemini structured calls
    intent = agent_module._AgentIntent(
        policy_query="hospital reserve heatwave",
        target_date="2026-07-15",
    )
    summary = agent_module._AgentSummary(
        summary="Schedule produced under policy Grid Safety SOP 03 with a 30% SoC reserve enforced.",
        tool_call_count=3,
    )
    _make_mock_genai_client(monkeypatch, intent, summary)

    # Mock the three tool functions to return plausible data
    monkeypatch.setattr(
        agent_module,
        "tool_query_policies",
        lambda query: {
            "doc_id": "grid_safety_sop_03",
            "doc_title": "Grid Safety SOP 03",
            "policy_text": "30% SoC reserve required.",
            "min_battery_buffer": 0.30,
            "retrieval_score": 0.95,
            "source": "rag",
        },
    )
    monkeypatch.setattr(
        agent_module,
        "tool_forecast_solar",
        lambda date: {
            "date": "2026-07-15",
            "hourly_forecast_kw": [0.0] * 24,
            "peak_kw": 0.0,
            "total_kwh": 0.0,
        },
    )
    monkeypatch.setattr(
        agent_module,
        "tool_optimize_grid",
        lambda solver_input: {
            "status": "OPTIMAL",
            "schedule": [
                {
                    "hour": h,
                    "charge_kw": 0.0,
                    "discharge_kw": 0.0,
                    "solar_stored_kw": 0.0,
                    "battery_soc_kwh": 500.0,
                    "action_label": "IDLE",
                    "reason": "stub",
                }
                for h in range(24)
            ],
            "total_profit_usd": 100.0,
            "carbon_saved_kg": 0.0,
            "safety_constraints_passed": True,
            "solver_time_ms": 2.5,
            "reason": None,
        },
    )

    agent = GridOptimizationAgent()
    response = agent.run(
        prompt="Optimize battery tomorrow. Hospital on site. Heatwave expected.",
        objective="MAXIMIZE_PROFIT",
        date="2026-07-15",
    )

    assert isinstance(response, OptimizationResponse)
    assert response.audit.agent_tool_calls, "audit.agent_tool_calls must be non-empty"
    _assert_tool_call_order(response, [
        "tool_query_policies",
        "tool_forecast_solar",
        "tool_optimize_grid",
    ])
    assert response.audit.policy_doc_retrieved == "grid_safety_sop_03"
    assert response.audit.constraint_injected == {"min_battery_buffer": 0.30}
    assert response.audit.solver_status == "OPTIMAL"
    assert response.audit.solver_time_ms == 2.5
    assert len(response.schedule) == 24
    assert response.summary is not None
    assert "30%" in response.summary


def test_agent_run_handles_infeasible_solver_result(monkeypatch):
    """When the solver returns INFEASIBLE, agent.run() still produces a valid
    OptimizationResponse — no exception escapes. The summary is a non-empty
    plain-language explanation of why no schedule was possible.
    """
    from services.agent import agent as agent_module
    from services.agent.agent import GridOptimizationAgent
    from shared.contracts import OptimizationResponse

    intent = agent_module._AgentIntent(
        policy_query="critical infrastructure high reserve",
        target_date="2026-07-15",
    )
    summary = agent_module._AgentSummary(
        summary="No feasible schedule: policy doc_42 requires an 80% SoC reserve that cannot be maintained given the forecast inputs.",
        tool_call_count=3,
    )
    _make_mock_genai_client(monkeypatch, intent, summary)

    monkeypatch.setattr(
        agent_module,
        "tool_query_policies",
        lambda query: {
            "doc_id": "doc_42",
            "doc_title": "Critical Infrastructure SOP",
            "policy_text": "80% SoC reserve required.",
            "min_battery_buffer": 0.80,
            "retrieval_score": 0.9,
            "source": "rag",
        },
    )
    monkeypatch.setattr(
        agent_module,
        "tool_forecast_solar",
        lambda date: {
            "date": "2026-07-15",
            "hourly_forecast_kw": [0.0] * 24,
            "peak_kw": 0.0,
            "total_kwh": 0.0,
        },
    )
    monkeypatch.setattr(
        agent_module,
        "tool_optimize_grid",
        lambda solver_input: {
            "status": "INFEASIBLE",
            "schedule": [],
            "total_profit_usd": 0.0,
            "carbon_saved_kg": 0.0,
            "safety_constraints_passed": False,
            "solver_time_ms": 1.0,
            "reason": "Hospital buffer 0.80 exceeds feasible SoC range given initial charge 0.50.",
        },
    )

    agent = GridOptimizationAgent()
    response = agent.run(
        prompt="hospital heatwave high reserve",
        objective="MAXIMIZE_PROFIT",
        date="2026-07-15",
    )

    assert isinstance(response, OptimizationResponse)
    assert response.audit.solver_status == "INFEASIBLE"
    assert response.status.value == "FAILURE"
    assert response.summary, "summary must be a non-empty string for INFEASIBLE"
    assert isinstance(response.summary, str)
    assert "infeasible" in response.summary.lower() or "no feasible" in response.summary.lower()
    assert response.error is not None
