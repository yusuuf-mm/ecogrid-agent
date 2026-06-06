"""Tests for services/workers/tasks.py.

Phase 3 scope: the stub solver path is gone. The task delegates to
`GridOptimizationAgent().run()`, so tests mock the agent class.
These tests run the Celery task in eager mode (no broker, no worker).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from services.workers import tasks as tasks_module
from services.workers.celery_app import app as celery_app
from services.workers.tasks import _run_pipeline, _tomorrow_iso, run_optimization_pipeline
from shared.contracts import (
    AuditTrail,
    OptimizationRequest,
    OptimizationResponse,
    ScheduleHour,
    SolverObjective,
    TaskStatus,
)


@pytest.fixture(autouse=True)
def _eager_celery():
    """Run tasks synchronously, in-process, with no broker."""
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True
    yield
    celery_app.conf.task_always_eager = False
    celery_app.conf.task_eager_propagates = False


def _fake_response(task_id: str) -> OptimizationResponse:
    return OptimizationResponse(
        task_id=task_id,
        status=TaskStatus.SUCCESS,
        summary="Schedule produced under policy doc_x with 10% SoC reserve.",
        schedule=[
            ScheduleHour(
                hour=h,
                charge_kw=0.0,
                discharge_kw=0.0,
                solar_stored_kw=0.0,
                battery_soc_kwh=500.0,
                action_label="IDLE",
                reason="stub",
            )
            for h in range(24)
        ],
        metrics={
            "total_profit_usd": 12.34,
            "carbon_saved_kg": 0.0,
            "safety_constraints_passed": True,
        },
        audit=AuditTrail(
            policy_doc_retrieved="doc_x",
            policy_raw_text="stub policy text",
            constraint_injected={"min_battery_buffer": 0.10},
            solar_forecast_used=[0.0] * 24,
            market_prices_used=[0.05] * 24,
            solver_status="OPTIMAL",
            solver_time_ms=1.5,
            agent_tool_calls=[
                {
                    "tool": "tool_query_policies",
                    "input": {"query": "hospital reserve"},
                    "output": {"doc_id": "doc_x"},
                    "duration_ms": 5.0,
                }
            ],
        ),
    )


def test_run_optimization_with_mocked_agent_returns_success():
    request_dict = OptimizationRequest(
        prompt="optimize tomorrow",
        objective=SolverObjective.MAXIMIZE_PROFIT,
    ).model_dump(mode="json")

    with patch.object(tasks_module, "GridOptimizationAgent") as mock_agent_class:
        mock_agent_class.return_value.run.return_value = _fake_response("celery-1")
        result = run_optimization_pipeline.apply(args=[request_dict]).get()

    assert isinstance(result, dict)
    assert result["status"] == TaskStatus.SUCCESS.value
    assert result["audit"]["policy_doc_retrieved"] == "doc_x"
    assert result["audit"]["solver_status"] == "OPTIMAL"
    assert result["metrics"]["safety_constraints_passed"] is True
    assert len(result["schedule"]) == 24
    assert len(result["audit"]["agent_tool_calls"]) == 1


def test_run_optimization_passes_request_fields_to_agent():
    request = OptimizationRequest(
        prompt="hospital heatwave",
        objective=SolverObjective.MINIMIZE_CARBON,
        date="2026-07-15",
    )

    with patch.object(tasks_module, "GridOptimizationAgent") as mock_agent_class:
        mock_agent_class.return_value.run.return_value = _fake_response("celery-2")
        _run_pipeline("celery-2", request)
        mock_agent_class.return_value.run.assert_called_once_with(
            prompt="hospital heatwave",
            objective="MINIMIZE_CARBON",
            date="2026-07-15",
        )


def test_run_optimization_defaults_to_tomorrow_when_date_omitted():
    request = OptimizationRequest(prompt="anytime", objective=SolverObjective.MAXIMIZE_PROFIT)
    expected = _tomorrow_iso()

    with patch.object(tasks_module, "GridOptimizationAgent") as mock_agent_class:
        mock_agent_class.return_value.run.return_value = _fake_response("celery-3")
        _run_pipeline("celery-3", request)
        _, kwargs = mock_agent_class.return_value.run.call_args
        assert kwargs["date"] == expected


def test_run_optimization_with_invalid_request_returns_failure_dict():
    result = run_optimization_pipeline.apply(args=[{"not_a_real_field": True}]).get()

    assert isinstance(result, dict)
    assert result["status"] == TaskStatus.FAILURE.value
    assert "Invalid request" in (result.get("error") or "")


def test_run_optimization_surfaces_agent_failure():
    request_dict = OptimizationRequest(prompt="any").model_dump(mode="json")
    failure = OptimizationResponse(
        task_id="celery-4",
        status=TaskStatus.FAILURE,
        error="Solver raised",
    )

    with patch.object(tasks_module, "GridOptimizationAgent") as mock_agent_class:
        mock_agent_class.return_value.run.return_value = failure
        result = run_optimization_pipeline.apply(args=[request_dict]).get()

    assert result["status"] == TaskStatus.FAILURE.value
    assert result["error"] == "Solver raised"
