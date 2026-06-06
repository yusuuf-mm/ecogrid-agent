"""Tests for services/api/main.py.

Routes are exercised through FastAPI's TestClient (httpx under the hood).
Celery is fully mocked — no broker required.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from services.api.main import app
from shared.contracts import (
    AuditTrail,
    OptimizationResponse,
    TaskStatus,
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_returns_ok(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_post_optimize_returns_202_with_task_id(client: TestClient):
    fake_async = SimpleNamespace(id="task-123")

    with patch(
        "services.api.main.run_optimization_pipeline.apply_async",
        return_value=fake_async,
    ) as mocked:
        response = client.post(
            "/api/v1/optimize",
            json={"prompt": "buy low, sell high", "objective": "MAXIMIZE_PROFIT"},
        )

    assert response.status_code == 202
    body = response.json()
    assert "task_id" in body
    assert body["task_id"] == "task-123"
    assert body["status"] == TaskStatus.QUEUED.value
    mocked.assert_called_once()


def test_post_optimize_rejects_invalid_objective(client: TestClient):
    response = client.post(
        "/api/v1/optimize",
        json={"prompt": "x", "objective": "NOT_A_REAL_OBJECTIVE"},
    )
    assert response.status_code == 422


def test_get_result_pending_returns_queued_status(client: TestClient):
    fake_result = SimpleNamespace(state="PENDING", result=None)
    with patch("services.api.main.AsyncResult", return_value=fake_result):
        response = client.get("/api/v1/results/task-pending")

    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == "task-pending"
    assert body["status"] == TaskStatus.QUEUED.value
    assert body["result"] is None


def test_get_result_success_returns_full_response(client: TestClient):
    success_payload = OptimizationResponse(
        task_id="task-success",
        status=TaskStatus.SUCCESS,
        metrics={"total_profit_usd": 42.0},
        audit=AuditTrail(solver_status="OPTIMAL"),
    ).model_dump(mode="json")

    fake_result = SimpleNamespace(state="SUCCESS", result=success_payload)
    with patch("services.api.main.AsyncResult", return_value=fake_result):
        response = client.get("/api/v1/results/task-success")

    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == "task-success"
    assert body["status"] == TaskStatus.SUCCESS.value
    assert body["result"]["metrics"]["total_profit_usd"] == 42.0
    assert body["error"] is None


def test_get_result_failure_returns_failure_status(client: TestClient):
    fake_result = SimpleNamespace(state="FAILURE", result="boom")
    with patch("services.api.main.AsyncResult", return_value=fake_result):
        response = client.get("/api/v1/results/task-fail")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == TaskStatus.FAILURE.value
    assert body["error"] == "boom"


def test_get_result_started_returns_running_status(client: TestClient):
    fake_result = SimpleNamespace(state="STARTED", result=None)
    with patch("services.api.main.AsyncResult", return_value=fake_result):
        response = client.get("/api/v1/results/task-running")

    assert response.status_code == 200
    assert response.json()["status"] == TaskStatus.RUNNING.value
