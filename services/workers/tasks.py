"""
services/workers/tasks.py

The single Celery task that runs the full optimization pipeline.

Phase 3 scope: deserialize the inbound request, hand it to the LangChain
agent, and serialize the resulting OptimizationResponse.

Branch: feat/agent-core
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from loguru import logger

from services.agent.agent import GridOptimizationAgent
from services.workers.celery_app import app
from shared.contracts import OptimizationRequest, OptimizationResponse, TaskStatus


def _tomorrow_iso() -> str:
    return (date.today() + timedelta(days=1)).isoformat()


def _run_pipeline(task_id: str, request: OptimizationRequest) -> OptimizationResponse:
    """Pure function form of the pipeline. Easy to unit-test without Celery."""
    logger.info(
        "optimization.start task_id={} prompt={!r} objective={}",
        task_id, request.prompt, request.objective.value,
    )

    agent = GridOptimizationAgent()
    result = agent.run(
        prompt=request.prompt,
        objective=request.objective.value,
        date=request.date or _tomorrow_iso(),
    )
    return result


@app.task(
    bind=True,
    name="services.workers.tasks.run_optimization_pipeline",
    max_retries=1,
    default_retry_delay=30,
)
def run_optimization_pipeline(self, request_dict: dict[str, Any]) -> dict[str, Any]:
    """Celery entrypoint. Always returns a dict (json-serializable)."""
    task_id = self.request.id or "unknown"
    try:
        request = OptimizationRequest.model_validate(request_dict)
    except Exception as exc:  # noqa: BLE001 - want to capture any validation error
        logger.error("optimization.invalid_request task_id={} error={}", task_id, exc)
        return OptimizationResponse(
            task_id=task_id,
            status=TaskStatus.FAILURE,
            error=f"Invalid request: {exc}",
        ).model_dump(mode="json")

    try:
        response = _run_pipeline(task_id, request)
    except Exception as exc:  # noqa: BLE001 - last-resort guard before Celery retry
        logger.exception("optimization.unhandled task_id={}", task_id)
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            return OptimizationResponse(
                task_id=task_id,
                status=TaskStatus.FAILURE,
                error=f"Pipeline failed after retry: {exc}",
            ).model_dump(mode="json")

    return response.model_dump(mode="json")
