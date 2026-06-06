"""
services/api/main.py

FastAPI surface for EcoGrid-Agent.

Two real endpoints plus a health check. Route handlers do no business logic:
they validate inbound JSON, enqueue a Celery task, and read result state back
from the Celery result backend. The pipeline itself lives in services/workers/.

Branch: feat/backend-api
"""
from __future__ import annotations

from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from services.api.schemas import (
    OptimizationResponse,
    OptimizeRequest,
    TaskAccepted,
    TaskResult,
    TaskStatus,
)
from services.workers.celery_app import app as celery_app
from services.workers.tasks import run_optimization_pipeline

app = FastAPI(
    title="EcoGrid-Agent",
    version="0.1.0",
    description="Autonomous VPP orchestrator. Phase 1 surface.",
)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/api/v1/optimize",
    response_model=TaskAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["optimize"],
)
def optimize(request: OptimizeRequest) -> JSONResponse:
    """Enqueue an optimization run. Returns immediately with a task id."""
    async_result = run_optimization_pipeline.apply_async(
        args=[request.model_dump(mode="json")],
    )
    payload = TaskAccepted(task_id=async_result.id, status=TaskStatus.QUEUED)
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content=payload.model_dump(mode="json"),
    )


@app.get(
    "/api/v1/results/{task_id}",
    response_model=TaskResult,
    tags=["optimize"],
)
def get_result(task_id: str) -> TaskResult:
    """Poll for an optimization result by Celery task id."""
    async_result = AsyncResult(task_id, app=celery_app)
    state = (async_result.state or "PENDING").upper()

    if state in {"PENDING", "RECEIVED", "STARTED", "RETRY"}:
        return TaskResult(
            task_id=task_id,
            status=TaskStatus.RUNNING if state == "STARTED" else TaskStatus.QUEUED,
        )

    if state == "SUCCESS":
        raw = async_result.result
        if not isinstance(raw, dict):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Worker returned non-dict result",
            )
        response = OptimizationResponse.model_validate(raw)
        return TaskResult(
            task_id=task_id,
            status=response.status,
            result=response,
            error=response.error,
        )

    if state == "FAILURE":
        return TaskResult(
            task_id=task_id,
            status=TaskStatus.FAILURE,
            error=str(async_result.result) if async_result.result else "Task failed",
        )

    return TaskResult(task_id=task_id, status=TaskStatus.QUEUED)
