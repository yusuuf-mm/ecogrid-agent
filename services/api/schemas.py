"""
services/api/schemas.py

API-facing request/response models. Anything that crosses the HTTP boundary
serializes through one of these.

`OptimizeRequest` aliases `shared.contracts.OptimizationRequest` so the API
surface has a name that reads naturally next to `TaskAccepted` / `TaskResult`,
without duplicating the field definitions.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from shared.contracts import (
    OptimizationRequest,
    OptimizationResponse,
    TaskStatus,
)

# Alias so the route handler reads `OptimizeRequest` even though the contract
# canonically names it `OptimizationRequest`.
OptimizeRequest = OptimizationRequest


class TaskAccepted(BaseModel):
    """202 response from POST /api/v1/optimize."""

    task_id: str = Field(..., description="Celery task id; poll /results/{task_id}")
    status: TaskStatus = TaskStatus.QUEUED


class TaskResult(BaseModel):
    """200 response from GET /api/v1/results/{task_id}."""

    task_id: str
    status: TaskStatus
    result: Optional[OptimizationResponse] = None
    error: Optional[str] = None


__all__ = [
    "OptimizeRequest",
    "TaskAccepted",
    "TaskResult",
    "OptimizationResponse",
    "TaskStatus",
]
