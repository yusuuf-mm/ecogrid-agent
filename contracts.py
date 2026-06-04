"""
shared/contracts.py

The only types that cross service boundaries live here.
If you need a new inter-service type, add it here — don't create local duplicates.

Services import from this module. This module imports from nothing inside this codebase.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SolverObjective(str, Enum):
    MAXIMIZE_PROFIT  = "MAXIMIZE_PROFIT"
    MINIMIZE_CARBON  = "MINIMIZE_CARBON"
    MINIMIZE_COST    = "MINIMIZE_COST"


class SolverStatus(str, Enum):
    OPTIMAL     = "OPTIMAL"
    INFEASIBLE  = "INFEASIBLE"
    ERROR       = "ERROR"


class TaskStatus(str, Enum):
    QUEUED      = "QUEUED"
    RUNNING     = "RUNNING"
    SUCCESS     = "SUCCESS"
    FAILURE     = "FAILURE"


# ---------------------------------------------------------------------------
# services/ml — WeatherFeatures, SolarForecast
# ---------------------------------------------------------------------------

class WeatherFeatures(BaseModel):
    """Input to the solar forecast model.

    All fields represent conditions for the TARGET day (the day being forecast).
    Temperatures in Celsius. Irradiance in W/m².
    """
    date: str                                  # ISO date: "2025-07-15"
    hourly_temperature_c: list[float]          # 24 values
    hourly_cloud_cover_pct: list[float]        # 24 values, 0–100
    hourly_irradiance_wm2: list[float]         # 24 values
    is_holiday: bool = False


class SolarForecast(BaseModel):
    """Output of the solar forecast model.

    Predicted solar generation for each hour of the target day.
    """
    date: str
    hourly_generation_kw: list[float]          # 24 values, non-negative
    model_version: str


# ---------------------------------------------------------------------------
# services/rag — PolicyQuery, PolicyResult
# ---------------------------------------------------------------------------

class PolicyQuery(BaseModel):
    """Input to the vector DB retrieval function."""
    query_text: str
    top_k: int = 1


class PolicyResult(BaseModel):
    """Output of the RAG retrieval + parameter extraction step."""
    doc_id: str
    doc_title: str
    raw_chunk: str                             # The retrieved text chunk
    constraint_float: float = Field(
        ge=0.0, le=1.0,
        description="Parsed safety buffer fraction, e.g. 0.30 for 30% SoC reserve."
    )
    parse_confidence: str                      # "parsed" | "fallback"


# ---------------------------------------------------------------------------
# services/solver — SolverConstraints, SolverResult, ScheduleHour
# ---------------------------------------------------------------------------

class SolverConstraints(BaseModel):
    """All parameters fed into the LP solver.

    Nothing in here is hardcoded in solver code — everything comes from upstream.
    """
    solar_forecast_kw: list[float]             # 24 values from ML model
    market_prices_kwh: list[float]             # 24 values from PostgreSQL
    carbon_intensity_g_kwh: Optional[list[float]] = None  # 24 values, for MINIMIZE_CARBON

    battery_capacity_kwh: float = 1000.0
    max_charge_rate_kw: float   = 250.0
    initial_soc_kwh: float      = 500.0        # Battery starts at 50% by default
    min_battery_buffer: float   = 0.10         # Fraction from policy (default 10%)

    objective: SolverObjective = SolverObjective.MAXIMIZE_PROFIT
    policy_doc_id: Optional[str] = None        # Which policy set the buffer


class ScheduleHour(BaseModel):
    """One hour of the optimized schedule."""
    hour: int                                  # 0–23
    charge_kw: float
    discharge_kw: float
    solar_stored_kw: float
    battery_soc_kwh: float
    action_label: str                          # Human-readable: CHARGE_FROM_GRID, etc.
    reason: str                                # Why this action was taken


class SolverResult(BaseModel):
    """Output of the LP solver."""
    status: SolverStatus
    schedule: list[ScheduleHour] = []
    total_profit_usd: float = 0.0
    carbon_saved_kg: float = 0.0
    safety_constraints_passed: bool = False
    solver_time_ms: float = 0.0
    reason: Optional[str] = None               # Set when status is INFEASIBLE or ERROR


# ---------------------------------------------------------------------------
# services/api — OptimizationRequest, OptimizationResponse
# ---------------------------------------------------------------------------

class OptimizationRequest(BaseModel):
    """Inbound API request."""
    prompt: str
    date: Optional[str] = None                 # Target date; defaults to tomorrow
    objective: SolverObjective = SolverObjective.MAXIMIZE_PROFIT


class AuditTrail(BaseModel):
    """Full reasoning trace attached to every response."""
    policy_doc_retrieved: Optional[str] = None
    policy_raw_text: Optional[str] = None
    constraint_injected: Optional[dict] = None
    solar_forecast_used: Optional[list[float]] = None
    market_prices_used: Optional[list[float]] = None
    solver_status: Optional[str] = None
    solver_time_ms: Optional[float] = None
    agent_tool_calls: list[dict] = []          # Sequence of {tool, input, output, duration_ms}


class OptimizationResponse(BaseModel):
    """Final API response, stored in Redis by the Celery worker."""
    task_id: str
    status: TaskStatus
    summary: Optional[str] = None
    schedule: list[ScheduleHour] = []
    metrics: Optional[dict] = None
    audit: AuditTrail = Field(default_factory=AuditTrail)
    error: Optional[str] = None               # Set when status is FAILURE
