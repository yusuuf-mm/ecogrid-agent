"""
services/solver/models.py

Local data models for the LP engine. These are the concrete shapes the solver code
constructs and returns. They are intentionally separate from `contracts.py` because
the solver's internal types are independent of the inter-service wire format and
have evolved to match the LP's actual decision variables.

The objective field reuses `SolverObjective` from `contracts` so callers can pass
either the enum or the equivalent string literal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from contracts import SolverObjective


ObjectiveName = Literal["MAXIMIZE_PROFIT", "MINIMIZE_CARBON", "MINIMIZE_COST"]


def _coerce_objective(value: Any) -> ObjectiveName:
    """Accept either a string literal or a SolverObjective enum; return the string."""
    if isinstance(value, SolverObjective):
        return value.value  # type: ignore[return-value]
    if isinstance(value, str):
        upper = value.upper()
        if upper not in ("MAXIMIZE_PROFIT", "MINIMIZE_CARBON", "MINIMIZE_COST"):
            raise ValueError(
                f"Unknown objective '{value}'. "
                "Expected one of MAXIMIZE_PROFIT, MINIMIZE_CARBON, MINIMIZE_COST."
            )
        return upper  # type: ignore[return-value]
    raise TypeError(f"objective must be str or SolverObjective, got {type(value).__name__}")


@dataclass
class SolverInput:
    """All parameters the LP needs to build a 24-hour schedule.

    `prices` and `solar_forecast` must each be exactly 24 floats (one per hour).
    `carbon_intensity` is only required when objective is MINIMIZE_CARBON;
    if absent in that case, all hours are treated as equal carbon intensity.
    `battery_capacity_kwh` and `max_charge_rate_kw` are overridable per call
    but default to the system constants (see engine.py).
    """

    prices: list[float]
    solar_forecast: list[float]
    min_battery_buffer: float
    initial_soc: float
    objective: ObjectiveName = "MAXIMIZE_PROFIT"
    carbon_intensity: Optional[list[float]] = None
    battery_capacity_kwh: float = 1000.0
    max_charge_rate_kw: float = 250.0

    def __post_init__(self) -> None:
        if len(self.prices) != 24:
            raise ValueError(f"prices must have 24 values, got {len(self.prices)}")
        if len(self.solar_forecast) != 24:
            raise ValueError(
                f"solar_forecast must have 24 values, got {len(self.solar_forecast)}"
            )
        if self.carbon_intensity is not None and len(self.carbon_intensity) != 24:
            raise ValueError(
                f"carbon_intensity must have 24 values, got {len(self.carbon_intensity)}"
            )
        if not (0.0 <= self.min_battery_buffer <= 1.0):
            raise ValueError(
                f"min_battery_buffer must be in [0.0, 1.0], got {self.min_battery_buffer}"
            )
        if self.battery_capacity_kwh <= 0:
            raise ValueError("battery_capacity_kwh must be > 0")
        if self.max_charge_rate_kw <= 0:
            raise ValueError("max_charge_rate_kw must be > 0")
        if not (0.0 <= self.initial_soc <= self.battery_capacity_kwh):
            raise ValueError(
                f"initial_soc {self.initial_soc} out of bounds [0, {self.battery_capacity_kwh}]"
            )
        self.objective = _coerce_objective(self.objective)


@dataclass
class HourlyAction:
    """One hour of the optimized schedule."""

    hour: int
    action: str
    charge_kw: float
    discharge_kw: float
    soc_kwh: float


@dataclass
class SolverMetrics:
    """Summary metrics for a solved run."""

    expected_revenue_usd: float
    carbon_saved_kg: float
    safety_constraints_passed: bool
    solver_time_ms: int


@dataclass
class SolverResult:
    """The solver's output. INFEASIBLE is a valid status — see `reason`."""

    status: str
    schedule: list[HourlyAction] = field(default_factory=list)
    metrics: SolverMetrics = field(
        default_factory=lambda: SolverMetrics(0.0, 0.0, False, 0)
    )
    reason: Optional[str] = None
