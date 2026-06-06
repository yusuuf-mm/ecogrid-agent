"""
services/solver/

LP engine for EcoGrid-Agent. Public surface:

    from services.solver import GridSolver, SolverInput, SolverResult

    solver = GridSolver()
    result = solver.solve(input_)      # always returns SolverResult, never raises
                                       # on infeasibility

    # Backward-compat adapter for the shared-contracts call site:
    from services.solver import optimize_battery_schedule
"""
from __future__ import annotations

from services.solver.engine import (
    BATTERY_CAPACITY_KWH,
    MAX_CHARGE_RATE_KW,
    GridSolver,
    optimize_battery_schedule,
)
from services.solver.models import (
    HourlyAction,
    ObjectiveName,
    SolverInput,
    SolverMetrics,
    SolverResult,
)

__all__ = [
    "BATTERY_CAPACITY_KWH",
    "GridSolver",
    "HourlyAction",
    "MAX_CHARGE_RATE_KW",
    "ObjectiveName",
    "SolverInput",
    "SolverMetrics",
    "SolverResult",
    "optimize_battery_schedule",
]
