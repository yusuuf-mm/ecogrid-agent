"""
services/solver/engine.py

LP engine for EcoGrid-Agent. Uses Google OR-Tools GLOP (continuous LP).

Decision variables over a 24-hour horizon:
    c[t]  charge power at hour t (kW)
    d[t]  discharge power at hour t (kW)
    s[t]  solar actually absorbed by the battery at hour t (0..solar[t])
          (the rest is curtailed — the inverter / plant operator decides)
    soc[t] battery state of charge at end of hour t (kWh)

Constraints:
    1. Physics:  soc[t] = soc[t-1] + c[t] + s[t] - d[t]
    2. Capacity: min_buffer * CAP <= soc[t] <= CAP
    3. Rates:    0 <= c[t], d[t] <= MAX_RATE
    4. Solar use:0 <= s[t] <= solar[t]
    5. Anti-simul: c[t] + d[t] <= MAX_RATE  (redundant with rates in a continuous LP,
                                              but enforces the spec's intent)
    6. Buffer:   soc[t] >= min_buffer * CAP  (hospital reserve, from policy)

`c[t]` is grid charge (costs money); `s[t]` is solar absorption (free). Both feed SoC.
Without `s[t]`, the physics equation would force every kWh of forecast solar into the
battery and the LP would be infeasible whenever solar[t] exceeds what the battery can
store plus export through `d[t]` in a single hour. `s[t]` is the curtailment lever.

INFEASIBILITY is not an exception — it is a SolverResult with status="INFEASIBLE"
and a human-readable `reason`. The agent decides what to do next.
"""
from __future__ import annotations

import os
import time
from typing import Optional

from loguru import logger
from ortools.linear_solver import pywraplp

from shared.contracts import (
    ScheduleHour,
    SolverConstraints,
    SolverResult,
    SolverStatus,
)

from services.solver.models import (
    HourlyAction,
    SolverInput,
    SolverMetrics,
    SolverResult as InternalSolverResult,
)


# ---------------------------------------------------------------------------
# Constants — overridable via environment variables
# ---------------------------------------------------------------------------

BATTERY_CAPACITY_KWH: float = float(os.getenv("ECOGRID_BATTERY_CAPACITY_KWH", "1000.0"))
MAX_CHARGE_RATE_KW: float = float(os.getenv("ECOGRID_MAX_CHARGE_RATE_KW", "250.0"))

# Tolerance for "is this variable effectively zero?"
_EPS_KW: float = 1e-3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_label(charge_kw: float, discharge_kw: float) -> str:
    if charge_kw > _EPS_KW and discharge_kw > _EPS_KW:
        return "SIMULTANEOUS"
    if charge_kw > _EPS_KW:
        return "CHARGE"
    if discharge_kw > _EPS_KW:
        return "DISCHARGE"
    return "HOLD"


def _solver_status_name(code: int) -> str:
    # Map pywraplp's integer return codes to stable names.
    return {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }.get(code, f"UNKNOWN_{code}")


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class GridSolver:
    """Builds and solves the 24-hour battery schedule LP.

    Thread-safety: each call to `solve` constructs a fresh pywraplp.Solver
    instance, so concurrent calls from different threads are safe.
    """

    def solve(self, input: SolverInput) -> InternalSolverResult:
        """Solve a 24-hour battery schedule LP. Never raises on infeasibility."""
        capacity = input.battery_capacity_kwh
        max_rate = input.max_charge_rate_kw
        buffer_kwh = input.min_battery_buffer * capacity
        initial_soc = input.initial_soc

        # ----- Trivial pre-checks for clear, actionable reason messages -----
        if initial_soc < buffer_kwh - 1e-6:
            return InternalSolverResult(
                status="INFEASIBLE",
                reason=(
                    f"Initial SoC {initial_soc:.2f} kWh is below the required safety "
                    f"buffer {buffer_kwh:.2f} kWh "
                    f"(min_battery_buffer={input.min_battery_buffer:.2f} × "
                    f"capacity={capacity:.2f} kWh)."
                ),
            )
        if input.min_battery_buffer > 1.0 + 1e-9:
            return InternalSolverResult(
                status="INFEASIBLE",
                reason=(
                    f"Safety buffer fraction {input.min_battery_buffer} exceeds 1.0; "
                    f"no feasible schedule can satisfy an SoC > 100%."
                ),
            )

        # ----- Build the LP -----
        solver = pywraplp.Solver.CreateSolver("GLOP")
        if solver is None:
            logger.error("GLOP solver unavailable in this OR-Tools build")
            return InternalSolverResult(
                status="ERROR",
                reason="GLOP solver backend not available.",
            )

        solver.SetTimeLimit(10_000)  # 10s — well above the LP's actual cost
        hours = range(24)

        # Decision variables
        c = [solver.NumVar(0.0, max_rate, f"c_{t}") for t in hours]
        d = [solver.NumVar(0.0, max_rate, f"d_{t}") for t in hours]
        s = [
            solver.NumVar(0.0, max(0.0, input.solar_forecast[t]), f"s_{t}")
            for t in hours
        ]
        soc = [
            solver.NumVar(buffer_kwh, capacity, f"soc_{t}")
            for t in hours
        ]

        # Physics: soc[t] = soc[t-1] + c[t] + s[t] - d[t]
        for t in hours:
            if t == 0:
                solver.Add(
                    soc[t] == initial_soc + c[t] + s[t] - d[t],
                    name=f"physics_{t}",
                )
            else:
                solver.Add(
                    soc[t] == soc[t - 1] + c[t] + s[t] - d[t],
                    name=f"physics_{t}",
                )

        # Anti-simultaneous: c[t] + d[t] <= max_rate
        for t in hours:
            solver.Add(c[t] + d[t] <= max_rate, name=f"no_simul_{t}")

        # ----- Objective -----
        objective = input.objective
        if objective == "MAXIMIZE_PROFIT":
            solver.Maximize(
                sum(d[t] * input.prices[t] - c[t] * input.prices[t] for t in hours)
            )
        elif objective == "MINIMIZE_CARBON":
            # If no carbon intensity provided, fall back to flat 1.0 per kWh —
            # the optimization still runs and prefers not to charge at all.
            carbon = input.carbon_intensity or [1.0] * 24
            solver.Minimize(sum(c[t] * carbon[t] for t in hours))
        elif objective == "MINIMIZE_COST":
            solver.Minimize(
                sum(c[t] * input.prices[t] - d[t] * input.prices[t] for t in hours)
            )
        else:  # pragma: no cover — SolverInput.__post_init__ rejects this
            return InternalSolverResult(
                status="ERROR",
                reason=f"Unknown objective '{objective}'.",
            )

        # ----- Solve -----
        t0 = time.perf_counter()
        rc = solver.Solve()
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        status_name = _solver_status_name(rc)

        logger.debug(
            "GLOP rc={} ({}), elapsed={}ms, objective={}",
            rc, status_name, elapsed_ms, objective,
        )

        if rc != pywraplp.Solver.OPTIMAL:
            return InternalSolverResult(
                status="INFEASIBLE",
                reason=(
                    f"LP returned {status_name} — constraints cannot be simultaneously "
                    f"satisfied. Buffer={input.min_battery_buffer:.2f}×{capacity:.0f} kWh, "
                    f"initial_soc={initial_soc:.0f} kWh, "
                    f"max_rate={max_rate:.0f} kW."
                ),
            )

        # ----- Extract schedule and metrics -----
        schedule: list[HourlyAction] = []
        total_revenue = 0.0
        total_carbon_charged = 0.0
        total_carbon_discharged = 0.0
        for t in hours:
            charge_v = c[t].solution_value()
            discharge_v = d[t].solution_value()
            soc_v = soc[t].solution_value()
            schedule.append(
                HourlyAction(
                    hour=t,
                    action=_action_label(charge_v, discharge_v),
                    charge_kw=charge_v,
                    discharge_kw=discharge_v,
                    soc_kwh=soc_v,
                )
            )
            total_revenue += (discharge_v - charge_v) * input.prices[t]
            if input.carbon_intensity is not None:
                total_carbon_charged += charge_v * input.carbon_intensity[t]
                total_carbon_discharged += discharge_v * input.carbon_intensity[t]

        # Carbon saved: net carbon benefit of the schedule.
        # Positive when the battery displaces more carbon than it stores.
        carbon_saved_kg = (total_carbon_discharged - total_carbon_charged) / 1000.0

        # Safety check: every SoC at or above buffer (within tolerance).
        safety_ok = all(s.soc_kwh >= buffer_kwh - 1e-3 for s in schedule)

        metrics = SolverMetrics(
            expected_revenue_usd=round(total_revenue, 6),
            carbon_saved_kg=round(carbon_saved_kg, 6),
            safety_constraints_passed=safety_ok,
            solver_time_ms=elapsed_ms,
        )
        return InternalSolverResult(
            status="OPTIMAL",
            schedule=schedule,
            metrics=metrics,
            reason=None,
        )


# ---------------------------------------------------------------------------
# Backward-compatible adapter for the existing shared-contracts call site.
# The agent (when built) will call this directly; keep it stable.
# ---------------------------------------------------------------------------

def optimize_battery_schedule(
    constraints: SolverConstraints,
) -> SolverResult:
    """Adapter: SolverConstraints (shared) -> GridSolver -> SolverResult (shared)."""
    solver_input = SolverInput(
        prices=list(constraints.market_prices_kwh),
        solar_forecast=list(constraints.solar_forecast_kw),
        min_battery_buffer=constraints.min_battery_buffer,
        initial_soc=constraints.initial_soc_kwh,
        objective=constraints.objective.value,
        carbon_intensity=(
            list(constraints.carbon_intensity_g_kwh)
            if constraints.carbon_intensity_g_kwh is not None
            else None
        ),
        battery_capacity_kwh=constraints.battery_capacity_kwh,
        max_charge_rate_kw=constraints.max_charge_rate_kw,
    )

    internal: InternalSolverResult = GridSolver().solve(solver_input)

    if internal.status != "OPTIMAL":
        return SolverResult(
            status=SolverStatus(internal.status),
            total_profit_usd=0.0,
            carbon_saved_kg=0.0,
            safety_constraints_passed=False,
            solver_time_ms=float(internal.metrics.solver_time_ms),
            reason=internal.reason,
        )

    schedule = [
        ScheduleHour(
            hour=a.hour,
            charge_kw=a.charge_kw,
            discharge_kw=a.discharge_kw,
            solar_stored_kw=0.0,
            battery_soc_kwh=a.soc_kwh,
            action_label=a.action,
            reason=a.action,
        )
        for a in internal.schedule
    ]
    return SolverResult(
        status=SolverStatus.OPTIMAL,
        schedule=schedule,
        total_profit_usd=internal.metrics.expected_revenue_usd,
        carbon_saved_kg=internal.metrics.carbon_saved_kg,
        safety_constraints_passed=internal.metrics.safety_constraints_passed,
        solver_time_ms=float(internal.metrics.solver_time_ms),
        reason=None,
    )
