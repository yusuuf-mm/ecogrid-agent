"""
services/solver/tests/test_engine.py

Unit tests for the GridSolver. No mocks on the solver itself — the real GLOP
LP is run end-to-end. These run in <100ms each.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable so `from services.solver...` resolves.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

from services.solver.engine import GridSolver
from services.solver.models import SolverInput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Realistic-ish price profile: low overnight, high in afternoon (USD/kWh).
PRICES_REALISTIC: list[float] = [
    0.04, 0.03, 0.03, 0.03, 0.04, 0.05,  # 00–05 cheap
    0.07, 0.10, 0.12, 0.14, 0.16, 0.18,  # 06–11 rising
    0.22, 0.25, 0.28, 0.30, 0.27, 0.22,  # 12–17 peak
    0.18, 0.15, 0.12, 0.10, 0.08, 0.06,  # 18–23 falling
]

# Modest solar generation curve (kW). All non-negative.
SOLAR_DAY: list[float] = [
    0, 0, 0, 0, 0, 0,
    20, 80, 180, 300, 420, 500,
    540, 520, 440, 300, 150, 40,
    0, 0, 0, 0, 0, 0,
]

CAPACITY = 1000.0
MAX_RATE = 250.0
BUFFER_FRAC = 0.10  # 10% hospital reserve
BUFFER_KWH = BUFFER_FRAC * CAPACITY  # 100 kWh


def _default_input(**overrides) -> SolverInput:
    base = dict(
        prices=list(PRICES_REALISTIC),
        solar_forecast=list(SOLAR_DAY),
        min_battery_buffer=BUFFER_FRAC,
        initial_soc=500.0,  # 50%
        objective="MAXIMIZE_PROFIT",
    )
    base.update(overrides)
    return SolverInput(**base)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_profit_objective_discharges_in_high_price_hours() -> None:
    """MAXIMIZE_PROFIT must produce more total discharge during high-LMP hours
    than during low-LMP hours, when the battery has stored energy to sell."""
    # Start full so the solver has energy available to discharge.
    solver = GridSolver()
    result = solver.solve(_default_input(initial_soc=CAPACITY))

    assert result.status == "OPTIMAL", f"Expected OPTIMAL, got {result.status}: {result.reason}"
    assert len(result.schedule) == 24

    # Split the day into low-price (00–05) and high-price (12–17) windows.
    low_hours = list(range(0, 6))
    high_hours = list(range(12, 18))

    discharge_low = sum(
        result.schedule[t].discharge_kw for t in low_hours
    )
    discharge_high = sum(
        result.schedule[t].discharge_kw for t in high_hours
    )

    assert discharge_high > discharge_low, (
        f"Expected more discharge in high-price hours (12–17): "
        f"high={discharge_high:.1f} kW, low={discharge_low:.1f} kW"
    )

    # And the schedule must show positive revenue on a price-differential day.
    assert result.metrics.expected_revenue_usd > 0.0


def test_safety_buffer_never_violated() -> None:
    """For any optimal schedule, every SoC value must be >= the safety buffer."""
    solver = GridSolver()
    result = solver.solve(_default_input())

    assert result.status == "OPTIMAL", f"Expected OPTIMAL, got {result.status}: {result.reason}"
    assert result.metrics.safety_constraints_passed is True

    tolerance = 1e-3
    for entry in result.schedule:
        assert entry.soc_kwh >= BUFFER_KWH - tolerance, (
            f"Hour {entry.hour}: SoC {entry.soc_kwh:.4f} kWh dipped below buffer "
            f"{BUFFER_KWH:.4f} kWh"
        )
        assert entry.soc_kwh <= CAPACITY + tolerance, (
            f"Hour {entry.hour}: SoC {entry.soc_kwh:.4f} kWh exceeded capacity "
            f"{CAPACITY:.4f} kWh"
        )


def test_physically_impossible_input_returns_infeasible() -> None:
    """A starting SoC below the required safety buffer is impossible — the solver
    must return a structured INFEASIBLE result, not raise an exception."""
    solver = GridSolver()

    # 50 kWh is half the 10% × 1000 kWh buffer — physically unreachable.
    impossible_input = _default_input(initial_soc=50.0)
    result = solver.solve(impossible_input)

    assert result.status == "INFEASIBLE"
    assert result.schedule == []
    assert result.reason is not None and len(result.reason) > 0
    assert "buffer" in result.reason.lower() or "soc" in result.reason.lower()

    # Also: the call must not have raised.
    assert result.metrics.safety_constraints_passed is False


def test_simultaneous_charge_and_discharge_never_occurs() -> None:
    """In a realistic price environment, the LP should never produce an hour
    where both charge_kw and discharge_kw are non-zero. The anti-simultaneous
    constraint c+d <= max_rate is enforced and the objective does not reward
    round-tripping energy at the same hour."""
    solver = GridSolver()
    result = solver.solve(_default_input())

    assert result.status == "OPTIMAL", f"Expected OPTIMAL, got {result.status}: {result.reason}"

    eps = 1e-3
    for entry in result.schedule:
        assert not (entry.charge_kw > eps and entry.discharge_kw > eps), (
            f"Hour {entry.hour}: simultaneous charge ({entry.charge_kw:.3f} kW) "
            f"and discharge ({entry.discharge_kw:.3f} kW)"
        )
        # And the action label must agree.
        if entry.charge_kw > eps:
            assert entry.action == "CHARGE"
        elif entry.discharge_kw > eps:
            assert entry.action == "DISCHARGE"
        else:
            assert entry.action == "HOLD"
