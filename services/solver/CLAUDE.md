# CLAUDE.md — services/solver

You are working on the `feat/or-solver` branch of EcoGrid-Agent.

Your scope is `services/solver/` only. Do not touch any other service directory,
`shared/`, or any root config files. If you notice something wrong outside your
scope, note it in a comment or a TODO — do not fix it here.

Read `CONTEXT.md` and `shared/contracts.py` before writing any code.
The types you implement against are already defined in `shared/contracts.py`.
Do not redefine `SolverInput`, `SolverResult`, `HourlyAction`, or `SolverMetrics` locally.

---

## What This Service Is

The LP engine. It takes numbers in and returns an optimal 24-hour battery
charge/discharge schedule. It knows nothing about the LLM, the vector DB,
or the ML model. It does one thing: solve a linear program.

This is the most deterministic service in the stack. Given the same input,
it must always return the same output. No randomness, no LLM calls, no
external I/O inside the solver itself.

---

## What to Build

### `services/solver/config.py`
Battery constants, read from environment with sensible defaults:
- `BATTERY_CAPACITY_KWH = 1000.0`
- `MAX_CHARGE_RATE_KW = 250.0`
- `INITIAL_SOC_KWH = 500.0`

Use `pydantic-settings` `BaseSettings`. Do not hardcode these in `engine.py`.

### `services/solver/engine.py`
Single class: `GridSolver`.
Single public method: `solve(input: SolverInput) -> SolverResult`.

The LP model (24-hour horizon, hourly steps t = 0..23):

Decision variables:
- `c[t]` — charge power in kW
- `d[t]` — discharge power in kW
- `soc[t]` — battery SoC in kWh at end of hour t

Constraints (all must be enforced, no exceptions):
1. Physics: `soc[t] = soc[t-1] + c[t] + solar[t] - d[t]`
   Initial condition: `soc[-1] = input.initial_soc`
2. Capacity: `0 <= soc[t] <= BATTERY_CAPACITY_KWH`
3. Rate: `0 <= c[t] <= MAX_CHARGE_RATE_KW`, `0 <= d[t] <= MAX_CHARGE_RATE_KW`
4. No simultaneous charge+discharge: `c[t] + d[t] <= MAX_CHARGE_RATE_KW`
5. Safety buffer: `soc[t] >= input.min_battery_buffer * BATTERY_CAPACITY_KWH`

Objectives (switch on `input.objective`):
- `MAXIMIZE_PROFIT`: maximize `Σ d[t]*price[t] - Σ c[t]*price[t]`
- `MINIMIZE_COST`: minimize `Σ c[t]*price[t] - Σ d[t]*price[t]`
- `MINIMIZE_CARBON`: minimize `Σ c[t]*carbon[t]`
  If `input.carbon_intensity` is None and objective is `MINIMIZE_CARBON`,
  return `SolverResult(status=ERROR, reason="carbon_intensity required for MINIMIZE_CARBON")`

Use Google OR-Tools GLOP solver (`from ortools.linear_solver import pywraplp`).
Solver name: `"GLOP"`.

After solving:
- If status is OPTIMAL: build the `schedule` list, compute `metrics`, return result.
- If status is INFEASIBLE or UNBOUNDED: return `SolverResult` with the appropriate
  `SolverStatus` and a `reason` string. Never raise an exception for solver status.
- Record wall-clock solver time in `metrics.solver_time_ms`.

For `HourlyAction.action`, classify each hour:
- `charge_kw > 0.1` and `solar[t] > 0`: `STORE_SOLAR`
- `charge_kw > 0.1` and `solar[t] <= 0.1`: `CHARGE_FROM_GRID`
- `discharge_kw > 0.1`: `DISCHARGE_TO_GRID`
- Both near zero and `soc[t]` is at or near the buffer floor: `HOLD_RESERVE`
- Otherwise: `IDLE`

Leave `HourlyAction.reason` as an empty string — that field is populated by
the agent layer, not the solver.

### `services/solver/tests/test_engine.py`

Test the real LP — do not mock the solver.

Required tests:
1. `MAXIMIZE_PROFIT` produces higher total discharge during high-price hours
   than during low-price hours (use a price array with obvious peak/off-peak pattern).
2. Battery SoC never drops below `min_battery_buffer * BATTERY_CAPACITY_KWH` in any
   hour of any result.
3. `c[t] + d[t] <= MAX_CHARGE_RATE_KW` holds for every hour in every result.
4. An impossible constraint combination (e.g. `min_battery_buffer=0.99` with
   `initial_soc=0`) returns `status=INFEASIBLE`, not an exception.
5. `MINIMIZE_CARBON` with no `carbon_intensity` returns `status=ERROR`.
6. Solver returns 24 `HourlyAction` entries for a valid OPTIMAL result.

### `services/solver/__init__.py`
Export `GridSolver` and all models re-exported from `shared.contracts`.

---

## What Not to Build

- No FastAPI routes here
- No database connections
- No LLM calls
- No logging beyond Python's stdlib `logging`
- No CLI entry point

---

## Definition of Done

- All six tests pass with `pytest services/solver/tests/`
- `mypy services/solver/` returns no errors on the public interface
- A manual smoke test: instantiate `GridSolver`, call `solve()` with a realistic
  `SolverInput`, print the result, confirm it looks right
- PR description includes: solver time on a standard 24-hour run, and one
  example schedule printed as a table
