# Domain Context: EcoGrid-Agent

This file defines the domain vocabulary for EcoGrid-Agent. Read it before building any service.
Skills like `diagnose`, `improve-codebase-architecture`, and `tdd` read this file to understand
what the code is actually doing.

---

## Core Domain Concepts

### Virtual Power Plant (VPP)
A network of distributed energy resources — industrial batteries, solar arrays, small generators
— that behaves as a single controllable entity to the main grid. EcoGrid-Agent is the software
orchestrator for a single-site VPP: one battery system connected to solar generation and the
public grid. The term "virtual" means the aggregation and control is done in software, not
through physical co-location.

### Battery State of Charge (SoC)
The primary state variable in the LP solver. Represents how full the battery is: `0.0` = empty,
`1.0` = full, `0.5` = half capacity. The solver tracks SoC across 24 hours as a decision
variable and enforces it never drops below the safety buffer retrieved from policy documents.

In code: `soc[t]` — a continuous variable in `[0.0, BATTERY_CAPACITY_KWH]` for each hour `t`.

### Charge / Discharge
- **Charging** (`c[t]`): drawing power from the grid or capturing solar generation. Increases
  SoC. Costs money (grid price) or is free (solar).
- **Discharging** (`d[t]`): releasing stored energy to the grid. Decreases SoC. Earns revenue
  at the current market price.
- Both are bounded by `MAX_CHARGE_RATE_KW` — the physical limit of how fast the battery can
  accept or release power.
- A battery cannot charge and discharge simultaneously. This is enforced as a constraint.

### Locational Marginal Price (LMP)
The wholesale electricity price at a specific grid node, varying by hour. Low LMP = cheap to
charge from the grid. High LMP = profitable to discharge to the grid. EcoGrid-Agent reads
historical LMPs from ERCOT (Texas grid operator). Stored in PostgreSQL, indexed by date and hour.

### Safety Buffer / Hospital Reserve
A minimum SoC fraction that must be maintained at all times (or during specific hours) to ensure
critical infrastructure — hospitals, water pumps, emergency services — never loses power.

This value is not hardcoded. It is retrieved from the vector database at runtime by querying the
policy document that governs the current scenario. The LP solver receives it as a parameter.

Default fallback if no policy is found: `0.10` (10%).

### INFEASIBLE Solver Status
When the LP solver cannot find a valid schedule that satisfies all constraints simultaneously.
Common causes:
- Hospital reserve requirement is higher than what the battery can maintain given starting SoC
- Total solar + grid capacity is insufficient to meet minimum charge targets within rate limits
- Constraint combination makes the feasible region empty

**INFEASIBLE is not an error.** It is a valid `SolverResult` with a `reason` string. The agent
receives it, logs it, and — in Phase 3 — can attempt to relax non-critical constraints and retry.

---

## Solver Model

The LP is a 24-hour planning horizon with hourly time steps (t = 0..23).

**Decision variables** (what the solver decides):
- `c[t]`: power to charge at hour t (kW)
- `d[t]`: power to discharge at hour t (kW)
- `soc[t]`: battery state of charge at end of hour t (kWh)

**Objective** (configurable, default is profit maximization):
- `MAXIMIZE_PROFIT`:  maximize Σ (d[t] × price[t]) − Σ (c[t] × price[t])
- `MINIMIZE_CARBON`:  minimize Σ (c[t] × carbon_intensity[t])
- `MINIMIZE_COST`:    minimize Σ (c[t] × price[t]) − Σ (d[t] × price[t])

**Constraints** (always enforced):
1. Battery physics: `soc[t] = soc[t-1] + c[t] + solar[t] − d[t]`
2. Capacity bounds: `0 ≤ soc[t] ≤ BATTERY_CAPACITY_KWH`
3. Rate bounds: `0 ≤ c[t] ≤ MAX_CHARGE_RATE_KW`, `0 ≤ d[t] ≤ MAX_CHARGE_RATE_KW`
4. No simultaneous charge + discharge: `c[t] + d[t] ≤ MAX_CHARGE_RATE_KW`
5. Safety buffer: `soc[t] ≥ min_battery_buffer × BATTERY_CAPACITY_KWH` (from policy)

**Solver used**: GLOP (Google Linear Optimization Package) — continuous LP. If binary on/off
plant decisions are added in the future, switch to SCIP (MIP). Document that switch in an ADR.

---

## System Modes

The solver objective is a runtime parameter, not a code path:
- `MAXIMIZE_PROFIT` — default. Buy cheap, sell expensive.
- `MINIMIZE_CARBON` — charge only during low-carbon-intensity hours (requires carbon intensity
  data as an additional input array).
- `MINIMIZE_COST` — minimize grid draw cost, treating discharge revenue as a secondary benefit.

---

## Data Model

### Market Prices (PostgreSQL)
Hourly LMP data sourced from ERCOT. Table: `market_prices`.
```
timestamp     TIMESTAMPTZ
price_kwh     FLOAT           -- $/kWh, wholesale LMP
carbon_g_kwh  FLOAT           -- gCO₂/kWh at that hour (optional, for MINIMIZE_CARBON)
node_id       VARCHAR         -- ERCOT pricing node
```

### Grid Policies (Qdrant vector collection: `grid_policies`)
Each document chunk stored with payload:
```json
{
  "text":      "Full text of the chunk",
  "doc_id":    "doc_id_442",
  "doc_title": "Grid Safety SOP 04",
  "chunk_idx": 2
}
```
The embedding is of the `text` field. Retrieval returns the top-1 match by cosine similarity.

### Trained Model Artifact
`data/models/solar_forecast.joblib` — scikit-learn Pipeline containing the XGBoost regressor
and all fitted preprocessing transforms. Load once at worker startup, not per-request.

---

## Audit Trail

Every optimization response includes an `audit` field that makes the system's reasoning
traceable:
- Which policy document was retrieved from the vector DB
- The raw text chunk retrieved
- The constraint value parsed from it
- The solver status and execution time
- The agent's tool call sequence (from structured logs)

This audit trail is not optional. An energy grid system that cannot explain why it made a
decision is not suitable for regulated infrastructure.

---

## Boundaries Between Technologies

**Where the LLM is responsible:**
- Parsing the user's natural language intent
- Deciding which tools to call and in what order
- Reading retrieved policy text and extracting a constraint float
- Synthesizing the final human-readable summary

**Where the LLM is not responsible:**
- Any numerical calculation
- Determining whether a schedule is physically feasible
- Enforcing hard constraints
- Computing profit or carbon metrics

The LP solver is the authority on math. The LLM is the authority on language and reasoning.
Mixing these responsibilities is the failure mode this architecture is designed to prevent.
