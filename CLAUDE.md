# EcoGrid-Agent — Agent Instructions

## What You're Building

An autonomous VPP orchestrator. The system takes natural language grid optimization requests,
retrieves regulatory constraints from a vector database, forecasts solar generation with an ML
model, and runs a Linear Programming solver to produce a mathematically optimal schedule.

The architecture is fixed. Your job is to implement it cleanly, make defensible engineering
decisions within it, and keep every service boundary honest. Read `CONTEXT.md` for domain
vocabulary. Read `shared/contracts.py` to understand the inter-service interface.

---

## Fixed Architecture

```
FastAPI → Celery/Redis → LangChain Agent → [Qdrant, XGBoost, OR-Tools] → JSON Response
```

Data flows in one direction: user prompt → API → task queue → agent → tools → response.
The agent calls tools. Tools do not call each other. Do not introduce circular dependencies.

If you find a reason to change this topology, write an ADR in `docs/adr/` and stop. Don't
implement it until the ADR is reviewed.

---

## Service Boundaries

Each service lives in `services/<name>/`. The only things that cross service boundaries are
types defined in `shared/contracts.py`. Never import from one service into another directly.

```
services/solver/   → implements: optimize_battery_schedule()
                     consumes from shared/: SolverConstraints, SolverResult

services/ml/       → implements: forecast_solar_generation()
                     consumes from shared/: WeatherFeatures, SolarForecast

services/rag/      → implements: query_grid_policies()
                     consumes from shared/: PolicyQuery, PolicyResult

services/agent/    → implements: the LangChain agent + three tool wrappers
                     consumes from shared/: all contracts

services/api/      → implements: POST /api/v1/optimize, GET /api/v1/results/{task_id}
                     consumes from shared/: OptimizationRequest, OptimizationResponse

services/workers/  → implements: run_optimization_pipeline Celery task
                     consumes from shared/: OptimizationRequest
```

---

## Branch Ownership

Before writing any code, confirm your branch:

```bash
git branch --show-current
```

Stay inside your service directory. The only exception is `shared/contracts.py` — changes there
go to `main` first, and all feature branches rebase against main before consuming them.

| Branch                        | Owns                                 |
|-------------------------------|--------------------------------------|
| `feat/phase-1-or-solver`      | `services/solver/`                   |
| `feat/phase-1-backend-api`    | `services/api/` + `services/workers/`|
| `feat/phase-2-ml-forecasting` | `services/ml/`                       |
| `feat/phase-2-vector-rag`     | `services/rag/`                      |
| `feat/phase-3-agent-core`     | `services/agent/`                    |
| `feat/phase-4-docker-infra`   | `infra/`                             |

If a service you depend on isn't built yet, stub its function with the correct signature and
mark it `# TODO: replace stub with real implementation`. Don't fake return values that would
cause downstream behavior to silently pass.

---

## Coding Standards

**Language and types**
- Python 3.11+. Type annotations on every function signature.
- Pydantic v2 for all data models. No raw dicts crossing service boundaries.
- Use `from __future__ import annotations` at the top of files that have forward references.

**Async**
- FastAPI route handlers: async.
- Celery task function: sync (Celery manages its own event loop).
- OR-Tools solver: sync — it's a C++ binding, forcing it async adds no value.
- ML inference: sync unless the model load is deferred to startup.

**Logging**
- Use `loguru`. Not `print`. Not stdlib `logging`.
- Log levels: `DEBUG` for solver internals, `INFO` for agent tool calls, `WARNING` for
  constraint fallbacks, `ERROR` for infeasibility or task failure.
- Every agent tool call gets an INFO log with the input and output. This is the audit trail.

**Dependencies**
- Add with `poetry add <package>`. Don't edit `pyproject.toml` by hand.
- Keep service-level deps in the root `pyproject.toml` with optional groups if needed.
  Don't create per-service `requirements.txt` files.

**Error handling**
- Infeasible solver status is not an exception. It's a valid `SolverResult` with
  `status="INFEASIBLE"` and a `reason` string.
- Network/DB errors in tools are caught, logged at ERROR, and returned as structured
  `ToolError` types. The agent decides what to do with them — don't swallow errors silently.
- Celery tasks retry once on failure (exponential backoff). Second failure returns a
  structured error stored in Redis — no unhandled exceptions propagating to the broker.

**Tests**
Write tests as you build. Don't defer them.
- Unit: pure functions (solver math, ML inference, policy parsing logic).
- Integration: anything touching Redis, Qdrant, Postgres, or a model file on disk.
- E2E: full pipeline from API call to result polling (runs against full docker-compose).

Run tests:
```bash
pytest tests/unit/ -v                  # no external deps
pytest tests/integration/ -v           # needs Redis + Qdrant + Postgres
pytest tests/e2e/ -v                   # needs full docker-compose stack
```

---

## Service-Specific Expectations

### services/solver/
The LP model is the core artifact. It must be:
- Fully parameterized — no hardcoded constraint values anywhere in the solver code
- Configurable objective: `MAXIMIZE_PROFIT` | `MINIMIZE_CARBON` | `MINIMIZE_COST`
- INFEASIBLE responses include a human-readable `reason` string explaining which constraint
  could not be satisfied (e.g., "Hospital buffer 0.40 exceeds max battery SoC 1.0 given
  starting charge 0.35")

Use GLOP for continuous LP (default). Use SCIP if/when MIP is needed (binary plant decisions).
Don't switch solvers without an ADR — SCIP has different performance characteristics.

See `services/solver/CLAUDE.md` for implementation details.

### services/ml/
Training and inference are separate concerns, in separate modules:
- `services/ml/training/` — data loading, feature engineering, model training, artifact saving
- `services/ml/inference/` — load artifact, apply identical feature transforms, return forecast

The model artifact saves to `data/models/solar_forecast.joblib`.
Inference returns `list[float]` of exactly 24 values (one per hour). Validate this at the boundary.

See `services/ml/CLAUDE.md` for feature engineering spec.

### services/rag/
Documents are chunked and embedded by `scripts/seed_vector_db.py`. The retrieval function
returns `PolicyResult` containing the raw chunk, the source document ID, and the parsed
constraint float.

Parsing floats from retrieved text: use a short LLM call with a structured prompt, not regex.
Always fall back to `0.10` on parse failure and log a WARNING with the raw text that failed.

Use `all-MiniLM-L6-v2` for embeddings. It runs locally, no API calls.

See `services/rag/CLAUDE.md` for the Qdrant collection schema and chunking strategy.

### services/agent/
Each LangChain tool wraps exactly one downstream function. No business logic inside tool
functions — they transform inputs, call the service function, transform outputs.

The system prompt lives in `services/agent/prompts/system.txt`. Edit it there. Not in code.

Agent executor: `max_iterations=5`. Not unbounded. On max_iterations exceeded, return a
structured partial result with a `"status": "MAX_ITERATIONS_EXCEEDED"` field.

Every tool call is logged as a structured JSON object (input, output, duration_ms). These logs
are the audit trail the system promises.

See `services/agent/CLAUDE.md` for tool signatures and the agent loop.

### services/api/
Two endpoints:
- `POST /api/v1/optimize` → validates `OptimizationRequest`, enqueues Celery task, returns
  `{"task_id": "<uuid>", "status": "QUEUED"}` with HTTP 202.
- `GET /api/v1/results/{task_id}` → polls Redis for result, returns current state.

No business logic in route handlers. Validate in, serialize out, enqueue, done.

See `services/api/CLAUDE.md` for request/response shapes.

### services/workers/
One task: `run_optimization_pipeline`. It receives `OptimizationRequest`, instantiates the
agent, runs it, stores `OptimizationResponse` in Redis.

Task config: `max_retries=1`, `default_retry_delay=30` (seconds).

See `services/workers/CLAUDE.md` for Celery app configuration.

---

## Shared Contracts

`shared/contracts.py` is the only file that defines types crossing service boundaries. Read it
before building any service. If you need a new type, add it here — don't invent local types
that duplicate shared ones.

The current contract shapes are documented in `shared/contracts.py` with inline comments.

---

## Data Sources

Use real data where possible:
- **Solar**: NREL NSRDB historical hourly CSVs. Download script: `scripts/download_nrel_data.py`
- **Market prices**: ERCOT historical LMP data. Download script: `scripts/download_ercot_prices.py`
- **Policies**: Fabricated regulatory documents in `data/policies/`. These mimic real FERC/NERC
  SLA format. Ingest them via `scripts/seed_vector_db.py`.

---

## Multi-Terminal Workflow

When running multiple Claude Code sessions simultaneously across branches:

1. Each terminal confirms its branch before writing code.
2. `shared/contracts.py` changes go to `main` first. Feature branches rebase to pick them up.
3. Branches never merge into each other — always into `main` via PR.
4. When a dependency isn't ready yet, stub it. Name the stub clearly. Add a comment with the
   expected real signature.
5. If you're uncertain whether a design decision crosses into another branch's territory, stop
   and write it in the PR description as a question rather than guessing.

---

## What to Avoid

- Don't add a frontend. This is a systems engineering portfolio. Terminal logs and JSON responses
  are the interface.
- Don't call an LLM for anything the solver can answer deterministically. The LLM's job is
  parsing and reasoning. The solver's job is math.
- Don't abstract prematurely. Build the thing that works first. Refactor when there's a second
  use case.
- Don't mock external dependencies in integration tests. Use real Redis, real Qdrant, real
  Postgres — docker-compose brings them up in seconds.
- Don't skip the audit trail fields in `OptimizationResponse`. They're not optional. A system
  that can't explain its decisions isn't suitable for production energy infrastructure.

---

## Agent Skills

### Issue tracker

Issues live in GitHub Issues on this repository. Use `gh issue create` to open issues and
`gh issue list` to browse them. See `docs/agents/issue-tracker.md`.

### Triage labels

Uses the default five-role label vocabulary. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout. `CONTEXT.md` at root contains domain vocabulary and key design
decisions. `docs/adr/` contains Architecture Decision Records. See `docs/agents/domain.md`.
