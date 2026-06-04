# CLAUDE.md — services/api

You are working on the `feat/backend-api` branch of EcoGrid-Agent.

Your scope covers two directories: `services/api/` and `services/workers/`.
They are one branch because the API and the task queue are tightly coupled —
the API enqueues tasks, the worker executes them. Read both this file and
`services/workers/CLAUDE.md` before starting.

Do not touch `services/solver/`, `services/ml/`, `services/rag/`, or
`services/agent/`. If something in those directories is wrong, note it —
don't fix it here.

Read `CONTEXT.md` and `shared/contracts.py` first. All request/response types
are defined there. Do not redefine them locally.

---

## What This Service Is

The HTTP entry point and async task layer. Its job is simple: receive a
request, put it on a queue, return a task ID immediately. Later, when the
worker finishes, the result is available to poll.

The API does not call the solver directly. It does not call the agent.
It enqueues a task and gets out of the way.

---

## What to Build

### `services/api/config.py`
Settings via `pydantic-settings`:
- `REDIS_URL: str = "redis://localhost:6379/0"`
- `API_PREFIX: str = "/api/v1"`
- `CELERY_TASK_SERIALIZER: str = "json"`

### `services/api/main.py`
FastAPI application. Three routes, nothing more:

```
GET  /health
POST /api/v1/optimize
GET  /api/v1/results/{task_id}
```

**GET /health**
Returns `{"status": "ok"}`. Used by Docker health checks.

**POST /api/v1/optimize**
- Accepts `OptimizeRequest` from `shared.contracts`
- Validates the request (Pydantic handles this automatically)
- Enqueues a Celery task: `run_optimization.delay(request.model_dump())`
- Returns `TaskAccepted` (HTTP 202) with `task_id` and `poll_url`
- Never blocks on the task result

**GET /api/v1/results/{task_id}**
- Looks up the Celery `AsyncResult` by `task_id`
- If `PENDING` or `STARTED`: return `{"task_id": task_id, "status": "PENDING"}`
- If `SUCCESS`: return the full `OptimizationResponse` (deserialise from task result)
- If `FAILURE`: return `{"task_id": task_id, "status": "ERROR", "reason": str(exc)}`
- HTTP 200 in all cases — the status field communicates the state, not the HTTP code

### `services/api/dependencies.py`
Any FastAPI dependency functions (e.g. getting a Celery app instance).
Keep the main.py clean.

### `services/api/tests/test_routes.py`
Use `pytest` + `httpx.AsyncClient`. Mock `run_optimization.delay` — do not
start a real Celery worker in unit tests.

Required tests:
1. `POST /api/v1/optimize` with valid body returns HTTP 202 and a `task_id` field.
2. `POST /api/v1/optimize` with a prompt shorter than 10 chars returns HTTP 422.
3. `GET /api/v1/results/{id}` for a PENDING task returns `status: PENDING`.
4. `GET /api/v1/results/{id}` for a completed task returns an `OptimizationResponse`
   shaped dict with a `schedule` key.
5. `GET /health` returns HTTP 200.

---

## Stubs (Phase 1 only)

In Phase 1, the worker's `run_optimization` task uses hardcoded inputs because
the ML model and RAG pipeline don't exist yet. Mark every stub clearly:

```python
# STUB: replaced by tool_forecast_solar() in feat/agent-core
solar_forecast = [0.0] * 24

# STUB: replaced by tool_query_policies() in feat/agent-core
min_battery_buffer = 0.10

# STUB: replaced by market_prices table query in feat/agent-core
prices = [0.05] * 24
```

These stubs live in `services/workers/tasks.py`, not here.

---

## What Not to Build

- No solver logic
- No LLM calls
- No database writes (the worker does that)
- No authentication (out of scope for this project)
