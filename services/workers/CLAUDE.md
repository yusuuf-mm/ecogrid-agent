# CLAUDE.md — services/workers

You are working on the `feat/backend-api` branch of EcoGrid-Agent.

This file covers the Celery worker side of the branch. Read `services/api/CLAUDE.md`
as well — both directories are part of the same branch and build together.

Your scope: `services/workers/` only.

---

## What This Service Is

The async execution layer. A Celery worker picks up optimization tasks from the
Redis queue, runs them, and stores the result back in Redis for the API to serve.

In Phase 1 (this branch), the worker calls the solver directly with stubbed inputs.
In Phase 3 (feat/agent-core), those stubs are replaced by the LangChain agent.
The worker's public interface — task name, input shape, result shape — does not
change between phases.

---

## What to Build

### `services/workers/celery_app.py`
Celery application instance.

```python
from celery import Celery
from services.api.config import Settings

settings = Settings()

celery_app = Celery(
    "ecogrid",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=3600,  # results live for 1 hour
)
```

### `services/workers/tasks.py`
One task: `run_optimization`.

```python
@celery_app.task(name="run_optimization", bind=True)
def run_optimization(self, request_data: dict) -> dict:
    ...
```

Phase 1 implementation:
1. Deserialise `request_data` into `OptimizeRequest` (from `shared.contracts`)
2. Build a `SolverInput` using stubbed values (mark each stub with a comment as
   shown in `services/api/CLAUDE.md`)
3. Call `GridSolver().solve(solver_input)`
4. Build an `OptimizationResponse` from the result
5. Return `response.model_dump()`

Log at the start and end of every task execution:
```
[task_id] run_optimization started
[task_id] run_optimization finished — status=OPTIMAL, solver_time=42ms
```

Use Python's stdlib `logging`, not print statements.

### `services/workers/tests/test_tasks.py`
Required tests:
1. `run_optimization` with a valid `OptimizeRequest` dict returns a dict with a
   `status` key equal to `"OPTIMAL"` or `"INFEASIBLE"`.
2. The returned dict has `schedule`, `metrics`, and `audit` keys.
3. `run_optimization` with an empty prompt dict raises a `ValidationError`
   (Pydantic catches malformed input before the solver runs).

Run these with `pytest services/workers/tests/`. Do not start Redis — call the
task function directly, not via `.delay()`.

---

## Phase 3 Handoff Note

When `feat/agent-core` is built and merged, `tasks.py` gets updated. The stub
inputs are replaced by `GridOptimizationAgent().run(...)`. The task signature
(name, input dict, output dict) stays identical. The API layer never changes.

This is why the stubs must be clearly marked — the agent branch author needs to
know exactly what to replace without reading the whole file.
