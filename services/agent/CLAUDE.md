# CLAUDE.md — services/agent

You are working on the `feat/agent-core` branch of EcoGrid-Agent.

Your scope is `services/agent/` and the update to `services/workers/tasks.py`.
Do not touch any other service directory except where explicitly stated below.

This is the integration branch. All upstream services are merged to `main`:
- `services/solver/` — LP engine
- `services/api/` + `services/workers/` — FastAPI + Celery (with stubs)
- `services/ml/` — XGBoost solar forecaster
- `services/rag/` — Qdrant vector retrieval

Read all of the following before writing a single line of code:
1. `CONTEXT.md` — domain vocabulary, boundaries between technologies
2. `shared/contracts.py` — every type this service uses
3. `services/solver/__init__.py` — the solver's actual public API
4. `services/workers/tasks.py` — find the stubs you are replacing (marked `# STUB`)
5. `services/ml/inference_module.py` — `SolarForecaster` lives here, not `inference.py`
6. `services/rag/retriever.py` and `services/rag/parser.py` — the retrieval interface

---

## Repo State Awareness — Read This First

Before writing code, run these and adjust if reality differs:

```bash
# Confirm actual solver public API
python -c "from services.solver import *; help(optimize_battery_schedule)"

# Confirm SolarForecaster import path
python -c "from services.ml.inference import SolarForecaster; print('ok')"

# Confirm RAG retriever interface
python -c "from services.rag.retriever import PolicyRetriever; print('ok')"

# Find the exact stub markers in tasks.py
grep -n "STUB" services/workers/tasks.py
```

The solver's actual public function is `optimize_battery_schedule(SolverConstraints)`
— not `GridSolver().solve(SolverInput)`. Use whatever `services/solver/__init__.py`
actually exports. Do not assume the spec matches the code.

---

## What This Service Is

The LangChain orchestration layer. It receives a natural language prompt and
coordinates three tools — policy retrieval, solar forecasting, LP optimization —
to produce a structured, auditable result.

The agent does no math. It reads language, decides which tools to call, passes
the right parameters, and synthesises the final summary. If you find yourself
writing arithmetic in the agent layer, stop — that belongs in a tool.

---

## What to Build

### `services/agent/config.py`
Settings via `pydantic-settings`:
```python
OPENAI_API_KEY: str          # required, no default
OPENAI_MODEL: str = "gpt-4o"
AGENT_MAX_ITERATIONS: int = 5
AGENT_TEMPERATURE: float = 0.0   # deterministic — this is an engineering system
```

### `services/agent/prompts.py`
System prompt as a module-level string constant `SYSTEM_PROMPT`.

The prompt must communicate:
1. The agent's role: orchestrate tools to produce a battery schedule
2. The mandatory tool call order: policies first → solar second → optimize third
3. The hard rule: **no math in the agent**. Extract numbers from tool outputs,
   pass them to the next tool. Never compute or estimate values yourself.
4. The output format: structured JSON matching `OptimizationResponse`
5. What to do on INFEASIBLE: return the result honestly with a clear summary
   explaining why no schedule was possible — do not retry with relaxed constraints
   unless explicitly asked

Keep the prompt under 400 tokens. Verbose system prompts dilute instruction-following.

### `services/agent/tools.py`
Three LangChain tools. Each tool has a clear docstring — the LLM reads these
to decide when to call each one.

**`tool_query_policies(query: str) -> dict`**

Calls `PolicyRetriever().retrieve(query, top_k=1)`.
Calls `extract_buffer_constraint(chunk.text)`.
Returns:
```python
{
    "doc_id": chunk.doc_id,
    "doc_title": chunk.doc_title,
    "policy_text": chunk.text,
    "min_battery_buffer": buffer or 0.10,   # default if parser returns None
    "retrieval_score": chunk.score,
}
```

**`tool_forecast_solar(date: str) -> dict`**

Imports `SolarForecaster` from `services.ml.inference` (the `__init__.py`
re-export — do not import from `inference_module` directly).
Calls `SolarForecaster().forecast(date)`.
Returns:
```python
{
    "date": result.date,
    "hourly_forecast_kw": result.hourly_forecast_kw,
    "peak_kw": result.peak_kw,
    "total_kwh": result.total_kwh,
}
```

**`tool_optimize_grid(solver_input: dict) -> dict`**

Accepts a dict that maps to whatever the solver's actual input type expects.
Deserialise it, call the solver, return the result as a dict.

The solver's actual API (check `services/solver/__init__.py`) is
`optimize_battery_schedule(constraints: SolverConstraints) -> SolverResult`.
Use that — not `GridSolver().solve()` unless that's what the code actually exports.

Returns the solver result serialised to dict.
INFEASIBLE is a valid return — do not raise, do not retry automatically.

### `services/agent/agent.py`

**Class: `GridOptimizationAgent`**

```python
class GridOptimizationAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model=settings.OPENAI_MODEL, temperature=settings.AGENT_TEMPERATURE)
        self.tools = [tool_query_policies, tool_forecast_solar, tool_optimize_grid]
        self.agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(agent=self.agent, tools=self.tools,
                                      max_iterations=settings.AGENT_MAX_ITERATIONS,
                                      return_intermediate_steps=True)

    def run(self, prompt: str, objective: str, date: str) -> OptimizationResponse:
        ...
```

Use `return_intermediate_steps=True` — you need the tool call sequence for the
audit trail.

After the executor finishes, build `AuditTrail` from intermediate steps:
- `policy_doc_retrieved` — from `tool_query_policies` output
- `policy_text` — from `tool_query_policies` output
- `constraint_injected` — `{"min_battery_buffer": float}`
- `solar_forecast_used` — from `tool_forecast_solar` output
- `solver_status` — from `tool_optimize_grid` output
- `solver_time_ms` — from `tool_optimize_grid` output
- `tool_call_sequence` — ordered list of tool names called

Build `HourlyAction.reason` strings here — the agent layer adds the natural
language explanation to each schedule entry based on price and SoC context.

Return a complete `OptimizationResponse` from `shared.contracts`.

### Update `services/workers/tasks.py`

This is the only file outside `services/agent/` you may modify.

Find the stub function (search for `# STUB` markers and `_build_stub_constraints`
or equivalent). Replace it with:

```python
from services.agent.agent import GridOptimizationAgent

# Inside run_optimization_pipeline task:
request = OptimizeRequest(**request_dict)
agent = GridOptimizationAgent()
result = agent.run(
    prompt=request.prompt,
    objective=request.objective.value,
    date=request.date or _tomorrow_iso(),
)
return result.model_dump()
```

The task name, signature, and error handling wrapper stay exactly as they are.
Only the stub body changes.

### `services/agent/tests/test_agent.py`

Mock all three tools — do not make real LLM calls or real service calls in
unit tests.

Required tests:
1. `tool_query_policies` returns a dict with `min_battery_buffer` key that is
   a float between 0.0 and 1.0.
2. `tool_forecast_solar` returns a dict with `hourly_forecast_kw` of length 24.
3. `tool_optimize_grid` with a valid input dict returns a dict with `status` key.
4. `GridOptimizationAgent.run()` with mocked tools and mocked LLM returns an
   `OptimizationResponse` with a non-empty `audit.tool_call_sequence`.
5. When `tool_optimize_grid` returns `status=INFEASIBLE`, `agent.run()` still
   returns a valid `OptimizationResponse` (no exception, `summary` explains
   the infeasibility in plain language).

---

## End-to-End Smoke Test

After tests pass, run a manual full-stack test:

```bash
# 1. Start services
docker-compose up redis qdrant postgres -d

# 2. Seed vector DB (if not already seeded)
python scripts/seed_vector_db.py

# 3. Start worker (separate terminal)
celery -A services.workers.celery_app worker --loglevel=info

# 4. Start API (separate terminal)
uvicorn services.api.main:app --reload --port 8000

# 5. Send a request
curl -X POST http://localhost:8000/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Optimize battery for tomorrow. Hospital on site. Heatwave expected.", "objective": "MAXIMIZE_PROFIT"}'

# 6. Poll the result using the task_id from step 5
curl http://localhost:8000/api/v1/results/{task_id}
```

The final response must include:
- `audit.policy_doc_retrieved` — a real doc_id from `data/policies/`
- `audit.tool_call_sequence` — all three tools in order
- `schedule` — 24 entries
- `metrics.safety_constraints_passed: true`

Document the actual tool call sequence in the PR description.

---

## Definition of Done

- All 5 unit tests pass with mocked dependencies
- Manual end-to-end test produces a valid `OptimizationResponse` with full audit trail
- `services/workers/tasks.py` stubs are fully replaced (no `# STUB` comments remain)
- PR description includes the actual JSON response from the smoke test,
  with the hospital heatwave prompt from the README
