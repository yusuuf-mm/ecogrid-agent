# EcoGrid-Agent

![Python 3.11](https://img.shields.io/badge/python-3.11-blue?logo=python)
![Docker](https://img.shields.io/badge/docker-compose-2496ED?logo=docker)
![License](https://img.shields.io/badge/license-MIT-green)

FastAPI → Celery → Gemini SDK Orchestrator → [Qdrant, XGBoost, OR-Tools].  
An autonomous VPP orchestrator: natural language in, mathematically optimal battery schedule out.

---

## What It Does

Send a plain-English grid optimization request. The system retrieves the relevant regulatory policy from a vector database, forecasts tomorrow's solar generation with an XGBoost model, and runs a Google OR-Tools LP solver to produce a provably optimal 24-hour charge/discharge schedule. Every constraint in the solver is traced back to a policy document. Every decision is logged in the audit trail.

No LLM guesses the schedule. No solver reads policy documents. The LLM handles language and reasoning; the solver handles math and hard guarantees. The bridge between them is the engineering artifact this project exists to demonstrate.

---

## Live Example

```bash
curl -X POST http://localhost:8000/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Optimize battery for tomorrow. Hospital on site. Heatwave expected afternoon.",
    "objective": "MAXIMIZE_PROFIT"
  }'
```

Response (poll `GET /api/v1/results/{task_id}` after receiving 202 Accepted):

```json
{
  "status": "SUCCESS",
  "summary": "The optimal schedule was found using policy grid_safety_sop_02 with a 20% safety buffer.",
  "policy_retrieved": "grid_safety_sop_02",
  "constraint_injected": {"min_battery_buffer": 0.2},
  "solver_status": "OPTIMAL",
  "safety_passed": true,
  "profit_usd": 15.0,
  "tool_calls": ["tool_query_policies", "tool_forecast_solar", "tool_optimize_grid"],
  "schedule_hours": 24
}
```

---

## How It Works

```
[ User Prompt ]
      │
      ▼
[ FastAPI ] ──(enqueue task)──► [ Celery Worker ]
                                       │ (via Redis)
                                       ▼
         [ Gemini SDK Orchestrator ]
         deterministic tool sequence:
                             ├── tool_query_policies()   → Qdrant Vector DB
                             ├── tool_forecast_solar()   → XGBoost Model
                             └── tool_optimize_grid()    → Google OR-Tools LP
                                       │
                                       ▼
         [ Structured JSON Response ]
         stored in Redis, polled via API
```

**Data flows one direction.** The agent calls tools. Tools don't call each other. The solver never touches the vector DB. The ML model doesn't know the solver exists.

---

## Key Engineering Decisions

### LLM + LP Solver, not LLM alone

An LLM cannot enforce a hard constraint. If you ask it to maintain a 30% SoC floor while maximizing profit, it may approximate — and approximation in energy infrastructure means blackouts. The LP solver (Google OR-Tools GLOP) enforces every constraint as a linear inequality. The LLM's job is limited to parsing the user's intent and the policy document's language. The solver owns the math.

### Deterministic tool orchestration

The three tools — policy retrieval, solar forecast, LP optimization — run in fixed order. There is no agent loop deciding which tool to call next. This makes the audit trail predictable: every response records exactly three tool calls with their inputs, outputs, and durations. If a tool fails, the response captures the error at the exact point of failure. No hidden state from a multi-step reasoning loop.

### RAG-driven constraints

The safety buffer is never hardcoded. The solver receives `min_battery_buffer` as a parameter, and that parameter comes from a policy document retrieved from Qdrant at runtime. Changing the buffer (10% → 30% for hospitals) requires updating a text file and re-seeding the vector DB — no code change, no deployment. This mirrors how real grid operators reference regulatory SOPs.

---

## Tech Stack

| Layer               | Technology                          | Reason                                                     |
|---------------------|-------------------------------------|------------------------------------------------------------|
| Backend API         | FastAPI + Uvicorn                   | Async, typed, production-grade Python API                  |
| Task Queue          | Celery + Redis                      | LP solvers block; don't hold HTTP connections              |
| AI Orchestration    | Google Gemini SDK                   | Structured output via response_schema, deterministic tool orchestration, no agent loop |
| Vector DB           | Qdrant                              | Fast, lightweight, no external dependencies                |
| Embeddings          | fastembed (BAAI/bge-small-en-v1.5)  | No torch dependency, 50MB footprint, 384-dim vectors       |
| ML Forecasting      | XGBoost + scikit-learn              | Interpretable, fast, handles tabular features well         |
| OR Solver           | Google OR-Tools (GLOP/SCIP)         | Industry-standard LP/MIP solver                            |
| Structured Data     | PostgreSQL                          | Market prices, operational state                           |
| Containerization    | Docker + docker-compose             | One command to run everything                               |
| Package Management  | Poetry                              | Deterministic dependency resolution                        |
| Language            | Python 3.11+                        | Best library coverage for all three domains                |

---

## Branch Map

| Branch | Ownership |
|--------|-----------|
| `feat/or-solver` | Solver core |
| `feat/backend-api` | API and worker pipeline |
| `feat/ml-forecasting` | Solar forecasting model |
| `feat/vector-rag` | Retrieval and policy ingestion |
| `feat/agent-core` | Gemini orchestration and tool wrappers |
| `feat/docker-infra` | Docker and deployment scaffolding |

---

## Running It

### Full stack (recommended)

```bash
git clone https://github.com/yusuuf-mm/ecogrid-agent.git
cd ecogrid-agent
cp .env.example .env          # add GEMINI_API_KEY from aistudio.google.com
docker compose up --build
```

If you do not already have a key, get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com/).

This brings up `postgres`, `redis`, `qdrant`, `api` (FastAPI on :8000), and `worker` (Celery). Each service has a healthcheck.

#### First-run seeding

```bash
docker compose --profile seed run --rm ingest
```

Runs `scripts/seed_vector_db.py` then `scripts/seed_market_prices.py` in a one-shot container. Idempotent — safe to re-run.

#### Smoke test

```bash
curl -X POST http://localhost:8000/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Optimize battery for tomorrow. Hospital on site. Heatwave expected.",
    "objective": "MAXIMIZE_PROFIT"
  }'
# → 202 Accepted, {"task_id": "<uuid>", "status": "QUEUED"}

curl http://localhost:8000/api/v1/results/<task_id>
# poll until status is SUCCESS or FAILURE
```

### Local development (without Docker)

```bash
poetry install
docker compose up postgres redis qdrant -d
uvicorn services.api.main:app --reload --port 8000       # terminal 1
celery -A services.workers.celery_app worker --loglevel=info  # terminal 2
```

Set `GEMINI_API_KEY` in your environment before starting the worker or API process.

---

## Project Structure

```
ecogrid-agent/
├── shared/                   # Pydantic contracts shared across services
│   └── contracts.py
├── services/
│   ├── api/                  # FastAPI application
│   ├── workers/              # Celery task definitions
│   ├── solver/               # Google OR-Tools LP engine
│   ├── ml/                   # XGBoost solar forecast model
│   ├── rag/                  # Qdrant + RAG retrieval pipeline
│   └── agent/                # Gemini orchestrator + tool wrappers
├── data/
│   ├── raw/                  # Source data (NREL solar, ERCOT LMPs)
│   ├── processed/            # Training-ready datasets
│   ├── policies/             # Grid regulatory text documents
│   └── models/               # Trained model artifacts (.joblib)
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── infra/
│   ├── docker/               # Per-service Dockerfiles
│   └── compose/              # docker-compose variants
├── scripts/                  # Seeding, setup, data download utilities
└── docs/
    ├── adr/                  # Architecture Decision Records
    └── agents/               # Agent skill configuration
```

---

## What This Demonstrates

**For deep-tech and AI-first companies**: The pipeline architecture (FastAPI → Celery → agent → tools) shows production engineering instincts. Tool isolation, typed contracts between every service, and a non-negotiable audit trail show a system designed to stay debuggable at scale. The Gemini `response_schema` pattern eliminates string-parsing risk in structured LLM output.

**For energy and supply chain companies**: The LP solver — fully parameterized, configurable objective (`MAXIMIZE_PROFIT` / `MINIMIZE_CARBON` / `MINIMIZE_COST`), INFEASIBLE as a first-class response — shows OR as a living skill, not a textbook exercise. The RAG pipeline turns regulatory documents into solver constraints without hardcoding.

**For infrastructure and government contractors**: The RAG-to-constraint pipeline means policy changes don't require code changes. Update the text file, re-seed Qdrant, and the next optimization uses the new constraint automatically. The audit trail satisfies the traceability requirement that regulated environments mandate.

---

## License

MIT
