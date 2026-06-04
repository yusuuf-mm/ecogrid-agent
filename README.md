# EcoGrid-Agent

An autonomous Virtual Power Plant (VPP) orchestrator. An LLM reasoning layer receives natural
language grid optimization requests, retrieves regulatory constraints from a vector database,
pulls a solar generation forecast from an ML model, then passes all parameters into a
deterministic LP solver. The output is a mathematically verified, auditable charge/discharge
schedule — not a suggestion.

This is a portfolio project. The engineering thesis is simple: LLMs are good at language and
reasoning. OR solvers are good at math and hard constraints. Connecting them properly, with
clean interfaces and full audit trails, is the actual work.

---

## What It Does

Send a natural language request:

```
POST /api/v1/optimize
{
  "prompt": "Optimize battery ops for tomorrow. Heatwave expected afternoon. Hospital must have zero risk of power loss."
}
```

The system returns a `task_id` immediately (202 Accepted). Poll for the result:

```
GET /api/v1/results/{task_id}
```

Result:

```json
{
  "status": "OPTIMAL",
  "summary": "Grid optimized for heatwave conditions. Enforced 30% hospital reserve per Policy doc_id_442.",
  "schedule": [
    { "hour": "02:00", "action": "CHARGE_FROM_GRID", "rate_kw": 200.0, "reason": "LMP at lowest ($0.04/kWh)" },
    { "hour": "13:00", "action": "STORE_SOLAR",      "rate_kw": 250.0, "reason": "Peak solar 450kW" },
    { "hour": "15:00", "action": "HOLD_RESERVE",     "rate_kw": 0.0,   "reason": "Heatwave peak. 30% buffer for hospital (doc_442)" },
    { "hour": "18:00", "action": "DISCHARGE_TO_GRID","rate_kw": 230.5, "reason": "Peak grid demand. LMP at $0.24/kWh" }
  ],
  "metrics": {
    "expected_revenue_usd": 1420.00,
    "carbon_saved_kg": 340.5,
    "safety_constraints_passed": true
  },
  "audit": {
    "policy_doc_retrieved": "doc_id_442",
    "policy_text": "Critical infrastructure must maintain minimum 30% SoC buffer during declared anomalies.",
    "constraint_injected": { "min_battery_buffer": 0.30 },
    "solver_status": "OPTIMAL",
    "solver_time_ms": 42
  }
}
```

The agent doesn't guess. Every constraint it enforces links back to a document. Every schedule
it produces is mathematically optimal given those constraints.

---

## Architecture

```
[ User Prompt ]
      │
      ▼
[ FastAPI ] ──(enqueue task)──► [ Celery Worker ]
                                       │ (via Redis)
                                       ▼
                             [ LangChain Agent ]
                             calls three tools:
                             ├── tool_query_policies()   → Qdrant Vector DB
                             ├── tool_forecast_solar()   → XGBoost Model
                             └── tool_optimize_grid()    → Google OR-Tools LP
                                       │
                                       ▼
                             [ Structured JSON Response ]
                             stored in Redis, polled via API
```

**Data flows one direction.** The agent calls tools. Tools don't call each other. The solver
never touches the vector DB. The ML model doesn't know the solver exists.

---

## Tech Stack

| Layer               | Technology                    | Reason                                              |
|---------------------|-------------------------------|-----------------------------------------------------|
| Backend API         | FastAPI + Uvicorn             | Async, typed, production-grade Python API           |
| Task Queue          | Celery + Redis                | LP solvers block; don't hold HTTP connections       |
| AI Orchestration    | LangChain                     | Tool-use agent loop with structured outputs         |
| Vector DB           | Qdrant                        | Fast, lightweight, no external dependencies         |
| Embeddings          | `all-MiniLM-L6-v2`           | Local inference, no API calls, fast                 |
| ML Forecasting      | XGBoost + scikit-learn        | Interpretable, fast, handles tabular features well  |
| OR Solver           | Google OR-Tools (GLOP/SCIP)  | Industry-standard LP/MIP solver                     |
| Structured Data     | PostgreSQL                    | Market prices, operational state                    |
| Containerization    | Docker + docker-compose       | One command to run everything                       |
| Package Management  | Poetry                        | Deterministic dependency resolution                 |
| Language            | Python 3.11+                  | Best library coverage for all three domains         |

---

## Branch Structure

Each branch owns one service. Shared types live in `shared/` and merge to `main` first.

| Branch                       | Service Directory              | What It Builds                                    |
|------------------------------|-------------------------------|---------------------------------------------------|
| `feat/phase-1-or-solver`     | `services/solver/`            | LP/MIP engine, constraint model, solver contracts |
| `feat/phase-1-backend-api`   | `services/api/` + `services/workers/` | FastAPI routes, Celery tasks           |
| `feat/phase-2-ml-forecasting`| `services/ml/`                | XGBoost model, training pipeline, inference       |
| `feat/phase-2-vector-rag`    | `services/rag/`               | Qdrant ingestion, RAG retrieval, param parsing    |
| `feat/phase-3-agent-core`    | `services/agent/`             | LangChain agent, tool wrappers, audit logging     |
| `feat/phase-4-docker-infra`  | `infra/`                      | Dockerfiles, docker-compose, scripts, CI          |

Branches merge into `main` via PR. Never merge feature branches into each other.

---

## Running It

### Full stack (recommended)

```bash
git clone https://github.com/your-username/ecogrid-agent.git
cd ecogrid-agent
cp .env.example .env          # fill in your OpenAI API key + any overrides
docker-compose up --build
```

Seed the vector DB and database on first run:

```bash
docker-compose exec api python scripts/seed_vector_db.py
docker-compose exec api python scripts/seed_market_prices.py
```

### Local development (without Docker)

```bash
poetry install
poetry shell

# Start dependencies (Redis + Qdrant + Postgres)
docker-compose up redis qdrant postgres -d

# Run API
uvicorn services.api.main:app --reload --port 8000

# Run Celery worker (separate terminal)
celery -A services.workers.celery_app worker --loglevel=info
```

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
│   └── agent/                # LangChain orchestrator + tools
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

## Data Sources

All publicly available, no API keys required for data:

- **Solar irradiance**: [NREL NSRDB](https://nsrdb.nrel.gov/) — historical hourly CSV downloads
- **Market prices**: [ERCOT historical LMPs](https://www.ercot.com/mktinfo/prices) — hourly wholesale prices, free download
- **Regulatory policies**: Fabricated SLA documents in `data/policies/` that mimic real FERC/NERC formatting

---

## What This Demonstrates

**For deep-tech and AI-first companies**: The async pipeline (FastAPI → Celery → agent → tools)
shows production instincts. The tool isolation pattern shows you know how to build systems that
stay debuggable when they grow.

**For energy and supply chain companies**: The LP solver with configurable objectives and
constraint sourcing shows OR as a living skill, not a textbook memory.

**For infrastructure and government contractors**: The RAG-to-constraint pipeline shows how
regulatory documents can drive system behavior automatically — no manual re-coding when policy
changes.

The specific design decision that this project is built to demonstrate: **an LLM agent's job is
reasoning and orchestration; a deterministic solver's job is math and hard guarantees**. They
are not alternatives. They are complements. Building the bridge between them is the engineering.

---

## License

MIT
