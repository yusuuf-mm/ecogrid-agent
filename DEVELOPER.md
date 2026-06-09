# EcoGrid-Agent — Developer Deep Dive

## 1. Why I Built This

I kept seeing the same demo. Someone types "optimize my battery schedule" into a chatbot, the chatbot writes a plausible looking table of numbers, and everyone nods like that's a production system. But energy storage is not a creative writing exercise. If you discharge when you should have charged, you lose money. If you drain the reserve for a hospital, people can die. You cannot approximate your way through hard constraints.

This project started from a simple observation: LLMs are excellent at understanding context and synthesizing language, but they cannot do arithmetic reliably. OR solvers are the opposite. They produce provably optimal solutions within strict constraints, but they need structured input and cannot read a regulatory document. The gap between them is where the actual engineering work lives.

I chose energy grid optimization because it has a clean boundary between the two domains. The language side is parsing user requests and policy documents. The math side is a well-defined linear program with known constraints and a measurable objective. Neither side is particularly hard on its own. Connecting them with typed contracts, an audit trail, and no shared mutable state is the part worth building.

The engineering thesis is this: an LLM agent's job is reasoning and orchestration. A deterministic LP solver's job is math and hard guarantees. They are not alternatives. If you treat them as one or the other, you get either a fragile demo or an inflexible calculator. Building the bridge between them, with clean service boundaries and a non-negotiable audit trail, is what this system demonstrates.

---

## 2. System Architecture — How It Actually Works

The full request lifecycle runs through ten stages. Here is the path a user's prompt takes from curl to schedule.

### Stage 1: API receives the request

`POST /api/v1/optimize` accepts a JSON body with `prompt`, optional `date`, and `objective`. The FastAPI route handler does exactly three things: validate the input against `OptimizationRequest`, generate a UUID, and enqueue a Celery task with the validated dict. It returns HTTP 202 with `{"task_id": "<uuid>", "status": "QUEUED"}`. No solver runs inside the HTTP request. That would block the worker for seconds and waste a connection.

### Stage 2: Celery picks up the task

Redis holds the task queue and the result store. The Celery worker runs `run_optimization_pipeline`, which deserializes the request and hands it to `GridOptimizationAgent.run()`. I chose Celery because LP solvers block. A synchronous FastAPI endpoint that takes two to ten seconds per request would need absurd numbers of workers just to stay responsive. Celery offloads that to a background pool and lets the API stay stateless.

### Stage 3: Gemini parses intent

The agent calls `client.models.generate_content` with `response_schema` set to `_AgentIntent`. This is the first of three phases. The model receives the user's natural language prompt and must extract two fields: `policy_query` (a short phrase for vector retrieval) and `target_date` (the ISO date to forecast). The `response_schema` parameter enforces the output shape at the API level. No JSON parsing, no regex, no NoneType risk. If Gemini returns a response, it is a valid `_AgentIntent` instance.

I chose this pattern over LangChain's tool-calling agent for a specific reason: the tool order is fixed. There is no need for an LLM to decide whether to call the policy tool or the solar tool first. The answer is always policies first. A deterministic three-phase pipeline is more auditable, faster, and cannot get stuck in a reasoning loop.

### Stage 4: Policy retrieval

`tool_query_policies` takes the extracted query phrase and calls `PolicyRetriever.retrieve()`. The retriever embeds the query with fastembed (BAAI/bge-small-en-v1.5, 384 dimensions), runs a cosine similarity search against the Qdrant `grid_policies` collection, and passes each retrieved chunk through `extract_buffer_constraint`. If the parser finds a value, `constraint_float` is set to that value and `parse_confidence` is "parsed". If not, it defaults to `0.10` with confidence "fallback".

### Stage 5: Solar forecast

`tool_forecast_solar` takes the target date and calls the XGBoost model. The model was trained on NREL NSRDB historical solar irradiance data. It returns 24 hourly values in kW. The tool wraps the raw forecast into a dict with `hourly_forecast_kw`, `peak_kw`, and `total_kwh`.

The solar forecast is currently the weakest link. The ML service has a trained model artifact, but the tool still passes dummy weather features (20 C, 30% cloud cover, zero irradiance) because the weather data pipeline is not wired to a live source. The forecast falls back to zeros if the model raises `NotImplementedError`. I will come back to this.

### Stage 6: LP solver

`tool_optimize_grid` receives a dict assembled from the policy and solar outputs plus default values for battery capacity, charge rate, and initial SoC. It deserializes these into `SolverConstraints` and calls `optimize_battery_schedule` (the adapter function that wraps `GridSolver.solve()`). The solver returns `SolverResult` with status, schedule, and metrics.

### Stage 7: Summary synthesis

The agent calls Gemini again, this time with `response_schema` set to `_AgentSummary`. This is phase three. The model receives the solver status, policy doc ID, buffer percentage, and tool call count. It must produce one plain-language sentence summarizing the result. If Gemini fails (network error, rate limit), the agent falls back to a deterministic `_summarize` function that produces a correct but less elegant summary.

### Stage 8: Response assembly

The agent constructs `OptimizationResponse` with the full `AuditTrail`. Every tool call is recorded with its input, output, and duration in milliseconds. The audit trail is not optional. An energy system that cannot explain its decisions does not belong in production.

### Stage 9: Redis storage

The Celery task stores the serialized `OptimizationResponse` in Redis keyed by task ID. The result expires after one hour.

### Stage 10: Result polling

The client calls `GET /api/v1/results/{task_id}`. The API reads Redis and returns the current state. If the task is still running, status is `RUNNING`. If finished, it returns the full response with the schedule.

### Why async matters

The LP solve takes 10 to 50 milliseconds. The network calls to Qdrant and Gemini add maybe 500 milliseconds each. Total wall time is under two seconds in the common case. That is still too long to hold an HTTP connection open under load. By returning 202 immediately and polling for the result, the API can handle hundreds of concurrent requests without connection backpressure. Celery also gives retry semantics and dead-letter handling for free.

---

## 3. The LP Formulation

The solver is a 24-hour linear program with hourly time steps. I will walk through the math and explain each decision.

### Decision variables

Three variables per hour, 72 total:

- `c[t]` — charge power from the grid in kW. Range: [0, max_charge_rate].
- `d[t]` — discharge power to the grid in kW. Range: [0, max_charge_rate].
- `soc[t]` — battery state of charge at the end of hour t in kWh. Range: [buffer, capacity].

There is a fourth variable `s[t]` for solar absorbed by the battery. I added this because the physics equation would force every forecast kWh of solar into the battery otherwise. When solar generation exceeds what the battery can store given its current SoC and charge rate, the surplus is curtailed. Without the curtailment variable, the LP would be infeasible on sunny days.

### Objective function

Default: maximize profit over the 24-hour window.

```
maximize Σ (d[t] * price[t] - c[t] * price[t]) for t = 0..23
```

Discharge earns revenue. Charge costs money. Solar absorption costs nothing. The solver naturally prefers to discharge during high-price hours and charge during low-price hours. It also prefers to absorb solar whenever possible since it is free.

The other objectives change the function:

- MINIMIZE_COST flips the sign: minimize Σ (c[t] * price[t] - d[t] * price[t]).
- MINIMIZE_CARBON minimizes Σ (c[t] * carbon_intensity[t]). Without carbon intensity data, the solver charges nothing and the result is uninteresting but technically correct.

### Constraints

Five constraints, all linear:

1. **Physics**: `soc[t] = soc[t-1] + c[t] + s[t] - d[t]`. Initial condition: soc[-1] = initial_soc. This is the state transition. Energy in minus energy out equals the change in stored energy.

2. **Capacity bounds**: `buffer <= soc[t] <= capacity`. The lower bound is the safety buffer from policy. The upper bound is the battery's physical capacity.

3. **Rate bounds**: `0 <= c[t], d[t] <= max_rate`. The battery cannot charge or discharge faster than its hardware allows.

4. **No simultaneous charge and discharge**: `c[t] + d[t] <= max_rate`. In a continuous LP this is technically redundant with the individual rate bounds, but it enforces the spec's intent: the battery is either charging, discharging, or idle, never both.

5. **Solar absorption bounds**: `0 <= s[t] <= solar_forecast[t]`. The battery cannot absorb more solar than the panels generate.

### Infeasibility

When the LP returns INFEASIBLE, it means no assignment of decision variables satisfies all constraints simultaneously. This happens most commonly when the safety buffer exceeds what the battery can maintain given its starting charge. For example, a hospital policy requiring 80% SoC reserve with the battery starting at 50%.

INFEASIBLE is not an exception. It is a valid `SolverResult` with status set to `INFEASIBLE` and a human-readable `reason` string. The agent receives it, logs it, and includes the explanation in the response summary. The system does not retry with relaxed constraints unless the user explicitly requests it.

### GLOP versus SCIP

The current solver uses GLOP, Google OR-Tools' continuous LP solver. GLOP is fast and sufficient for this problem because all decision variables and constraints are linear and continuous. I would switch to SCIP (a MIP solver) if the model needed binary decisions on/off for individual generators or minimum runtime constraints for thermal plants. That would be an ADR-level decision because SCIP has different performance characteristics and requires a separate binary.

---

## 4. The RAG Pipeline — Design Decisions

### Why policy documents, not hardcoded values

The safety buffer is a regulatory parameter that changes by scenario and jurisdiction. A default of 10% applies in standard conditions. A hospital requires 30%. A maintenance window allows 5%. Hardcoding these in the solver would mean a code deploy every time a policy changes. Storing them as text documents in Qdrant means updating a file and re-seeding the vector DB. No code change, no deployment pipeline, no risk of introducing a solver bug during a policy update.

### Chunking strategy

Documents are split on double newlines to recover paragraphs. I ran into a problem early on: section headers like "Section 3 -- Hospital Reserve Requirements" are short standalone paragraphs that contain no constraint value. When stored as separate chunks, retrieval could return the header instead of the paragraph with the actual number.

The fix merges any paragraph shorter than 60 characters with the paragraph that follows it. This is a heuristic. It assumes headers are always short and always precede their content. It works for the five policy documents in the repository. It would fail for a document with a legitimately short paragraph followed by an unrelated short paragraph. I have not encountered that case yet.

### fastembed over sentence-transformers

sentence-transformers requires torch. The full torch package is roughly 2 GB. On a resource-constrained machine, that matters. fastembed is around 50 MB, uses ONNX runtime internally, and produces the same 384-dimensional embeddings that Qdrant expects. The tradeoff is model selection: I use BAAI/bge-small-en-v1.5 instead of all-MiniLM-L6-v2. Both produce compatible vector sizes, so the collection schema is unchanged. If I had a machine with more disk and a GPU, I would switch back to sentence-transformers for marginally better retrieval accuracy.

### Two-phase retrieval

The retrieval pipeline has two phases separated by a parser. Phase one: embed the query, search Qdrant, return the raw text chunk and metadata. Phase two: run `extract_buffer_constraint` on the chunk text to find the numeric safety buffer.

The parser handles five formats: "30%", "0.30", "30 percent", "thirty percent", and "five percent". It searches for the phrase "minimum state of charge buffer of" first, then falls back to any percentage mention in the text. The regex patterns handle all five policy documents correctly.

### parse_confidence and the fallback

When the parser cannot extract a value from a chunk, `parse_confidence` is set to "fallback" and `constraint_float` defaults to 0.10. The fallback is 0.10, not 0.0, because a missing policy should default to the standard operating procedure, not to an unsafe zero-buffer state. A zero buffer means the battery can drain completely, which violates the core safety invariant of the system.

---

## 5. What I Would Do Differently

This section is honest about what is not finished and what I would change starting over.

### The solver's public API inconsistency

The solver has two public entry points. `GridSolver.solve()` takes an internal `SolverInput` namedtuple. `optimize_battery_schedule()` takes a shared `SolverConstraints` Pydantic model and wraps the internal solver. The agent calls the adapter function. The tests call the class directly. This dual interface exists because I built the solver first with its own types, then introduced `shared/contracts.py` later. If I were starting over, the solver would consume `SolverConstraints` directly and the internal types would not exist.

### Python 3.14 incompatibility

The project targets Python 3.11. During an upgrade attempt to 3.14, LangChain and several of its transitive dependencies had not published compatible wheels. Google OR-Tools also lagged. The lock file could not resolve. I reverted to 3.11 and pinned it. This is not a problem with 3.11 itself, but it means upgrading will require coordination across multiple dependency maintainers.

### The chunking bug that separated headers from content

This is the bug I mentioned in section 4. The original chunking code split on double newlines and stored each paragraph independently. Section headers like "Section 3 -- Hospital Reserve Requirements" became separate chunks with no constraint value. When the retriever returned the header chunk, the parser returned None and the system fell back to the 0.10 default. The system was effectively ignoring the hospital policy. The fix was a header-merging heuristic, which works for this data but is not general. A better approach would be to include the full section when a header is detected rather than merging blindly.

### Market prices from Postgres not wired to the solver

The solver accepts `market_prices_kwh` as an input array, and the database has a `market_prices` table seeded with synthetic ERCOT data. But the agent hardcodes a flat `_FALLBACK_PRICE_KWH = [0.05] * 24` instead of querying Postgres at runtime. The database query logic exists in the API layer but is not connected through the agent. The tool currently passes the fallback every time. Wiring Postgres into the agent is a single function call, but I have not done it yet because the pipeline works end to end without it and the priority was completing the full chain first.

### Solar forecast returning zeros

The XGBoost model is trained and the artifact is saved. But `tool_forecast_solar` creates `WeatherFeatures` with dummy values (20 C, 30% cloud cover, 0 W/m2 irradiance) because there is no weather data pipeline feeding it real forecasts. The solar forecast returns 24 zeros. The LP still produces a valid schedule with zero solar generation. It just misses the opportunity to charge for free during sunny hours. Fixing this requires either a weather API integration or a synthetic data generator that produces plausible forecasts from historical patterns.

### What proper CI/CD would look like

The project has no CI pipeline. There is no GitHub Actions workflow, no automated test run, no lint check, no Docker build test. A production version would need:

- PR workflows that run unit tests, mypy, and ruff
- Integration tests against ephemeral Docker services
- A staging environment with a real Qdrant instance and a test database
- Container image builds pushed to a registry
- Healthcheck-based deployment with rolling updates

None of these are conceptually difficult. They are absent because the project has not crossed the threshold where manual testing becomes more expensive than writing the CI config.

---

## 6. Mock Interview Q&A

**Q: Why not just use the LLM to generate the schedule directly?**

A: Because LLMs cannot enforce hard constraints. If you ask it to maintain a 30% SoC floor, it might approximate. In energy infrastructure, approximation means blackouts. The LP solver enforces every constraint as a linear inequality. The answer is provably optimal or provably infeasible. There is no maybe.

**Q: How does the system handle a case where no policy document matches?**

A: The retriever returns top-k results from Qdrant. If the top result is a poor semantic match, the constraint parser will fail to extract a value. In that case, `parse_confidence` is "fallback" and `constraint_float` defaults to 0.10, which represents standard operating conditions. The audit trail records which chunk was retrieved and whether parsing succeeded, so an operator can see that the policy match was weak.

**Q: What happens if the LP solver returns INFEASIBLE?**

A: INFEASIBLE is a valid `SolverResult`, not an exception. The adapter function returns it with a human-readable reason string, the agent includes it in the response summary, and the API returns it as a successful HTTP response with status FAILURE. The system does not retry with relaxed constraints unless the user explicitly asks. The audit trail captures the solver status and reason.

**Q: How would you scale this to handle 1000 concurrent optimization requests?**

A: The API is stateless and returns immediately, so it scales horizontally behind a load balancer with no changes. Celery workers scale independently by adding more container replicas. Redis handles the task queue and result storage. The bottleneck would be Qdrant, which runs on a single node in the current setup. For 1000 concurrent requests, Qdrant would need a cluster configuration. Gemini API calls are rate-limited per API key. The system would need either multiple keys with round-robin routing or a caching layer for repeated policy queries.

**Q: Why Celery and Redis instead of a simpler synchronous approach?**

A: The LP solve takes 10 to 50 milliseconds. Network calls to Qdrant and Gemini add maybe 500 milliseconds each. Total is under two seconds. That is too long to hold an HTTP connection under load. A synchronous approach would require thread-per-request or async I/O, and the blocking solver call would defeat both. Celery separates the concern: FastAPI serves requests, workers crunch numbers, Redis stores results. Each component scales independently.

**Q: What is the difference between your RAG retrieval and just doing a keyword search?**

A: Keyword search on the phrase "minimum state of charge buffer" would find the right chunk if the phrase appears verbatim. But the user's prompt might say "hospital reserve" or "critical infrastructure" or "heatwave protocol". These are semantically related but lexically different. The embedding model captures that relationship. The query "hospital reserve" will return a chunk about "minimum state of charge buffer of 30%" because the model maps both phrases to nearby regions in the embedding space, even though they share no keywords.

**Q: How do you ensure the solver's output is actually optimal and not just a good solution?**

A: GLOP is a simplex-based LP solver. When it returns OPTIMAL status, the solution is provably optimal: no other assignment of decision variables can produce a higher (or lower) objective value while satisfying all constraints. The proof is in the dual variables and reduced costs, which GLOP computes internally. The system records the solver status and elapsed time in the audit trail. An OPTIMAL status is a mathematical guarantee, not a heuristic estimate.

**Q: If you were to productionize this, what would you add first?**

A: CI/CD, no question. Manual testing works for a portfolio project. For production, every change needs automated test coverage, type checking, linting, and container build verification before it reaches a deployment environment. After CI/CD, I would wire Postgres market prices into the agent. That is the largest correctness gap in the current system, because the flat price fallback means the solver optimizes against fake data.

**Q: Walk me through what happens when a new regulatory policy is introduced.**

A: Someone writes the policy as a text file following the existing format, puts it in `data/policies/`, and runs `python scripts/seed_vector_db.py`. The seed script chunks the document, embeds each chunk with fastembed, and upserts the vectors into Qdrant. The next optimization request that matches this policy's semantic content will retrieve the new chunk, parse its constraint value, and feed it to the solver. No code changes, no deployment, no service restart.

**Q: Why XGBoost for solar forecasting instead of a neural network?**

A: The data is tabular (hourly temperature, cloud cover, irradiance) with a relatively small training set. XGBoost handles tabular data well, trains faster than a neural network on CPU, and produces interpretable feature importance scores. A neural network would be appropriate if the dataset grew to include satellite imagery or time-series sequences long enough to benefit from an LSTM. At the current scale, XGBoost is the pragmatic choice.

**Q: How does the audit trail help with regulatory compliance?**

A: Every optimization response includes `audit.agent_tool_calls` listing every tool invocation with input, output, and duration. It also records which policy document was retrieved, the raw text chunk, the parsed constraint value, the solver status, and the solver execution time. A regulator can inspect any historical optimization and see exactly which policy drove which constraint, which forecast fed the solver, and whether the solver found a feasible solution. This traceability is the minimum bar for regulated energy infrastructure.

**Q: What is the weakest part of the current system?**

A: The solar forecast, which returns zeros because the weather data pipeline is not connected to a live source. The system produces a valid schedule without solar generation, but it is leaving money on the table by not capturing free solar energy during sunny hours. The fix requires either a weather API integration or a synthetic forecast generator. Neither is architecturally hard. Both are blocked by the priority order I chose.

**Q: How did you handle the Python 3.14 / LangChain incompatibility?**

A: I dropped LangChain entirely. The agent originally used LangChain's `AgentExecutor` with `create_tool_calling_agent` backed by OpenAI. When I tried upgrading, LangChain had not released 3.14-compatible wheels and OR-Tools had the same issue. Instead of fighting dependency resolution, I rewrote the agent to use the `google-genai` SDK directly with `client.models.generate_content` and `response_schema` for structured output. The rewrite eliminated the dependency on LangChain entirely and simplified the architecture.

**Q: Why did you choose fastembed over sentence-transformers?**

A: Disk space. sentence-transformers pulls in torch, which is roughly 2 GB. fastembed is about 50 MB and uses ONNX runtime. On the machine where this was built, that difference mattered. The embedding quality difference between BAAI/bge-small-en-v1.5 (fastembed) and all-MiniLM-L6-v2 (sentence-transformers) is small enough that retrieval accuracy did not degrade noticeably on the five policy documents.

**Q: What does INFEASIBLE mean in the context of this LP and how do you handle it?**

A: INFEASIBLE means the constraints cannot be simultaneously satisfied. No assignment of charge, discharge, and SoC variables meets every bound and equation. The most common cause is a safety buffer higher than what the battery can maintain given its starting SoC. The solver returns a `SolverResult` with status INFEASIBLE and a reason string explaining which constraint caused the conflict. The agent does not retry. It returns the result with a plain-language summary explaining why no schedule was possible. The audit trail captures the solver status and reason.

---

## 7. Technical Glossary

**VPP (Virtual Power Plant).** A network of distributed energy resources (batteries, solar arrays, generators) that behaves as a single controllable entity to the grid. This project orchestrates a single-site VPP: one battery connected to solar and the public grid.

**SoC (State of Charge).** How full the battery is, expressed as a fraction of capacity. 0.0 is empty, 1.0 is full. The LP tracks SoC across 24 hours and enforces it never drops below the regulatory safety buffer.

**LMP (Locational Marginal Price).** The wholesale electricity price at a specific grid node. Varies by hour. Low LMP means cheap charging. High LMP means profitable discharging. The system reads historical ERCOT LMPs from Postgres.

**GLOP (Google Linear Optimization Package).** The default LP solver in OR-Tools. Handles continuous linear programs with up to thousands of variables. Sufficient for this 72-variable problem.

**BESS (Battery Energy Storage System).** The physical hardware: battery cells, inverter, grid connection. The solver's decision variables represent the BESS' charge and discharge actions.

**LMP arbitrage.** The basic profit mechanism: charge when the price is low, discharge when the price is high. The objective function for MAXIMIZE_PROFIT is a mathematical expression of this strategy.

**RAG (Retrieval-Augmented Generation).** The pattern of retrieving relevant documents from a vector database and using their content to inform the LLM's output. In this system, RAG retrieves policy documents and the parser extracts constraint values from the retrieved text.

**constraint_float.** The numeric safety buffer fraction extracted from a policy document, normalized to [0.0, 1.0]. 0.30 means the solver must keep at least 30% of capacity reserved at all times.

**parse_confidence.** Either "parsed" (the constraint parser found a value in the chunk) or "fallback" (parser returned None, defaulted to 0.10). Recorded in every PolicyResult to make weak matches traceable.

**AuditTrail.** Every OptimizationResponse includes an audit field with the retrieved policy doc ID, raw chunk text, parsed constraint, solver status, solver timing, and the full tool call sequence with inputs and outputs. Not optional.

**INFEASIBLE.** A solver status indicating no assignment of decision variables can satisfy all constraints simultaneously. Not an error. A valid result with a human-readable reason.

**Curtailment.** Intentionally reducing solar generation below what the panels could produce. The LP models curtailment through the `s[t]` variable, which allows the solver to discard surplus solar instead of forcing it into a full battery.
