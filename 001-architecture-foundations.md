# 001: Architecture Foundations

## Status
Accepted

## Context

EcoGrid-Agent needs to combine three technically distinct systems: an LLM reasoning layer,
an ML forecasting model, and a deterministic LP solver. Several architectural patterns could
work. This ADR records the decisions made at project start and why.

## Decisions

### 1. One-directional data flow: API → Queue → Agent → Tools

Rejected: having tools call each other (e.g., solver calling the ML model directly).
Accepted: the agent is the only orchestrator. Tools are pure functions.

Reason: bidirectional tool dependencies make tracing and debugging significantly harder.
If the solver fails, we need to know exactly what inputs it received. That's only clean if
inputs flow from one place: the agent.

### 2. Deterministic LP solver, not an RL agent or LLM math

Rejected: using an LLM to calculate the schedule. Rejected: training an RL policy.
Accepted: Google OR-Tools LP solver (GLOP).

Reason: the system makes decisions about physical infrastructure. LLMs hallucinate numbers.
RL policies need millions of iterations and don't generalize cleanly to new constraints. An LP
solver with explicit constraints guarantees mathematical optimality and can prove infeasibility
when constraints conflict. These properties matter for regulated infrastructure.

### 3. RAG for constraint injection, not for generation

Standard RAG generates text. This system uses RAG differently: retrieved policy text is parsed
into a single float (e.g., 0.30) and that float is injected as an LP constraint parameter.

Reason: the output of the RAG step drives math, not narrative. By narrowing the LLM's role at
the RAG step to "extract a number from text," we minimize the surface area for hallucination.

### 4. Async task queue (Celery + Redis), not sync API response

Rejected: having the FastAPI route call the solver synchronously and wait.
Accepted: enqueue to Celery, return task_id, poll for result.

Reason: LP solvers can take 100ms–5s depending on problem size. Holding HTTP connections for
that long causes timeout issues and makes load balancing harder. A task queue also gives us
retry, backpressure, and horizontal scaling for free.

### 5. Single `shared/contracts.py` for inter-service types

Rejected: letting services define their own input/output types independently.
Accepted: all types crossing service boundaries live in one file.

Reason: when six services are being built on separate branches simultaneously, type drift is
the most common source of integration bugs. One authoritative file makes the interfaces visible
and forces negotiation before they change.

## Consequences

What becomes easier:
- Debugging: every decision traces back to a specific tool call, log entry, or constraint source
- Auditing: the `AuditTrail` type in contracts.py captures the full reasoning chain by design
- Scaling: stateless tools and a task queue make horizontal scaling straightforward
- Regulatory compliance: policy constraints link to document IDs, not hardcoded values

What becomes harder:
- Adding a new constraint type requires updating `shared/contracts.py` and rebasing all branches
- The fixed tool-call order (policy → forecast → solver) means the agent can't parallelise
  these calls; acceptable at this scale, revisit if latency becomes a constraint
