# Domain Docs

## Layout

**Single-context.** One `CONTEXT.md` at the repository root covers the entire project.

```
CONTEXT.md         # domain vocabulary, solver model, data sources, tech boundaries
docs/adr/          # Architecture Decision Records
```

## How to Read CONTEXT.md

`CONTEXT.md` defines:
- Core domain terms (SoC, LMP, VPP, INFEASIBLE, safety buffer)
- The LP model variables and constraints
- System modes and their meaning
- What decisions belong to the LLM vs. the solver
- Data model (Postgres schema, Qdrant collection shape, model artifact path)

Read it before making architectural decisions in any service.

## Architecture Decision Records (ADRs)

When a significant architectural choice is made — especially anything that changes the
fixed pipeline topology — document it in `docs/adr/`.

File naming: `docs/adr/NNN-short-title.md` (e.g., `docs/adr/001-use-glop-over-cbc.md`)

Minimum ADR structure:
```
# NNN: Decision Title

## Status
Proposed | Accepted | Superseded

## Context
What problem required a decision.

## Decision
What was decided.

## Consequences
What becomes easier. What becomes harder.
```

An ADR is required before: changing the solver backend, adding a new inter-service dependency,
modifying `shared/contracts.py` in a breaking way, or restructuring the async pipeline.
