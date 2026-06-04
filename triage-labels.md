# Triage Labels

The five canonical triage roles use default label names:

| Label             | Meaning                                                        |
|-------------------|----------------------------------------------------------------|
| `needs-triage`    | Maintainer needs to evaluate this — hasn't been looked at yet |
| `needs-info`      | Waiting on the reporter for more detail                        |
| `ready-for-agent` | Fully specified, unambiguous — an agent can pick it up         |
| `ready-for-human` | Needs human judgment or implementation                         |
| `wontfix`         | Will not be actioned in this project scope                     |

## Additional project labels

| Label                         | Meaning                                              |
|-------------------------------|------------------------------------------------------|
| `feat/phase-1-or-solver`      | Scoped to services/solver/                           |
| `feat/phase-1-backend-api`    | Scoped to services/api/ + services/workers/          |
| `feat/phase-2-ml-forecasting` | Scoped to services/ml/                               |
| `feat/phase-2-vector-rag`     | Scoped to services/rag/                              |
| `feat/phase-3-agent-core`     | Scoped to services/agent/                            |
| `feat/phase-4-docker-infra`   | Scoped to infra/                                     |
| `contracts`                   | Touches shared/contracts.py — affects all branches   |
