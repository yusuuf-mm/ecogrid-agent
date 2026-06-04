# Issue Tracker

Issues for this repository live in **GitHub Issues**.

Use the `gh` CLI to interact with them:

```bash
# Create an issue
gh issue create --title "feat: add MINIMIZE_CARBON objective to solver" --label "feat/phase-1-or-solver"

# List open issues
gh issue list

# View an issue
gh issue view 12

# Close an issue
gh issue close 12
```

## Conventions

- Title prefix: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `infra:`
- Tag with the relevant branch label (e.g., `feat/phase-2-ml-forecasting`) so issues stay
  scoped to the right service.
- For cross-service contract changes (`shared/contracts.py`), tag with `contracts` and assign
  to the maintainer before proceeding — these affect all branches.
- Link issues to PRs by including `Closes #<number>` in the PR description.
