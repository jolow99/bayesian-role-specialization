# 2026-05-12 Current Export Behavioral Analysis

Descriptive behavioral pass over the current Empirica exports in
`analysis/data/exports/`. This re-runs the 2026-05-11 analysis now that
all five treatment-condition exports are available:

- `bayesian-role-specialization-2026-04-23-09-12-55`
- `bayesian-role-specialization-2026-04-27-10-05-17`
- `bayesian-role-specialization-2026-04-27-11-56-13`
- `bayesian-role-specialization-2026-05-11-15-10-24`
- `bayesian-role-specialization-2026-05-11-16-42-04`

This analysis intentionally does not fit model parameters. It summarizes:

- export and dropout scope
- human-round outcomes, stat-optimal adherence, role switching, and inference accuracy
- bot-round outcomes, deviation to deviate-optimal role, and inference accuracy
- per-export and pooled "All" summaries

Dropout handling: per-export scope counts include dropout games, but every
behavioral metric (human outcomes, stat-optimal rate, switch rate, inference
accuracy, bot deviation, bot inference) is computed over clean records
only. Clean = the player-round is not a dropout and (for human teams) no
player in the team is a dropout or has any auto-submitted stage. Bot-round
metrics drop any player-round with `is_dropout=True`.

Run from the repo root:

```bash
analysis/.venv/bin/python analysis/experiments/2026-05-12-current-export-behavioral-analysis/analyze.py
```

Outputs are written into this folder:

- `summary.md`
- `tables/*.csv`
- `figures/*.png`
