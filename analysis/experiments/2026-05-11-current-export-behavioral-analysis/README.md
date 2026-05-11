# 2026-05-11 Current Export Behavioral Analysis

Descriptive behavioral pass over the current Empirica exports in
`analysis/data/exports/`.

This analysis intentionally does not fit model parameters. It summarizes:

- export and dropout scope
- human-round outcomes, stat-optimal adherence, role switching, and inference accuracy
- bot-round outcomes, deviation to deviate-optimal role, and inference accuracy
- per-export and pooled "All" summaries

Run from the repo root:

```bash
analysis/.venv/bin/python analysis/experiments/2026-05-11-current-export-behavioral-analysis/analyze.py
```

Outputs are written into this folder:

- `summary.md`
- `tables/*.csv`
- `figures/*.png`
