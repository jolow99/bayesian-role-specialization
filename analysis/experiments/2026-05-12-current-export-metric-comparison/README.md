# 2026-05-12 Current Export Metric Comparison

Combined model-comparison pipeline over all five current treatment-condition
Empirica exports:

- `bayesian-role-specialization-2026-04-23-09-12-55`
- `bayesian-role-specialization-2026-04-27-10-05-17`
- `bayesian-role-specialization-2026-04-27-11-56-13`
- `bayesian-role-specialization-2026-05-11-15-10-24`
- `bayesian-role-specialization-2026-05-11-16-42-04`

This is the 2026-05-11 experiment re-run now that all five treatment
conditions are available. The loader path now reads value matrices from
`analysis/data/human_envs_value_matrices/` (the renamed location of the
big-pilot matrices).

## Data Scope

- Human rounds only.
- Games with any dropout player-round are excluded entirely
  (`pipeline.discover_dropout_games` / `filter_clean_prs`).
- Remaining human team-rounds must have all three players present.
- Stage 1 uses all clean complete human teams and does not require value
  matrices.
- Stage 2 requires a matching value matrix under
  `analysis/data/human_envs_value_matrices/<stat_profile>__<role_combo>/`.
  Missing matrices raise `MissingValueMatrixError` rather than silently
  dropping records (which would bias the comparison).

## How To Run

From this directory:

```bash
# Stage 1: tune inference params
python stage1_inference/tune.py

# Stage 2: fit/evaluate 3 objectives x 7 models
python run_comparison.py
```

Or from the repo root:

```bash
analysis/.venv/bin/python analysis/experiments/2026-05-12-current-export-metric-comparison/stage1_inference/tune.py
analysis/.venv/bin/python analysis/experiments/2026-05-12-current-export-metric-comparison/run_comparison.py
```

## Outputs

- `stage1_inference/best_inference_params.json`
- `results.json`
- `comparison_table.md`
- `figures/ranking_heatmap.png`
- `figures/param_sensitivity.png`
- `figures/metric_correlation.png`
