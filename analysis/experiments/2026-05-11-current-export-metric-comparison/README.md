# 2026-05-11 Current Export Metric Comparison

Combined model-comparison pipeline for the three current Empirica exports:

- `bayesian-role-specialization-2026-04-23-09-12-55`
- `bayesian-role-specialization-2026-04-27-10-05-17`
- `bayesian-role-specialization-2026-04-27-11-56-13`

This is copied from the single-export `2026-04-23-metric-comparison`
experiment, but its loaders now tune/evaluate on the combined current-export
dataset. The original `2026-04-23` experiment is left unchanged.

## Data Scope

- Human rounds only.
- Games with any dropout are excluded entirely.
- Remaining human team-rounds must have all three players present.
- Stage 1 uses all clean complete human teams and does not require value
  matrices.
- Stage 2 requires a matching value matrix under
  `analysis/data/human-envs-big-pilot-matrices/<stat_profile>__<role_combo>/`.
  Missing matrices raise an error rather than silently dropping records.

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
analysis/.venv/bin/python analysis/experiments/2026-05-11-current-export-metric-comparison/stage1_inference/tune.py
analysis/.venv/bin/python analysis/experiments/2026-05-11-current-export-metric-comparison/run_comparison.py
```

## Outputs

- `stage1_inference/best_inference_params.json`
- `results.json`
- `comparison_table.md`
- `figures/ranking_heatmap.png`
- `figures/param_sensitivity.png`
- `figures/metric_correlation.png`
