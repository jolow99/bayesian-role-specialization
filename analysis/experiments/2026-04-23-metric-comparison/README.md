# 2026-04-23 Metric Comparison

Re-run of the metric-comparison pipeline on the 04-23 Empirica export.

## Data scope

- Single export: `bayesian-role-specialization-2026-04-23-09-12-55`
- 10 games total; **3 had dropouts → excluded entirely**
- 7 clean games contribute **42 complete human team-rounds**
- **Stage 2 uses 21 of those 42** — only the team-rounds whose `(role_combo,
  stat_profile, maxTeamHealth, maxEnemyHealth, bossDamage)` matches a
  precomputed value matrix under `analysis/data/human_envs_value_matrices/`.
  The other 21 (env IDs `411_222_222_FFT`, `141_222_222_MFF`,
  `114_222_222_FFF`) have no precomputed values and are skipped so that all
  seven models are fit and evaluated on the same records. Stage 1 still
  uses all 42 teams (it only needs posteriors, not value matrices).
- Bot rounds are not used.

## Self-contained

This experiment imports only from `shared/` — nothing from any other
`experiments/` folder. Stage 1 params are re-tuned from scratch. `pipeline.py`
replicates the bits of the 04-12 pipeline it needs (memory strategies,
trajectory precompute, metrics, checkpoint helpers, record loader with value
matrix lookup).

## Design

### Stage 1: human-only inference tune

3-phase grid search (coarse → refined → L-BFGS-B polish) over
(tau_prior, epsilon, memory_strategy), using the 542 human inference queries
from the 42 clean teams.

### Stage 2: triple-objective sweep

For each of 3 objectives × 7 models = 21 cells:
- Fit the model's params on the 21 matched team-rounds using that objective
- Evaluate ALL metrics at the fitted point

**Objectives:** combo_r, agg_ll, mean_ll

**Models:** belief, value, walk, thresh, walk_ps, thresh_ps, mixture_ps

`mixture_ps` freezes walk_ps / thresh_ps params fit under the **same**
objective.

## How to run

```bash
cd analysis/experiments/2026-04-23-metric-comparison

# Stage 1 (~a few minutes)
python stage1_inference/tune.py

# Stage 2 (~a minute)
python run_comparison.py
```

## Files

| File | Role |
|------|------|
| `pipeline.py` | Memory strategies, trajectory precompute, metrics, record loading, checkpoint helpers |
| `models.py` | 7 model factories (belief, value, walk, thresh, walk_ps, thresh_ps, mixture_ps) |
| `stage1_inference/tune.py` | Stage 1 inference-param tune |
| `run_comparison.py` | Stage 2 triple-objective × 7-model comparison |

## Outputs

- `stage1_inference/best_inference_params.json` — human-only S1 params
- `results.json` — structured results (all 21 cells)
- `comparison_table.md` — human-readable metric tables
- `figures/ranking_heatmap.png` — model ranking stability
- `figures/param_sensitivity.png` — fitted param variation across objectives
- `figures/metric_correlation.png` — scatter matrix of eval metrics

## Results at a glance

Stage 1: `drift_prior_0.400`, tau_prior = 5.057, epsilon = 0.001,
inference LL = −0.8225 (542 queries).

Model ranking by combo_r under each objective:

| Objective | Ranking |
|-----------|---------|
| combo_r   | thresh (0.414) > walk (0.337) > value (0.311) > thresh_ps / mixture_ps (0.298) > walk_ps (0.246) > belief (0.233) |
| agg_ll    | thresh (0.410) > walk (0.302) > value (0.287) > thresh_ps (0.263) > mixture_ps (0.244) > walk_ps (0.238) > belief (0.233) |
| mean_ll   | thresh (0.410) > walk (0.298) > thresh_ps (0.255) > walk_ps / mixture_ps (0.236) > belief (0.233) > value (0.224) |

Stability is high: `bayesian_thresh` wins under every objective,
`bayesian_walk` is a clear second, and `bayesian_belief` is the floor
(no params → constant across objectives). With only 21 teams the numbers
are noisy — re-evaluate once more envs are fit with value matrices.
