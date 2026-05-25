# 2026-05-25 Full Pipeline

End-to-end two-stage Bayesian role-specialization fitting on **all five
exports currently in `analysis/data/exports/`**:

- `bayesian-role-specialization-2026-04-23-09-12-55`
- `bayesian-role-specialization-2026-04-27-10-05-17`
- `bayesian-role-specialization-2026-04-27-11-56-13`
- `bayesian-role-specialization-2026-05-11-15-10-24`
- `bayesian-role-specialization-2026-05-11-16-42-04`

## Why a new experiment

The four predecessors (04-09, 04-11, 04-12, 04-23) differ along three
axes — dataset scope, bot-round inclusion, Stage-2 objective — and none of
them fits the current need. Recap:

| Experiment | Exports          | Bot rounds | Stage-2 objective | Notes |
|------------|------------------|------------|-------------------|-------|
| 04-09      | 02-13, 03-06, 03-18 | yes (buggy bot layout) | `combo_r` | first 7-model two-stage |
| 04-11      | same             | yes (buggy) | Stage 1 only | richer memory-strategy sweep |
| 04-12      | same             | yes (FIXED layout) | `agg_ll` | pooled, disaggregated, CV |
| 04-23      | only 04-23       | **no**     | three objectives  | self-contained, strict clean teams |
| 05-25 (this) | **all five (04-23, 04-27 x2, 05-11 x2)** | **no** | `agg_ll` | one fit, one table, one set of figures |

This experiment is **human-only** (bot rounds excluded). Reasoning:

1. Bot rounds have two of three roles deterministic, so model "predictions"
   over the joint role combo are dominated by a known constant — the
   resulting `agg_ll` / `combo_r` numbers say more about how well the
   model recovers the bot's hard-coded role than about how humans choose.
2. The interesting research question (do humans deviate to the
   deviate-optimal role under social pressure?) is best measured by per-
   player role choice, which is a separate analysis from this one.
3. The bot-round layout in `gameSummary` is a documented three-bug trap
   (CLAUDE.md → "Bot Round Ground Truth"); 04-09 and 04-11 silently fit on
   corrupt posteriors and 04-12 still had to disaggregate to interpret.
   Skipping bot rounds removes the trap entirely.

## Scope

- Games with **any** dropout are excluded entirely (matches 04-23). A
  game with a single dropout round-2 player is removed even for rounds
  1, 3, 4 where everyone was present.
- Only human team-rounds with all three players are kept.
- Every team-round must have a precomputed value matrix in
  `data/human_envs_value_matrices/<stat_profile>__<role_combo>/` — all 15
  envs in the current 5-export dataset have matrices, so nothing is
  dropped on this account.

After filtering: **34 clean games, 204 human team-rounds, 15 envs.**

## Design

### Stage 1 — inference parameter tune
Grid → refined → L-BFGS-B polish over (tau_prior, epsilon, memory_strategy)
on per-query mean log-likelihood of the human's inferred role under the
posterior marginal. Memory strategies: `full`, `window_{1..4}`,
`drift_prior_δ`, `drift_uniform_δ`, `temper_γ` (same expanded space as
04-12 / 04-23).

### Stage 2 — model fitting
Each of the 7 models is fit on **pooled human team-rounds** against the
aggregate cross-entropy `agg_ll` (proper scoring rule from 04-12), then
evaluated at the fitted point on the full metric panel:

- `combo_r` (Pearson over canonical-combo × stage × env, used in summary notebooks)
- `agg_ll` (per-sample mean cross-entropy)
- `mean_ll` (per-env mean log-likelihood of the chosen combo)
- `mean_P(chosen)` (mean predicted probability of the chosen combo, per-env then averaged)
- `IoU` (intersection-over-union of predicted vs observed marginal, per-env then averaged)
- `TVD` (total variation distance, per-env then averaged)

The four right-hand metrics replicate the 03-30 "additional metrics"
panel so that this experiment's figures are comparable.

Mixture-PS freezes walk_ps and thresh_ps params fit under the same
objective (matches 04-12 and 04-23).

## How to run

```bash
cd analysis/experiments/2026-05-25-full-pipeline

# Stage 1 (~minutes)
python stage1_inference/tune.py

# Stage 2 (~minutes)
python run_comparison.py
```

Both phases are resumable (JSON checkpoints under `*/checkpoints/`).

## Outputs

| File                          | Contents |
|-------------------------------|----------|
| `stage1_inference/best_inference_params.json` | Best (tau_prior, epsilon, memory_strategy) |
| `results.json`                | Stage-2 fits + eval metrics for all 7 models |
| `comparison_table.md`         | Human-readable metric table |
| `figures/additional_metrics.png` | Bar chart of Mean P(chosen) / combo_r / IoU / TVD |
| `figures/scatter_pred_vs_obs.png` | Per-model pred-prob vs observed-freq scatter (per canonical combo × stage × env) |

## Files

| File | Role |
|------|------|
| `pipeline.py` | Data loading (5 exports, human-only, clean), memory strategies, trajectory precompute, metric blocks (incl. P(chosen)/TVD/IoU), checkpoint helpers |
| `models.py` | 7 model factories — belief, value, walk, thresh, walk_ps, thresh_ps, mixture_ps |
| `stage1_inference/tune.py` | Stage 1 grid + polish |
| `run_comparison.py` | Stage 2 fit + figures + table |
