# 2026-04-12 Metric Comparison

Does the choice of fitting objective (combo_r vs agg_ll vs mean_ll) change
model ranking or fitted parameters?

## Background

The 04-12 aggregate-LL pipeline revealed that switching from `combo_r` to
`agg_ll` as the Stage 2 objective, combined with pooled (human+bot) fitting,
dropped human-only `combo_r` from 0.50 to 0.32. Diagnostic analysis showed
three independent causes:

1. **Pooled fitting** pushed ε_switch too low — biggest effect
2. **Adding 02-13 export** introduced harder envs — medium effect
3. **Objective choice** (combo_r vs agg_ll) — negligible (~0.006)

This experiment verifies finding (3) rigorously by fitting all 7 models under
three objectives on **human-only** data with a **human-only Stage 1**.

## Design

### Stage 1: Human-only re-tune

Same 3-phase grid search (coarse → refined → L-BFGS-B polish) as 04-12, but
restricted to the 1164 human inference queries. Bot records are excluded.

### Stage 2: Triple-objective sweep

For each of 3 objectives × 7 models = 21 cells:
- Fit model params on human-only records using that objective
- Evaluate ALL metrics at the fitted point

**Objectives:** combo_r, agg_ll, mean_ll

**Models:** belief, value, walk, thresh, walk_ps, thresh_ps, mixture_ps

mixture_ps freezes walk_ps/thresh_ps params fit under the **same** objective.

## How to run

```bash
cd analysis/experiments/2026-04-12-metric-comparison

# Stage 1: human-only inference param tuning (~10-15 min)
python stage1_inference/tune.py

# Stage 2: triple-objective comparison (~30-60 min)
python run_comparison.py
```

## Outputs

- `stage1_inference/best_inference_params.json` — human-only S1 params
- `results.json` — structured results (all 21 cells)
- `comparison_table.md` — human-readable metric tables
- `figures/ranking_heatmap.png` — model ranking stability
- `figures/param_sensitivity.png` — fitted param variation across objectives
- `figures/metric_correlation.png` — scatter matrix of eval metrics

## Verification

1. S1 should have exactly 1164 queries (no bot leakage)
2. belief produces identical metrics across all 3 objectives (no params to fit)
3. walk_ps ε_switch under combo_r > under mean_ll (combo_r rewards switching)
4. Model ranking should be mostly stable across objectives
