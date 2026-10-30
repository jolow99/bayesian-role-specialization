# 2026-10-30 — Fig. 2 (calibration) by inference type

The paper's calibration figure for the population, Bayesians and non-Bayesians. Both the split and
the x-axis come from the [10-27 two-type classifier](../2026-10-27-inference-soft-classifier/):
Bayesian = `p_prior_type` < 0.5; x = the classifier's posterior P_N (τ_prior = 4.05, ε = 0.246,
β = 0.726), the one it scores role reports against. Binning, cluster bootstrap and layout are the
paper's (2026-06-16 `calibration_revision.py`).

Check the script asserts: these posteriors reproduce the classifier's per-participant log B_i.

Run from `analysis/`:
`uv run python experiments/2026-10-30-calibration-by-type/calibration_by_type.py`

| File | Contents |
|---|---|
| `figures/R1_calibration_{all,bayesian,nonbayesian}.{png,pdf}` | one figure per group |
| `calibration_summary.md` | binned / raw-pair r, accuracy, per-bin detail |

## Results

| Group | n | binned r | raw-pair r | human acc. | sampler acc. |
|---|--:|--:|--:|--:|--:|
| All | 102 | 0.97 | 0.46 | 62.9% | 77.8% |
| Bayesian | 55 | 0.99 | 0.74 | 80.1% | 76.9% |
| Non-Bayesian | 47 | 0.79 | 0.15 | 43.4% | 78.9% |

Bayesians lie on the identity line (mean |bin − identity| 0.037). The population no longer matches
the posterior sampler (63% vs 78%; the paper's 05-25 posterior gave 63% vs 61%). Bayesians were
selected for fitting this posterior, so their curve is partly so by construction.
