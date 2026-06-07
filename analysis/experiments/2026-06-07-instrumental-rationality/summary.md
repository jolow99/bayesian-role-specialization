# Instrumental rationality — value-rank of played combos + model comparison

Scope: 204 clean human team-rounds (5 exports), 721 team-stage observations. Rank = position of the played joint combo among all 27 by the precomputed value matrices (eap-weighted, section2 conventions; 1 = best, chance = 14). All CIs are percentile cluster bootstraps over team-rounds (10,000 resamples).

## Reproduction check (vs 05-28 `topk_summary.md`)

Mean rank **9.71** (expected 9.71), Top-1 **11.7%** (11.7%), Top-5 **42.4%** (42.4%), Bottom-5 **8.3%** (8.3%) — all match.

## Figure 1 — mean value-rank by game number (`rank_by_game`)

| Game | n stages | n team-rounds | Mean rank | 95% CI | Top-1 | Top-5 |
|---|--:|--:|--:|---|--:|--:|
| 1 | 94 | 27 | 9.61 | [7.71, 11.61] | 6.4% | 41.5% |
| 2 | 99 | 26 | 9.61 | [7.68, 11.69] | 9.1% | 43.4% |
| 3 | 98 | 28 | 10.37 | [8.44, 12.34] | 7.1% | 38.8% |
| 4 | 76 | 22 | 10.75 | [8.17, 13.38] | 14.5% | 36.8% |
| 5 | 81 | 24 | 7.83 | [5.98, 9.73] | 16.0% | 49.4% |
| 6 | 105 | 29 | 9.94 | [7.86, 12.09] | 15.2% | 41.0% |
| 7 | 75 | 23 | 8.99 | [6.85, 11.23] | 14.7% | 49.3% |
| 8 | 93 | 25 | 10.35 | [8.07, 12.73] | 11.8% | 40.9% |

Trend (OLS slope, rank per game; negative = improving): **-0.011** [-0.342, +0.317].

## Figure 2 — Top-K / Bottom-K curves (`topk_curves`)

| K | Top-K | 95% CI | Bottom-K | 95% CI | Uniform |
|--:|--:|---|--:|---|--:|
| 1 | 11.7% | [9.1%, 14.4%] | 1.8% | [0.8%, 2.9%] | 3.7% |
| 3 | 26.4% | [22.5%, 30.5%] | 5.0% | [3.3%, 6.9%] | 11.1% |
| 5 | 42.4% | [37.9%, 47.2%] | 8.3% | [6.1%, 10.7%] | 18.5% |
| 7 | 51.7% | [47.0%, 56.5%] | 13.6% | [10.5%, 16.8%] | 25.9% |
| 10 | 62.1% | [57.5%, 66.8%] | 20.9% | [17.2%, 24.9%] | 37.0% |
| 14 | 71.7% | [67.4%, 75.9%] | 30.2% | [25.9%, 34.6%] | 51.9% |

## Figure 3 — individual model fitting (`individual_fitting`)

Posteriors over 13 models for 102 participants, renormalized from the per-participant log-likelihoods in `2026-05-28-paper-figures/results.json` (agg_ll-objective fits; lapse = 0.05; Stage-1: τ_prior = 4.6385, ε = 0.0624, memory = `drift_prior_0.500`). **Mixture-PS is excluded**: its fitted mixture weight under the agg_ll objective is w = 1.0, i.e. it collapses exactly onto Bayesian Walk-PS (identical posteriors for all 102 participants), so keeping it would double-count Walk-PS in the model-posterior normalizer. Dropping it changes no dominant-model assignment. Dominant-model counts (recomputed, match the stored `dominant_counts`):

| Model | n participants |
|-------|--:|
| Bayesian Walk | 15 |
| Bayesian Walk-PS | 6 |
| Bayesian-Belief | 7 |
| Bayesian-Value | 7 |
| Bayesian Thresh-PS | 8 |
| Random Walk | 5 |
| Top-7 | 20 |
| Random-to-Optimal | 5 |
| Optimal | 9 |
| Contradict Others | 8 |
| Random | 12 |

## Table — `aggregate_table.tex`

Regenerated from `2026-05-28-paper-figures/results.json`; diff-identical to `2026-06-05-paper-figures-v2/aggregate_table.tex` (modulo the auto-generated header comment).
