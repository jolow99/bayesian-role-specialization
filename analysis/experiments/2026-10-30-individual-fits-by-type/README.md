# 2026-10-30 — Fig. 5 (individual fits) by inference type

Per-participant model posteriors for the population, Bayesians and non-Bayesians, stacked in one
PNAS column. Split from the [10-27 two-type classifier](../2026-10-27-inference-soft-classifier/)
(`p_prior_type` < 0.5 = Bayesian); Bayesian models run on its mixed beliefs (τ_prior = 4.05,
ε = 0.246, β = 0.726). Teams mix types, so each group's models are fit per player (per-player
agg-LL, λ = 0.01) on that group's own choices; Top-k fixed at k = 5. A model whose
per-participant log-likelihoods match another's to < 1e-3 is dropped, as in the paper.

Scoring, fitting and panel code: `fit_helpers.py`.

Run from `analysis/`:
`uv run python experiments/2026-10-30-individual-fits-by-type/individual_fits.py`

| File | Contents |
|---|---|
| `figures/individual_fits_by_type.{png,pdf}` | the figure |
| `individual_fits_summary.json` | best-model counts, fitted parameters, dropped models per group |

## Results

| | Population | Bayesians | Non-Bayesians |
|---|--:|--:|--:|
| Bayesian model best | 57/102 | 32/55 | 21/47 |
| Walk-BR fit | τ_v = 10.9, ε_s = 0.53 | τ_v = 9.0, ε_s = 0.55 | τ_v = 15.9, ε_s = 0.42 |

Check (`check_beliefs.py`): the beliefs match the classifier's own (1 − β)·P_N + β·P_0 at all 721
stages of the 204 team-rounds (max difference 6e-17).
