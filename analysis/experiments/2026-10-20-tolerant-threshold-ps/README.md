# 2026-10-20 — tolerant Threshold-PS, λ = 0.01 lapse, no floor

Follow-up to the 10-15 pure Threshold-PS refit. That model never put the current role in its
candidate set, so it forced a switch whenever any role cleared δ, gave 0 to staying and to every
other switch, and had −∞ log-likelihood at every δ (the paper's numbers came from 1e-12 / 1e-20
floors). This implements Xuan's fix from the Slack thread:

    allowed(δ) = {prev} ∪ {r ≠ prev : P(r) − P(prev) > δ},   δ ∈ [−1, 1]
    π(r)       = posterior renormalised over allowed(δ)

δ = −1 is exactly Bayesian-PS (asserted in the script). Likelihoods use a λ = 0.01 uniform lapse
with no floor in both fits: on the 27-combo joint for agg-LL / mean-LL, and on each player's
marginal for the individual fit. combo-r / marg-r use raw predictions. The other 12 models keep
their paper (agg-LL) parameters. They are re-scored under the lapse but not refit, and
Bayesian-Threshold-BR is still the old forced-switch version.

Run (≈30 s): `uv run python experiments/2026-10-20-tolerant-threshold-ps/tolerant_threshold_ps.py` from `analysis/`.

| File | Contents |
|---|---|
| `summary.md` | δ fits per objective, observed-switch gaps, 13-model table (new lapse vs paper floors), dominant-model counts, δ profile |
| `tables_snippets.tex` | main-text aggregate rows and SI fitted-params rows |
| `stuff to incorporate/R4_individual_fitting.{pdf,png}` | regenerated Fig. R4 |
| `results.json` | all numbers incl. per-participant log-likelihoods |

Checks the script asserts: the 12 untouched models reproduce the paper's floored metrics to 1e-9
(so the local data and value matrices match), and δ = −1 reproduces Bayesian-PS's predictions to 1e-12.
