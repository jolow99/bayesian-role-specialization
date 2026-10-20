# 2026-10-20 — tolerant Threshold-BR, λ = 0.01 lapse, no floor

Companion to [2026-10-20-tolerant-threshold-ps](../2026-10-20-tolerant-threshold-ps/). The old
Threshold-BR never put the current role in its candidate set, so it forced a switch whenever any
role's expected value beat the current one by δ and gave 0 to staying and to every other switch.
Same fix as Threshold-PS (Xuan's option 2):

    allowed(δ) = {prev} ∪ {r ≠ prev : V(r) − V(prev) > δ}
    π(r)       ∝ exp(V(r) / τ_v)  over allowed(δ)

δ is in expected-team-value units. Values here run 0–90 and per-stage gaps run from −68 to +61, so
δ ∈ [−70, 70]. δ = −70 lets every role through, which is exactly Bayesian-BR (asserted in the script).

Scoring is imported from the PS folder: λ = 0.01 lapse with no floor for agg-LL, mean-LL and the
individual fit, and raw predictions for combo-r / marg-r. The final 13-model table and Fig. R4 use
**both** tolerant threshold models (Threshold-PS at its agg-LL fit from the PS folder). The other 11
models keep their paper (agg-LL) parameters; they are re-scored under the lapse but not refit.

Fit: a coarse grid (δ step 1 on [−70, 70] × 20 log-spaced τ_v on [0.5, 50]), then δ in steps of
0.25 within ±2 of each objective's best, with τ_v optimised (bounded, [0.01, 50]) at each δ. The
objective is piecewise constant in δ, so δ is never fit with L-BFGS-B.

Run (a few minutes; needs the PS folder's `results.json`):
`uv run python experiments/2026-10-20-tolerant-threshold-br/tolerant_threshold_br.py` from `analysis/`.

| File | Contents |
|---|---|
| `summary.md` | (τ_v, δ) fits per objective, observed-switch value gaps, 13-model table (new lapse vs paper floors), dominant-model counts, δ profile |
| `tables_snippets.tex` | main-text aggregate rows (both tolerant models) and SI fitted-params rows for Threshold-BR |
| `stuff to incorporate/R4_individual_fitting.{pdf,png}` | regenerated Fig. R4 with both tolerant models |
| `results.json` | all numbers incl. per-participant log-likelihoods |

Checks the script asserts: the 11 untouched models reproduce the paper's floored metrics to 1e-9;
δ = −70 reproduces Bayesian-BR's predictions to 1e-12; the fitted agg-LL is at least Bayesian-BR's;
the tolerant Threshold-PS row matches the PS folder.
