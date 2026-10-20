# 2026-10-20 — every model refit under a λ = 0.01 lapse, no floor

The two 10-20 tolerant-threshold folders
([PS](../2026-10-20-tolerant-threshold-ps/), [BR](../2026-10-20-tolerant-threshold-br/)) fit only
the threshold models under the lapse. Every other model kept its 05-25 parameters, which were fit
under the old floored objective. Here every model with a free parameter is refit with the same
scoring, so the comparison uses one rule throughout:

- **agg-LL / mean-LL:** (1 − λ)·p + λ/27 on the 27-combo joint, no floor
- **individual fit:** (1 − λ)·p + λ/3 on each player's marginal, no floor
- **combo-r / marg-r:** raw predictions (correlations, not likelihoods)

| Model | Free parameters | Search |
|---|---|---|
| Bayesian-BR | τ_v | 20-point log grid, then bounded 1-D on [0.01, 50] |
| Bayesian-Walk-BR | τ_v, ε_s | 15 × 21 grid, then L-BFGS-B |
| Bayesian-Walk-PS | ε_s | 21-point grid, then bounded 1-D on [0, 1] |
| Bayesian-Threshold-PS | δ_PS ∈ [−1, 1] | as in the tolerant-threshold-ps folder |
| Bayesian-Threshold-BR | τ_v, δ_BR ∈ [−70, 70] | as in the tolerant-threshold-br folder |
| Random Walk | ε (switch probability) | 21-point grid, then bounded 1-D on [0, 1] |
| Top-k | k | every k from 1 to 27 |
| PS, Optimal, Random-to-Optimal, Copy Others, Contradict Others, Random | none | — |

Each model is fit under all three objectives (combo-r, agg-LL, mean-LL). The main table and Fig. R4
use the agg-LL fits. ε_s is the stick probability (the code's `epsilon_switch` is 1 − ε_s).

Run (a few minutes; needs both tolerant-threshold folders):
`uv run python experiments/2026-10-20-lapse-refit-all-models/refit_all.py` from `analysis/`.

| File | Contents |
|---|---|
| `summary.md` | fitted params × objective for every model, 13-model table (new vs BR folder vs paper), dominant-model counts, Top-k profile |
| `tables_snippets.tex` | main-text aggregate table rows (all 13) and SI fitted-params rows |
| `stuff to incorporate/R4_individual_fitting.{pdf,png}` | regenerated Fig. R4 |
| `results.json` | all numbers incl. per-participant log-likelihoods |

Checks the script asserts: the paper's models reproduce the paper's floored metrics to 1e-9; the
fast expected-value versions of BR / Walk-BR give the same predictions as the 05-25 model code to
1e-12; and every refit scores at least as well as the paper's parameters on its own objective.

## Caveats

- **Top-k collapses to Random.** agg-LL rises with k, so the agg-LL fit is k = 27, which predicts
  exactly what Random predicts (combo-r alone picks k = 4). The paper's k = 7 was hand-set, never
  fit. `results.json` still keys this model as `"Top-7"`.
- **Fig. R4 ties.** Top-27 and Random have identical log-likelihoods for every participant; the
  dominant-model argmax picks Top-k because it comes first in the model order, so the 16 "Top-27"
  participants are really Random and "Random: 0" is an artifact. Threshold-BR is also equal to
  Bayesian-BR (its agg-LL fit sits at the δ_BR = −70 bound), so BR / Threshold-BR splits are
  arbitrary too.
