# 2026-10-22 — paper figures and tables after the 10-20 lapse refit

Brings the paper in line with [2026-10-20-lapse-refit-all-models](../2026-10-20-lapse-refit-all-models/)
(every model refit under a λ = 0.01 lapse, no floor). Only fixes what the paper already reports;
nothing new is added.

- **Fig. 4 + SI Figs. S1–S4** (team case studies): re-rendered through the
  [10-15 final6pt](../2026-10-15-fig4-legend-options/final6pt.py) code with the refit
  Bayesian-Walk-BR agg-LL parameters (τ_v = 11.67, ε_s = 0.46; was τ_v = 12.09, ε_s = 0.46).
- **Fig. 5** (individual fits): 11 models. Following Xuan, models that collapse onto another model
  are dropped from the individual comparison: Top-k (agg-LL fit k = 27, same log-likelihood as
  Random for every participant) and Bayesian-Threshold-BR (agg-LL fit at the δ_BR = −70 bound, same
  as Bayesian-BR to < 1e-3 per participant). Both stay in Table 2.
- **Table 2 and SI Table S4** rows, from the refit's `results.json`.

Run (a few minutes) from `analysis/`:
`uv run python experiments/2026-10-22-paper-refit-updates/make_paper_updates.py`

| File | Contents |
|---|---|
| `stuff to incorporate/R3_team_case*.{pdf,png}` | Fig. 4 and SI Figs. S1–S4 |
| `stuff to incorporate/R4_individual_fitting.{pdf,png}` | Fig. 5 (11 models) |
| `case_probabilities.md` | probability on each chosen role, old vs new parameters (for the captions) |
| `summary.md` | dominant-model counts, group totals, leading posterior > 0.5 |
| `tables.tex` | Table 2 rows and SI Table S4 rows |

Checks the script asserts: rendered with the old parameters, each case figure matches the paper's
PNG pixel for pixel (so the new renders differ only by the parameters); the two dropped models
score every participant like Random / Bayesian-BR.

## Results

- Individual fits (n = 102): Bayesian 62 (61%; Walk-BR 29, Bayesian-BR 11, Walk-PS 10, PS 6,
  Threshold-PS 6), value-based baselines 11 (Optimal 6, Random-to-Optimal 5), noise / contrarian 29
  (Random 16, Contradict Others 7, Random Walk 6). Leading posterior > 0.5 for 49 of 102.
- Case-study probabilities move by at most 0.01. Changes that reach the text: Fig. 4 P1's stage-2
  move 0.36 → 0.37 and held roles 0.61–0.93 → 0.61–0.94; Fig. S1 P3 0.49–0.73 → 0.49–0.74, P1's
  switch 0.26 → 0.27, P2's switches 0.13 → 0.12, P2 settled 0.73 → 0.74; Fig. S3 held roles
  0.62–0.90 → 0.62–0.91; Fig. S4 other choices 0.51–0.97 → 0.52–0.97. Every "most probable" claim
  and the Fig. 4 stage means (0.28, 0.39, 0.73, 0.75; mean 0.54) are unchanged.

## Caveat

The Bayesian share rises from 43% to 61% mainly because Top-7 left the comparison, not because of
the refit: with the refit parameters plus the hand-set Top-7, the split is Bayesian 45, value-based
31, noise 26. The value-sensitive total (Bayesian + value-based) is stable: 77 (paper), 76, 73.
