# 2026-10-27 — Top-k with k fixed at 5

From the 10-26 meeting. Fitting k by agg-LL drives it to k = 27, which predicts exactly what Random
predicts (uniform over all 27 combos → uniform marginals), so on 10-22 Top-k was dropped from
Fig. 5. Instead, k is now **fixed at 5** rather than fit: a round number that keeps Table 2
independent of Fig. 3 (the "top 7 covers ~50% of team-stages" reason for the old k = 7 was
backfilled, so it isn't used).

Everything else is unchanged from [2026-10-20-lapse-refit-all-models](../2026-10-20-lapse-refit-all-models/)
(λ = 0.01 lapse, no floor, agg-LL fits) and [2026-10-22-paper-refit-updates](../2026-10-22-paper-refit-updates/)
(Bayesian-Threshold-BR stays out of Fig. 5 because it collapses onto Bayesian-BR).

Run (about a minute) from `analysis/`:
`uv run python experiments/2026-10-27-top-k-fixed-5/top5.py`

| File | Contents |
|---|---|
| `tables.tex` | Table 2 rows with Top-$k$ ($k=5$); bold recomputed |
| `stuff to incorporate/R4_individual_fitting.{pdf,png}` | Fig. 5 (12 models: the 10-22 eleven + Top-5) |
| `summary.md` | Top-5 row, dominant-model counts, group totals, leading posterior > 0.5 |
| `results.json` | Top-5 metrics, per-participant Top-5 log-likelihoods, counts |

Checks the script asserts: this scoring reproduces the refit's Top-27 metrics and per-participant
log-likelihoods, and the refit's k = 5 Top-k profile entry, to 1e-9.

## Results

- **Table 2:** Top-5 scores combo-r 0.370, marg-r **0.516**, agg-LL −4.317, mean-LL −4.863.
  marg-r is now best in the column (was Optimal, 0.484), so the bold moves. Bayesian-Walk-BR is
  still best on combo-r, agg-LL and mean-LL. Top-5 puts zero mass outside its five combos, so its
  log-likelihoods lean on the lapse and sit well below Random.
- **Fig. 5 (n = 102):** Bayesian 52 (51%; Walk-BR 25, Bayesian-BR 9, Walk-PS 7, PS 6,
  Threshold-PS 5), value-based baselines 23 (23%; Top-5 13, Optimal 5, Random-to-Optimal 5),
  noise / contrarian 27 (26%; Random 15, Contradict Others 7, Random Walk 5). Leading posterior
  > 0.5 for 50 of 102.
- Compared with 10-22 (11 models): Top-5 takes 13 participants, 10 of them from Bayesian models.
  The value-sensitive total (Bayesian + value-based) goes from 73 to 75.
