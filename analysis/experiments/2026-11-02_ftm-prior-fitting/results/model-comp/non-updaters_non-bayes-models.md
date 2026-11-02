# non-updaters — non-bayes-models

Fit on 52 non-updaters by per-player agg-LL (λ = 0.01 lapse); higher is better. Bold = best in column. *combo-r is team-level: all 204 team-rounds, scored with the non-updaters fit, each player on their own type's belief.

| Model | Fitted params | combo-r* | marg-r | agg-LL | mean-LL |
|---|---|--:|--:|--:|--:|
| Optimal | — | 0.239 | 0.333 | -2.356 | -2.380 |
| Top-5 | k = 5 (fixed) | 0.370 | **0.349** | -1.355 | -1.390 |
| Random-to-Optimal | — | 0.180 | 0.195 | -1.371 | -1.447 |
| Copy Others | — | -0.004 | -0.017 | -1.391 | -2.521 |
| Contradict Others | — | 0.293 | 0.081 | -1.140 | -1.382 |
| Random Walk | ε = 0.298 | **0.400** | 0.206 | **-1.053** | **-1.050** |
| Random | — | 0.243 | -0.000 | -1.099 | -1.099 |
