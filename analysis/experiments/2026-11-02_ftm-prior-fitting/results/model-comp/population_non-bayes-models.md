# population — non-bayes-models

Fit on 102 population by per-player agg-LL (λ = 0.01 lapse); higher is better. Bold = best in column. *combo-r is team-level: all 204 team-rounds, scored with the population fit, each player on their own type's belief.

| Model | Fitted params | combo-r* | marg-r | agg-LL | mean-LL |
|---|---|--:|--:|--:|--:|
| Optimal | — | 0.239 | 0.484 | -2.207 | -2.244 |
| Top-5 | k = 5 (fixed) | 0.370 | **0.516** | -1.246 | -1.291 |
| Random-to-Optimal | — | 0.180 | 0.354 | -1.281 | -1.376 |
| Copy Others | — | -0.004 | 0.265 | -1.316 | -2.704 |
| Contradict Others | — | 0.293 | -0.215 | -1.097 | -1.307 |
| Random Walk | ε = 0.203 | **0.381** | 0.265 | **-1.021** | **-1.031** |
| Random | — | 0.243 | -0.000 | -1.099 | -1.099 |
