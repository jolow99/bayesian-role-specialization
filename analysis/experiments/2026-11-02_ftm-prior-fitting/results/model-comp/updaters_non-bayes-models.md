# updaters — non-bayes-models

Fit on 50 updaters by per-player agg-LL (λ = 0.01 lapse); higher is better. Bold = best in column. *combo-r is team-level: all 204 team-rounds, scored with the updaters fit, each player on their own type's belief.

| Model | Fitted params | combo-r* | marg-r | agg-LL | mean-LL |
|---|---|--:|--:|--:|--:|
| Optimal | — | 0.239 | 0.476 | -2.053 | -2.101 |
| Top-5 | k = 5 (fixed) | 0.370 | 0.387 | -1.132 | -1.189 |
| Random-to-Optimal | — | 0.180 | 0.450 | -1.210 | -1.301 |
| Copy Others | — | -0.004 | 0.102 | -1.447 | -2.901 |
| Contradict Others | — | 0.293 | -0.126 | -1.089 | -1.228 |
| Random Walk | ε = 0.179 | **0.374** | **0.568** | **-0.968** | **-0.963** |
| Random | — | 0.243 | -0.000 | -1.099 | -1.099 |
