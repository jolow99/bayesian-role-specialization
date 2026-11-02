# non-updaters — bayes-models

Fit on 52 non-updaters by per-player agg-LL (λ = 0.01 lapse); higher is better. Bold = best in column. *combo-r is team-level: all 204 team-rounds, scored with the non-updaters fit, each player on their own type's belief.

| Model | Fitted params | combo-r* | marg-r | agg-LL | mean-LL |
|---|---|--:|--:|--:|--:|
| Bayesian-BR | τ_v = 16.97 | 0.408 | 0.036 | -1.057 | -1.063 |
| Bayesian-Walk-BR | τ_v = 16.51, ε_s = 0.43 | **0.481** | 0.207 | **-1.034** | **-1.003** |
| Bayesian-Threshold-BR | τ_v = 17.98, δ = -31.250 | 0.411 | 0.049 | -1.057 | -1.064 |
| Bayesian-PS | — | 0.394 | 0.393 | -1.105 | -1.103 |
| Bayesian-Walk-PS | ε_s = 0.58 | 0.443 | **0.426** | -1.041 | -1.058 |
| Bayesian-Threshold-PS | δ = -1.000 | 0.394 | 0.393 | -1.105 | -1.103 |
