# population — bayes-models

Fit on 102 population by per-player agg-LL (λ = 0.01 lapse); higher is better. Bold = best in column. *combo-r is team-level: all 204 team-rounds, scored with the population fit, each player on their own type's belief.

| Model | Fitted params | combo-r* | marg-r | agg-LL | mean-LL |
|---|---|--:|--:|--:|--:|
| Bayesian-BR | τ_v = 11.33 | 0.412 | 0.206 | -1.016 | -1.044 |
| Bayesian-Walk-BR | τ_v = 10.14, ε_s = 0.52 | **0.488** | 0.371 | **-0.985** | **-0.956** |
| Bayesian-Threshold-BR | τ_v = 11.33, δ = -70.000 | 0.412 | 0.206 | -1.016 | -1.044 |
| Bayesian-PS | — | 0.394 | 0.437 | -1.038 | -1.026 |
| Bayesian-Walk-PS | ε_s = 0.60 | 0.439 | **0.441** | -0.999 | -0.994 |
| Bayesian-Threshold-PS | δ = -0.345 | 0.457 | 0.404 | -1.016 | -1.031 |
