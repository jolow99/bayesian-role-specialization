# updaters — bayes-models

Fit on 50 updaters by per-player agg-LL (λ = 0.01 lapse); higher is better. Bold = best in column. *combo-r is team-level: all 204 team-rounds, scored with the updaters fit, each player on their own type's belief.

| Model | Fitted params | combo-r* | marg-r | agg-LL | mean-LL |
|---|---|--:|--:|--:|--:|
| Bayesian-BR | τ_v = 9.48 | 0.408 | 0.165 | -0.983 | -1.013 |
| Bayesian-Walk-BR | τ_v = 8.54, ε_s = 0.58 | **0.482** | **0.591** | **-0.920** | **-0.889** |
| Bayesian-Threshold-BR | τ_v = 9.48, δ = -60.750 | 0.408 | 0.165 | -0.983 | -1.013 |
| Bayesian-PS | — | 0.394 | 0.377 | -0.985 | -0.945 |
| Bayesian-Walk-PS | ε_s = 0.59 | 0.442 | 0.564 | -0.943 | -0.920 |
| Bayesian-Threshold-PS | δ = -0.356 | 0.458 | 0.293 | -0.955 | -0.941 |
