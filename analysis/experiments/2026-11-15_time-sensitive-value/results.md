# Time-sensitive value function: V(10 − 2·stage) instead of V(10)

Δ vs shipped V = agg-LL minus the same model's agg-LL in the 2026-11-12 table with the paper's time-insensitive V (positive = time-sensitive fits better). Inference params fixed (2026-11-02).

## Updaters, mixed posterior (n = 50, 1064 choices), refit

| Model | Params | agg-LL | Δ vs shipped V | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 7.13 | -0.9976 | -0.0144 | -1.0093 | 0.190 | 0.400 | -1068.7 |
| Bayesian-Walk-BR | τ_v = 7.7, ε_s = 0.362 | -0.9248 | -0.0052 | -0.8937 | 0.593 | 0.469 | -944.4 |
| Bayesian-Threshold-BR | τ_v = 7.13, δ = -45.8 | -0.9976 | -0.0144 | -1.0093 | 0.190 | 0.400 | -1068.7 |
| Bayesian-PS | – | -0.9847 | +0.0000 | -0.9453 | 0.377 | 0.394 | -1004.1 |
| Bayesian-Walk-PS | ε_s = 0.413 | -0.9428 | +0.0000 | -0.9200 | 0.564 | 0.442 | -973.8 |
| Bayesian-Threshold-PS | δ = -0.356 | -0.9548 | +0.0000 | -0.9410 | 0.293 | 0.458 | -993.1 |

## Non-updaters on updater beliefs, refit (n = 52, 1099 choices)

| Model | Params | agg-LL | Δ vs shipped V | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 11.2 | -1.0546 | -0.0062 | -1.0512 | 0.251 | 0.388 | -1157.4 |
| Bayesian-Walk-BR | τ_v = 11.1, ε_s = 0.54 | -1.0287 | -0.0004 | -1.0011 | 0.311 | 0.485 | -1103.9 |
| Bayesian-Threshold-BR | τ_v = 11.3, δ = -33.2 | -1.0546 | -0.0061 | -1.0508 | 0.251 | 0.388 | -1157.1 |
| Bayesian-PS | – | -1.0460 | +0.0000 | -1.0156 | 0.177 | 0.447 | -1115.8 |
| Bayesian-Walk-PS | ε_s = 0.671 | -1.0378 | +0.0000 | -1.0175 | 0.205 | 0.459 | -1120.0 |
| Bayesian-Threshold-PS | δ = -0.383 | -1.0414 | +0.0000 | -1.0472 | 0.161 | 0.464 | -1154.7 |

## Non-updaters on updater beliefs AND updater (time-sensitive) parameters, nothing refit

| Model | Params | agg-LL | Δ vs shipped V | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 7.13 | -1.0648 | -0.0021 | -1.0666 | 0.259 | 0.398 | -1175.3 |
| Bayesian-Walk-BR | τ_v = 7.7, ε_s = 0.362 | -1.0374 | +0.0026 | -1.0687 | 0.310 | 0.471 | -1181.2 |
| Bayesian-Threshold-BR | τ_v = 7.13, δ = -45.8 | -1.0649 | -0.0021 | -1.0666 | 0.259 | 0.398 | -1175.3 |
| Bayesian-PS | – | -1.0460 | +0.0000 | -1.0156 | 0.177 | 0.447 | -1115.8 |
| Bayesian-Walk-PS | ε_s = 0.413 | -1.0444 | +0.0000 | -1.0911 | 0.207 | 0.421 | -1203.6 |
| Bayesian-Threshold-PS | δ = -0.356 | -1.0479 | +0.0000 | -1.0588 | 0.115 | 0.458 | -1171.6 |
