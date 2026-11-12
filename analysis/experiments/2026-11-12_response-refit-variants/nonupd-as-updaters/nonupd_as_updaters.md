# Non-updaters scored as updaters (updater beliefs and updater parameters, nothing refit)

n = 52 non-updaters, 1099 choices. Beliefs: mixed posterior from the stat prior (τ_p, ε, β from the shared inference fit). Parameters: the updaters' agg-LL fit (U-mixed). Refit = NU-mixed (same beliefs, parameters fit on non-updaters).

| Model | Updater params | agg-LL | refit agg-LL | mean-LL | marg-r | combo-r | Σ participant LL | refit Σ LL |
|---|---|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 9.48 | -1.0627 | -1.0484 | -1.0757 | 0.142 | 0.402 | -1188.2 | -1156.7 |
| **Bayesian-Walk-BR** | τ_v = 8.54, ε_s = 0.416 | **-1.0400** | -1.0283 | -1.0574 | 0.286 | 0.488 | -1170.3 | -1096.4 |
| Bayesian-Threshold-BR | τ_v = 9.48, δ = -60.8 | -1.0627 | -1.0484 | -1.0757 | 0.142 | 0.402 | -1188.2 | -1156.8 |
| Bayesian-PS | – | -1.0460 | -1.0460 | -1.0156 | 0.177 | 0.447 | -1115.8 | -1115.8 |
| Bayesian-Walk-PS | ε_s = 0.413 | -1.0444 | -1.0378 | -1.0911 | 0.207 | 0.421 | -1203.6 | -1120.0 |
| Bayesian-Threshold-PS | δ = -0.356 | -1.0479 | -1.0414 | -1.0588 | 0.115 | 0.458 | -1171.6 | -1154.7 |

Non-Bayesian baselines on non-updaters (2026-11-02 non-updater fits):

| Model | agg-LL | Σ participant LL |
|---|---|---|
| Random Walk | -1.0531 | -1155.6 |
| Random | -1.0986 | -1207.4 |
| Top-5 | -1.3550 | -1535.8 |

Individual fits: Bayesian model best for **25/52** at updater params (vs 24/52 refit on non-updaters). Figure: indiv_fits.png

| Best model | updater params | refit |
|---|---|---|
| Bayesian-Walk-BR | 8 | 7 |
| Bayesian-Walk-PS | 1 | 3 |
| Bayesian-PS | 4 | 2 |
| Bayesian-BR | 1 | 5 |
| Bayesian-Threshold-BR | 2 | 0 |
| Bayesian-Threshold-PS | 9 | 7 |
| Random Walk | 2 | 2 |
| Top-5 | 5 | 4 |
| Random-to-Optimal | 1 | 4 |
| Optimal | 2 | 2 |
| Contradict Others | 1 | 1 |
| Prior Sampling | 12 | 12 |
| Random | 4 | 3 |
