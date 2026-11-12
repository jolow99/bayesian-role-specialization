# Bayesian response models: mixed vs unmixed posterior, and non-updaters on updater beliefs

Fixed inference params (2026-11-02 params.json): τ_p = 6.714, ε = 0.218, β = 0.683. Posteriors start from the stat prior P_0. Free parameters maximise agg-LL on the group's rows; best agg-LL in bold.

## Updaters

### Updaters — mixed posterior (1 − β)·P_N + β·P_0 (paper) (n = 50, 1064 choices)

| Model | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 9.48 | -0.9832 | -1.0127 | 0.165 | 0.408 | -1072.8 |
| **Bayesian-Walk-BR** | τ_v = 8.54, ε_s = 0.416 | **-0.9196** | -0.8892 | 0.591 | 0.482 | -939.7 |
| Bayesian-Threshold-BR | τ_v = 9.48, δ = -60.8 | -0.9832 | -1.0127 | 0.165 | 0.408 | -1072.7 |
| Bayesian-PS | – | -0.9847 | -0.9453 | 0.377 | 0.394 | -1004.1 |
| Bayesian-Walk-PS | ε_s = 0.413 | -0.9428 | -0.9200 | 0.564 | 0.442 | -973.8 |
| Bayesian-Threshold-PS | δ = -0.356 | -0.9548 | -0.9410 | 0.293 | 0.458 | -993.1 |

&nbsp;

### Updaters — unmixed posterior P_N (n = 50, 1064 choices)

| Model | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 8.53 | -0.9755 | -1.0311 | 0.338 | 0.410 | -1089.9 |
| **Bayesian-Walk-BR** | τ_v = 8.19, ε_s = 0.427 | **-0.9157** | -0.8881 | 0.629 | 0.481 | -936.6 |
| Bayesian-Threshold-BR | τ_v = 9.05, δ = -19.2 | -0.9751 | -1.0406 | 0.333 | 0.409 | -1097.0 |
| Bayesian-PS | – | -0.9650 | -1.1030 | 0.420 | 0.458 | -1167.5 |
| Bayesian-Walk-PS | ε_s = 0.709 | -0.9607 | -1.1218 | 0.495 | 0.468 | -1185.9 |
| Bayesian-Threshold-PS | δ = -0.992 | -0.9650 | -1.1030 | 0.420 | 0.458 | -1167.4 |

&nbsp;

## Non-updaters

### Non-updaters — paper: fixed FTM prior (Table S6 / Fig S6) (n = 52, 1099 choices)

| Model | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 17 | -1.0565 | -1.0626 | 0.036 | 0.408 | -1170.3 |
| **Bayesian-Walk-BR** | τ_v = 16.5, ε_s = 0.57 | **-1.0339** | -1.0031 | 0.207 | 0.481 | -1106.1 |
| Bayesian-Threshold-BR | τ_v = 18, δ = -31.2 | -1.0567 | -1.0642 | 0.049 | 0.411 | -1172.8 |
| Bayesian-PS | – | -1.1053 | -1.1030 | 0.393 | 0.394 | -1214.7 |
| Bayesian-Walk-PS | ε_s = 0.417 | -1.0412 | -1.0582 | 0.426 | 0.443 | -1167.3 |
| Bayesian-Threshold-PS | δ = -1 | -1.1053 | -1.1030 | 0.393 | 0.394 | -1214.7 |

&nbsp;

### Non-updaters — updater beliefs: mixed posterior (1 − β)·P_N + β·P_0 (n = 52, 1099 choices)

| Model | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 16 | -1.0484 | -1.0502 | 0.143 | 0.408 | -1156.7 |
| **Bayesian-Walk-BR** | τ_v = 15.1, ε_s = 0.584 | **-1.0283** | -0.9944 | 0.268 | 0.489 | -1096.4 |
| Bayesian-Threshold-BR | τ_v = 16, δ = -70 | -1.0484 | -1.0502 | 0.143 | 0.408 | -1156.8 |
| Bayesian-PS | – | -1.0460 | -1.0156 | 0.177 | 0.447 | -1115.8 |
| Bayesian-Walk-PS | ε_s = 0.671 | -1.0378 | -1.0175 | 0.205 | 0.459 | -1120.0 |
| Bayesian-Threshold-PS | δ = -0.383 | -1.0414 | -1.0472 | 0.161 | 0.464 | -1154.7 |

&nbsp;

### Non-updaters — updater beliefs: unmixed posterior P_N (n = 52, 1099 choices)

| Model | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 16.2 | -1.0483 | -1.0602 | 0.161 | 0.400 | -1169.1 |
| **Bayesian-Walk-BR** | τ_v = 15.5, ε_s = 0.582 | **-1.0279** | -0.9990 | 0.285 | 0.489 | -1102.8 |
| Bayesian-Threshold-BR | τ_v = 16.2, δ = -51.8 | -1.0483 | -1.0624 | 0.163 | 0.401 | -1172.3 |
| Bayesian-PS | – | -1.0566 | -1.2962 | 0.190 | 0.393 | -1437.6 |
| Bayesian-Walk-PS | ε_s = 1 | -1.0566 | -1.2962 | 0.190 | 0.393 | -1437.6 |
| Bayesian-Threshold-PS | δ = -0.995 | -1.0566 | -1.2962 | 0.190 | 0.393 | -1437.6 |

&nbsp;

In NU-paper, Bayesian-PS is the paper's Prior Sampling (sampling from the FTM prior).
