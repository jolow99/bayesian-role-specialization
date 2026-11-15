# Master table: value function × belief × group (every variant refit)

Value tables: `value_by_turns.npz`. Inference params fixed (2026-11-02). Fit objective: agg-LL. Best agg-LL per block in bold. Value = paper V(10) at every stage vs time-aware V(10 − 2·stage) (refit); 'time-aware (frozen)' = time-aware V scored at the paper-V fitted params, nothing refit; belief = mixed (paper) vs unmixed P_N.

## Summary: Bayesian-Walk-BR (winner) in every variant

| Group | Value fn | Belief | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|---|---|
| U | paper | mixed | τ_v = 8.54, ε_s = 0.416 | -0.9196 | -0.8892 | 0.591 | 0.482 | -939.7 |
| NU-as-U | paper | mixed | τ_v = 15.1, ε_s = 0.584 | -1.0283 | -0.9944 | 0.268 | 0.489 | -1096.4 |
| NU-as-U-nr | paper | mixed | τ_v = 8.54, ε_s = 0.416 | -1.0400 | -1.0574 | 0.286 | 0.488 | -1170.3 |
| NU-FTM | paper | n/a | τ_v = 16.5, ε_s = 0.57 | -1.0339 | -1.0031 | 0.207 | 0.481 | -1106.1 |
| U | paper | unmixed | τ_v = 8.19, ε_s = 0.427 | -0.9157 | -0.8881 | 0.629 | 0.481 | -936.6 |
| NU-as-U | paper | unmixed | τ_v = 15.5, ε_s = 0.582 | -1.0279 | -0.9990 | 0.285 | 0.489 | -1102.8 |
| NU-as-U-nr | paper | unmixed | τ_v = 8.19, ε_s = 0.427 | -1.0402 | -1.0689 | 0.304 | 0.488 | -1184.9 |
| U | time-aware | mixed | τ_v = 7.7, ε_s = 0.362 | -0.9248 | -0.8937 | 0.593 | 0.469 | -944.4 |
| NU-as-U | time-aware | mixed | τ_v = 11.1, ε_s = 0.54 | -1.0287 | -1.0011 | 0.311 | 0.485 | -1103.9 |
| NU-as-U-nr | time-aware | mixed | τ_v = 7.7, ε_s = 0.362 | -1.0374 | -1.0687 | 0.310 | 0.471 | -1181.2 |
| NU-FTM | time-aware | n/a | τ_v = 12.7, ε_s = 0.525 | -1.0339 | -1.0089 | 0.270 | 0.478 | -1113.2 |
| U | time-aware (frozen) | mixed | τ_v = 8.54, ε_s = 0.416 | -0.9259 | -0.8821 | 0.589 | 0.477 | -932.2 |
| NU-as-U | time-aware (frozen) | mixed | τ_v = 15.1, ε_s = 0.584 | -1.0307 | -0.9953 | 0.300 | 0.475 | -1096.7 |
| NU-as-U-nr | time-aware (frozen) | mixed | τ_v = 8.54, ε_s = 0.416 | -1.0328 | -1.0405 | 0.311 | 0.479 | -1149.2 |
| NU-FTM | time-aware (frozen) | n/a | τ_v = 16.5, ε_s = 0.57 | -1.0353 | -1.0019 | 0.263 | 0.470 | -1104.7 |
| U | time-aware | unmixed | τ_v = 7.62, ε_s = 0.373 | -0.9232 | -0.8922 | 0.591 | 0.471 | -941.9 |
| NU-as-U | time-aware | unmixed | τ_v = 11.3, ε_s = 0.549 | -1.0281 | -1.0020 | 0.314 | 0.484 | -1105.6 |
| NU-as-U-nr | time-aware | unmixed | τ_v = 7.62, ε_s = 0.373 | -1.0370 | -1.0711 | 0.313 | 0.471 | -1184.6 |
| U | time-aware (frozen) | unmixed | τ_v = 8.19, ε_s = 0.427 | -0.9241 | -0.8812 | 0.587 | 0.479 | -930.4 |
| NU-as-U | time-aware (frozen) | unmixed | τ_v = 15.5, ε_s = 0.582 | -1.0300 | -0.9959 | 0.303 | 0.474 | -1097.8 |
| NU-as-U-nr | time-aware (frozen) | unmixed | τ_v = 8.19, ε_s = 0.427 | -1.0330 | -1.0444 | 0.316 | 0.480 | -1154.3 |

## Best model by agg-LL in every variant

| Group | Value fn | Belief | Best model |
|---|---|---|---|
| U | paper | mixed | Bayesian-Walk-BR |
| NU-as-U | paper | mixed | Bayesian-Walk-BR |
| NU-as-U-nr | paper | mixed | Bayesian-Walk-BR |
| NU-FTM | paper | n/a | Bayesian-Walk-BR |
| U | paper | unmixed | Bayesian-Walk-BR |
| NU-as-U | paper | unmixed | Bayesian-Walk-BR |
| NU-as-U-nr | paper | unmixed | Bayesian-Walk-BR |
| U | time-aware | mixed | Bayesian-Walk-BR |
| NU-as-U | time-aware | mixed | Bayesian-Walk-BR |
| NU-as-U-nr | time-aware | mixed | Bayesian-Walk-BR |
| NU-FTM | time-aware | n/a | Bayesian-Walk-BR |
| U | time-aware (frozen) | mixed | Bayesian-Walk-BR |
| NU-as-U | time-aware (frozen) | mixed | Bayesian-Walk-BR |
| NU-as-U-nr | time-aware (frozen) | mixed | Bayesian-Walk-BR |
| NU-FTM | time-aware (frozen) | n/a | Bayesian-Walk-BR |
| U | time-aware | unmixed | Bayesian-Walk-BR |
| NU-as-U | time-aware | unmixed | Bayesian-Walk-BR |
| NU-as-U-nr | time-aware | unmixed | Bayesian-Walk-BR |
| U | time-aware (frozen) | unmixed | Bayesian-Walk-BR |
| NU-as-U | time-aware (frozen) | unmixed | Bayesian-Walk-BR |
| NU-as-U-nr | time-aware (frozen) | unmixed | Bayesian-Walk-BR |

## Full tables

### Updaters (n = 50), own beliefs, refit

**value = paper, belief = mixed**

| Model | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 9.48 | -0.9832 | -1.0127 | 0.165 | 0.408 | -1072.8 |
| **Bayesian-Walk-BR** | τ_v = 8.54, ε_s = 0.416 | **-0.9196** | -0.8892 | 0.591 | 0.482 | -939.7 |
| Bayesian-Threshold-BR | τ_v = 9.48, δ = -60.8 | -0.9832 | -1.0127 | 0.165 | 0.408 | -1072.7 |
| Bayesian-PS | – | -0.9847 | -0.9453 | 0.377 | 0.394 | -1004.1 |
| Bayesian-Walk-PS | ε_s = 0.413 | -0.9428 | -0.9200 | 0.564 | 0.442 | -973.8 |
| Bayesian-Threshold-PS | δ = -0.356 | -0.9548 | -0.9410 | 0.293 | 0.458 | -993.1 |

**value = paper, belief = unmixed**

| Model | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 8.53 | -0.9755 | -1.0311 | 0.338 | 0.410 | -1089.9 |
| **Bayesian-Walk-BR** | τ_v = 8.19, ε_s = 0.427 | **-0.9157** | -0.8881 | 0.629 | 0.481 | -936.6 |
| Bayesian-Threshold-BR | τ_v = 9.05, δ = -19.2 | -0.9751 | -1.0406 | 0.333 | 0.409 | -1097.0 |
| Bayesian-PS | – | -0.9650 | -1.1030 | 0.420 | 0.458 | -1167.5 |
| Bayesian-Walk-PS | ε_s = 0.709 | -0.9607 | -1.1218 | 0.495 | 0.468 | -1185.9 |
| Bayesian-Threshold-PS | δ = -0.992 | -0.9650 | -1.1030 | 0.420 | 0.458 | -1167.4 |

**value = time-aware, belief = mixed**

| Model | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 7.13 | -0.9976 | -1.0093 | 0.190 | 0.400 | -1068.7 |
| **Bayesian-Walk-BR** | τ_v = 7.7, ε_s = 0.362 | **-0.9248** | -0.8937 | 0.593 | 0.469 | -944.4 |
| Bayesian-Threshold-BR | τ_v = 7.13, δ = -45.8 | -0.9976 | -1.0093 | 0.190 | 0.400 | -1068.7 |
| Bayesian-PS | – | -0.9847 | -0.9453 | 0.377 | 0.394 | -1004.1 |
| Bayesian-Walk-PS | ε_s = 0.413 | -0.9428 | -0.9200 | 0.564 | 0.442 | -973.8 |
| Bayesian-Threshold-PS | δ = -0.356 | -0.9548 | -0.9410 | 0.293 | 0.458 | -993.1 |

**value = time-aware (frozen), belief = mixed**

| Model | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 9.48 | -1.0021 | -1.0086 | 0.180 | 0.397 | -1069.1 |
| **Bayesian-Walk-BR** | τ_v = 8.54, ε_s = 0.416 | **-0.9259** | -0.8821 | 0.589 | 0.477 | -932.2 |
| Bayesian-Threshold-BR | τ_v = 9.48, δ = -60.8 | -1.0021 | -1.0086 | 0.180 | 0.397 | -1069.1 |
| Bayesian-PS | – | -0.9847 | -0.9453 | 0.377 | 0.394 | -1004.1 |
| Bayesian-Walk-PS | ε_s = 0.413 | -0.9428 | -0.9200 | 0.564 | 0.442 | -973.8 |
| Bayesian-Threshold-PS | δ = -0.356 | -0.9548 | -0.9410 | 0.293 | 0.458 | -993.1 |

**value = time-aware, belief = unmixed**

| Model | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 6.86 | -0.9903 | -1.0174 | 0.213 | 0.406 | -1075.1 |
| **Bayesian-Walk-BR** | τ_v = 7.62, ε_s = 0.373 | **-0.9232** | -0.8922 | 0.591 | 0.471 | -941.9 |
| Bayesian-Threshold-BR | τ_v = 7.28, δ = -15.5 | -0.9879 | -1.0198 | 0.212 | 0.408 | -1076.8 |
| Bayesian-PS | – | -0.9650 | -1.1030 | 0.420 | 0.458 | -1167.5 |
| Bayesian-Walk-PS | ε_s = 0.709 | -0.9607 | -1.1218 | 0.495 | 0.468 | -1185.9 |
| Bayesian-Threshold-PS | δ = -0.992 | -0.9650 | -1.1030 | 0.420 | 0.458 | -1167.4 |

**value = time-aware (frozen), belief = unmixed**

| Model | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 8.53 | -0.9930 | -1.0103 | 0.206 | 0.405 | -1068.9 |
| **Bayesian-Walk-BR** | τ_v = 8.19, ε_s = 0.427 | **-0.9241** | -0.8812 | 0.587 | 0.479 | -930.4 |
| Bayesian-Threshold-BR | τ_v = 9.05, δ = -19.2 | -0.9924 | -1.0177 | 0.203 | 0.406 | -1075.5 |
| Bayesian-PS | – | -0.9650 | -1.1030 | 0.420 | 0.458 | -1167.5 |
| Bayesian-Walk-PS | ε_s = 0.709 | -0.9607 | -1.1218 | 0.495 | 0.468 | -1185.9 |
| Bayesian-Threshold-PS | δ = -0.992 | -0.9650 | -1.1030 | 0.420 | 0.458 | -1167.4 |

### Non-updaters (n = 52), paper fixed FTM prior, refit

**value = paper, belief = n/a**

| Model | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 17 | -1.0565 | -1.0626 | 0.036 | 0.408 | -1170.3 |
| **Bayesian-Walk-BR** | τ_v = 16.5, ε_s = 0.57 | **-1.0339** | -1.0031 | 0.207 | 0.481 | -1106.1 |
| Bayesian-Threshold-BR | τ_v = 18, δ = -31.2 | -1.0567 | -1.0642 | 0.049 | 0.411 | -1172.8 |
| Bayesian-PS | – | -1.1053 | -1.1030 | 0.393 | 0.394 | -1214.7 |
| Bayesian-Walk-PS | ε_s = 0.417 | -1.0412 | -1.0582 | 0.426 | 0.443 | -1167.3 |
| Bayesian-Threshold-PS | δ = -1 | -1.1053 | -1.1030 | 0.393 | 0.394 | -1214.7 |

**value = time-aware, belief = n/a**

| Model | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 12.5 | -1.0623 | -1.0600 | 0.186 | 0.385 | -1167.8 |
| **Bayesian-Walk-BR** | τ_v = 12.7, ε_s = 0.525 | **-1.0339** | -1.0089 | 0.270 | 0.478 | -1113.2 |
| Bayesian-Threshold-BR | τ_v = 12.5, δ = -70 | -1.0623 | -1.0600 | 0.186 | 0.385 | -1167.8 |
| Bayesian-PS | – | -1.1053 | -1.1030 | 0.393 | 0.394 | -1214.7 |
| Bayesian-Walk-PS | ε_s = 0.417 | -1.0412 | -1.0582 | 0.426 | 0.443 | -1167.3 |
| Bayesian-Threshold-PS | δ = -1 | -1.1053 | -1.1030 | 0.393 | 0.394 | -1214.7 |

**value = time-aware (frozen), belief = n/a**

| Model | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 17 | -1.0646 | -1.0612 | 0.177 | 0.366 | -1168.9 |
| **Bayesian-Walk-BR** | τ_v = 16.5, ε_s = 0.57 | **-1.0353** | -1.0019 | 0.263 | 0.470 | -1104.7 |
| Bayesian-Threshold-BR | τ_v = 18, δ = -31.2 | -1.0654 | -1.0612 | 0.173 | 0.362 | -1169.2 |
| Bayesian-PS | – | -1.1053 | -1.1030 | 0.393 | 0.394 | -1214.7 |
| Bayesian-Walk-PS | ε_s = 0.417 | -1.0412 | -1.0582 | 0.426 | 0.443 | -1167.3 |
| Bayesian-Threshold-PS | δ = -1 | -1.1053 | -1.1030 | 0.393 | 0.394 | -1214.7 |

### Non-updaters (n = 52) on updater beliefs, refit

**value = paper, belief = mixed**

| Model | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 16 | -1.0484 | -1.0502 | 0.143 | 0.408 | -1156.7 |
| **Bayesian-Walk-BR** | τ_v = 15.1, ε_s = 0.584 | **-1.0283** | -0.9944 | 0.268 | 0.489 | -1096.4 |
| Bayesian-Threshold-BR | τ_v = 16, δ = -70 | -1.0484 | -1.0502 | 0.143 | 0.408 | -1156.8 |
| Bayesian-PS | – | -1.0460 | -1.0156 | 0.177 | 0.447 | -1115.8 |
| Bayesian-Walk-PS | ε_s = 0.671 | -1.0378 | -1.0175 | 0.205 | 0.459 | -1120.0 |
| Bayesian-Threshold-PS | δ = -0.383 | -1.0414 | -1.0472 | 0.161 | 0.464 | -1154.7 |

**value = paper, belief = unmixed**

| Model | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 16.2 | -1.0483 | -1.0602 | 0.161 | 0.400 | -1169.1 |
| **Bayesian-Walk-BR** | τ_v = 15.5, ε_s = 0.582 | **-1.0279** | -0.9990 | 0.285 | 0.489 | -1102.8 |
| Bayesian-Threshold-BR | τ_v = 16.2, δ = -51.8 | -1.0483 | -1.0624 | 0.163 | 0.401 | -1172.3 |
| Bayesian-PS | – | -1.0566 | -1.2962 | 0.190 | 0.393 | -1437.6 |
| Bayesian-Walk-PS | ε_s = 1 | -1.0566 | -1.2962 | 0.190 | 0.393 | -1437.6 |
| Bayesian-Threshold-PS | δ = -0.995 | -1.0566 | -1.2962 | 0.190 | 0.393 | -1437.6 |

**value = time-aware, belief = mixed**

| Model | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 11.2 | -1.0546 | -1.0512 | 0.251 | 0.388 | -1157.4 |
| **Bayesian-Walk-BR** | τ_v = 11.1, ε_s = 0.54 | **-1.0287** | -1.0011 | 0.311 | 0.485 | -1103.9 |
| Bayesian-Threshold-BR | τ_v = 11.3, δ = -33.2 | -1.0546 | -1.0508 | 0.251 | 0.388 | -1157.1 |
| Bayesian-PS | – | -1.0460 | -1.0156 | 0.177 | 0.447 | -1115.8 |
| Bayesian-Walk-PS | ε_s = 0.671 | -1.0378 | -1.0175 | 0.205 | 0.459 | -1120.0 |
| Bayesian-Threshold-PS | δ = -0.383 | -1.0414 | -1.0472 | 0.161 | 0.464 | -1154.7 |

**value = time-aware (frozen), belief = mixed**

| Model | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 16 | -1.0580 | -1.0536 | 0.245 | 0.368 | -1159.8 |
| **Bayesian-Walk-BR** | τ_v = 15.1, ε_s = 0.584 | **-1.0307** | -0.9953 | 0.300 | 0.475 | -1096.7 |
| Bayesian-Threshold-BR | τ_v = 16, δ = -70 | -1.0580 | -1.0536 | 0.245 | 0.368 | -1159.8 |
| Bayesian-PS | – | -1.0460 | -1.0156 | 0.177 | 0.447 | -1115.8 |
| Bayesian-Walk-PS | ε_s = 0.671 | -1.0378 | -1.0175 | 0.205 | 0.459 | -1120.0 |
| Bayesian-Threshold-PS | δ = -0.383 | -1.0414 | -1.0472 | 0.161 | 0.464 | -1154.7 |

**value = time-aware, belief = unmixed**

| Model | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 11.2 | -1.0524 | -1.0594 | 0.262 | 0.392 | -1166.2 |
| **Bayesian-Walk-BR** | τ_v = 11.3, ε_s = 0.549 | **-1.0281** | -1.0020 | 0.314 | 0.484 | -1105.6 |
| Bayesian-Threshold-BR | τ_v = 11.2, δ = -37.8 | -1.0524 | -1.0592 | 0.262 | 0.392 | -1166.0 |
| Bayesian-PS | – | -1.0566 | -1.2962 | 0.190 | 0.393 | -1437.6 |
| Bayesian-Walk-PS | ε_s = 1 | -1.0566 | -1.2962 | 0.190 | 0.393 | -1437.6 |
| Bayesian-Threshold-PS | δ = -0.995 | -1.0566 | -1.2962 | 0.190 | 0.393 | -1437.6 |

**value = time-aware (frozen), belief = unmixed**

| Model | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 16.2 | -1.0559 | -1.0559 | 0.257 | 0.373 | -1162.0 |
| **Bayesian-Walk-BR** | τ_v = 15.5, ε_s = 0.582 | **-1.0300** | -0.9959 | 0.303 | 0.474 | -1097.8 |
| Bayesian-Threshold-BR | τ_v = 16.2, δ = -51.8 | -1.0561 | -1.0555 | 0.255 | 0.372 | -1161.7 |
| Bayesian-PS | – | -1.0566 | -1.2962 | 0.190 | 0.393 | -1437.6 |
| Bayesian-Walk-PS | ε_s = 1 | -1.0566 | -1.2962 | 0.190 | 0.393 | -1437.6 |
| Bayesian-Threshold-PS | δ = -0.995 | -1.0566 | -1.2962 | 0.190 | 0.393 | -1437.6 |

### Non-updaters (n = 52) on updater beliefs AND updater params, no refit

**value = paper, belief = mixed**

| Model | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 9.48 | -1.0627 | -1.0757 | 0.142 | 0.402 | -1188.2 |
| **Bayesian-Walk-BR** | τ_v = 8.54, ε_s = 0.416 | **-1.0400** | -1.0574 | 0.286 | 0.488 | -1170.3 |
| Bayesian-Threshold-BR | τ_v = 9.48, δ = -60.8 | -1.0627 | -1.0757 | 0.142 | 0.402 | -1188.2 |
| Bayesian-PS | – | -1.0460 | -1.0156 | 0.177 | 0.447 | -1115.8 |
| Bayesian-Walk-PS | ε_s = 0.413 | -1.0444 | -1.0911 | 0.207 | 0.421 | -1203.6 |
| Bayesian-Threshold-PS | δ = -0.356 | -1.0479 | -1.0588 | 0.115 | 0.458 | -1171.6 |

**value = paper, belief = unmixed**

| Model | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 8.53 | -1.0654 | -1.1166 | 0.171 | 0.389 | -1237.8 |
| **Bayesian-Walk-BR** | τ_v = 8.19, ε_s = 0.427 | **-1.0402** | -1.0689 | 0.304 | 0.488 | -1184.9 |
| Bayesian-Threshold-BR | τ_v = 9.05, δ = -19.2 | -1.0722 | -1.1379 | 0.166 | 0.389 | -1263.7 |
| Bayesian-PS | – | -1.0566 | -1.2962 | 0.190 | 0.393 | -1437.6 |
| Bayesian-Walk-PS | ε_s = 0.709 | -1.0628 | -1.3485 | 0.201 | 0.383 | -1497.3 |
| Bayesian-Threshold-PS | δ = -0.992 | -1.0566 | -1.2969 | 0.190 | 0.393 | -1438.6 |

**value = time-aware, belief = mixed**

| Model | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 7.13 | -1.0648 | -1.0666 | 0.259 | 0.398 | -1175.3 |
| **Bayesian-Walk-BR** | τ_v = 7.7, ε_s = 0.362 | **-1.0374** | -1.0687 | 0.310 | 0.471 | -1181.2 |
| Bayesian-Threshold-BR | τ_v = 7.13, δ = -45.8 | -1.0649 | -1.0666 | 0.259 | 0.398 | -1175.3 |
| Bayesian-PS | – | -1.0460 | -1.0156 | 0.177 | 0.447 | -1115.8 |
| Bayesian-Walk-PS | ε_s = 0.413 | -1.0444 | -1.0911 | 0.207 | 0.421 | -1203.6 |
| Bayesian-Threshold-PS | δ = -0.356 | -1.0479 | -1.0588 | 0.115 | 0.458 | -1171.6 |

**value = time-aware (frozen), belief = mixed**

| Model | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 9.48 | -1.0558 | -1.0535 | 0.254 | 0.394 | -1160.2 |
| **Bayesian-Walk-BR** | τ_v = 8.54, ε_s = 0.416 | **-1.0328** | -1.0405 | 0.311 | 0.479 | -1149.2 |
| Bayesian-Threshold-BR | τ_v = 9.48, δ = -60.8 | -1.0558 | -1.0535 | 0.254 | 0.394 | -1160.2 |
| Bayesian-PS | – | -1.0460 | -1.0156 | 0.177 | 0.447 | -1115.8 |
| Bayesian-Walk-PS | ε_s = 0.413 | -1.0444 | -1.0911 | 0.207 | 0.421 | -1203.6 |
| Bayesian-Threshold-PS | δ = -0.356 | -1.0479 | -1.0588 | 0.115 | 0.458 | -1171.6 |

**value = time-aware, belief = unmixed**

| Model | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 6.86 | -1.0631 | -1.0914 | 0.271 | 0.397 | -1203.4 |
| **Bayesian-Walk-BR** | τ_v = 7.62, ε_s = 0.373 | **-1.0370** | -1.0711 | 0.313 | 0.471 | -1184.6 |
| Bayesian-Threshold-BR | τ_v = 7.28, δ = -15.5 | -1.0661 | -1.1062 | 0.266 | 0.399 | -1221.7 |
| Bayesian-PS | – | -1.0566 | -1.2962 | 0.190 | 0.393 | -1437.6 |
| Bayesian-Walk-PS | ε_s = 0.709 | -1.0628 | -1.3485 | 0.201 | 0.383 | -1497.3 |
| Bayesian-Threshold-PS | δ = -0.992 | -1.0566 | -1.2969 | 0.190 | 0.393 | -1438.6 |

**value = time-aware (frozen), belief = unmixed**

| Model | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |
|---|---|---|---|---|---|---|
| Bayesian-BR | τ_v = 8.53 | -1.0553 | -1.0719 | 0.267 | 0.399 | -1180.7 |
| **Bayesian-Walk-BR** | τ_v = 8.19, ε_s = 0.427 | **-1.0330** | -1.0444 | 0.316 | 0.480 | -1154.3 |
| Bayesian-Threshold-BR | τ_v = 9.05, δ = -19.2 | -1.0557 | -1.0796 | 0.266 | 0.399 | -1191.2 |
| Bayesian-PS | – | -1.0566 | -1.2962 | 0.190 | 0.393 | -1437.6 |
| Bayesian-Walk-PS | ε_s = 0.709 | -1.0628 | -1.3485 | 0.201 | 0.383 | -1497.3 |
| Bayesian-Threshold-PS | δ = -0.992 | -1.0566 | -1.2969 | 0.190 | 0.393 | -1438.6 |

