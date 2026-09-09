# R3_team_case with best-fitting model predictions (2026-09-02)

Model: **Bayesian Walk** (paper: Bayesian-Walk-BR), agg_ll fit from `2026-05-28-paper-figures/results.json`: tau_softmax = 12.0943, epsilon_switch = 0.5415. Stage-1: tau_prior = 4.6385, epsilon = 0.0624, memory `drift_prior_0.500`.

P(chosen) = the model's predicted probability of the role the player actually chose at that stage (from the start-of-stage posterior + the previous stage's roles). 'value-softmax only' drops the stickiness term (what Bayesian-BR alone would say). In a round's final stage the value matrix is often flat (every combo wins), so the softmax is uniform and P(chosen) collapses to the stickiness floor (1 - eps + eps/3 = 0.64 for a repeated role).

Primary output `R3_team_case.pdf` = `R3_team_case_2F8H1E_r2`.

### `01KQ6YDA7NHWHSTQNZB82F8H1E` round 2 — `114_222_222`, WIN, 4 live stages

| Stage | combo | value rank | P(chosen) P1 | P2 | P3 | mean | model top-role = chosen? | value-softmax only P1 | P2 | P3 |
|--:|---|--:|--:|--:|--:|--:|---|--:|--:|--:|
| 1 | FTT | 24/27 | 0.24 | 0.30 | 0.30 | **0.28** | n n n | 0.24 | 0.30 | 0.30 |
| 2 | MFT | 1/27 | 0.36 | 0.19 | 0.62 | **0.39** | n n Y | 0.67 | 0.35 | 0.29 |
| 3 | MFT | 2/27 | 0.91 | 0.67 | 0.61 | **0.73** | Y Y Y | 0.83 | 0.40 | 0.28 |
| 4 | MFT | 2/27 | 0.93 | 0.70 | 0.61 | **0.75** | Y Y Y | 0.87 | 0.45 | 0.29 |

Mean P(chosen) over all player-stages: **0.54** (chance 0.33); min 0.19. Model's most likely role = chosen role in **7/12** player-stages (green carets, in-card prediction chart). Humans' own reported inferences correct in **9/18** reports (green carets, belief charts). Model's most likely inferred role (end-of-stage belief) = played role in 22/24 cells (not drawn).

| Position | participant | dominant model (R4 posterior) | P(Bayesian-Walk-BR) |
|---|---|---|--:|
| P1 | `01KQ6YPPQD4CBF57CPAAHK4JGG` | Bayesian Walk | 0.33 |
| P2 | `01KQ6YS1CJVE8CYDXJMXBEBW3T` | Top-7 | 0.12 |
| P3 | `01KQ6YSWF78Z5NY1RH8T3QQYNK` | Bayesian Walk | 0.55 |

### `01KQ6YDF6T28MRGBK1E911B0J4` round 8 — `141_222_222`, WIN, 4 live stages

| Stage | combo | value rank | P(chosen) P1 | P2 | P3 | mean | model top-role = chosen? | value-softmax only P1 | P2 | P3 |
|--:|---|--:|--:|--:|--:|--:|---|--:|--:|--:|
| 1 | TFF | 1/27 | 0.48 | 0.55 | 0.55 | **0.53** | Y Y Y | 0.48 | 0.55 | 0.55 |
| 2 | TMF | 4/27 | 0.70 | 0.11 | 0.76 | **0.52** | Y n Y | 0.44 | 0.21 | 0.55 |
| 3 | TMF | 9/27 | 0.65 | 0.60 | 0.75 | **0.67** | Y Y Y | 0.35 | 0.27 | 0.54 |
| 4 | TMF | 10/27 | 0.64 | 0.64 | 0.64 | **0.64** | Y Y Y | 0.33 | 0.33 | 0.33 |

Mean P(chosen) over all player-stages: **0.59** (chance 0.33); min 0.11. Model's most likely role = chosen role in **11/12** player-stages (green carets, in-card prediction chart). Humans' own reported inferences correct in **11/18** reports (green carets, belief charts). Model's most likely inferred role (end-of-stage belief) = played role in 24/24 cells (not drawn).

| Position | participant | dominant model (R4 posterior) | P(Bayesian-Walk-BR) |
|---|---|---|--:|
| P1 | `01KQ6YRNQGRKZ178DVXDETEQ0Y` | Top-7 | 0.06 |
| P2 | `01KQ6Z0AATFB9GNSGA8NBWVY78` | Bayesian Walk | 0.58 |
| P3 | `01KQ6ZNMKV8DAPT3AF6CQD00CG` | Bayesian Walk | 0.38 |

### `01KQ6YDC2Q5SV4P2SZHVK72DV0` round 5 — `141_222_222`, WIN, 5 live stages

| Stage | combo | value rank | P(chosen) P1 | P2 | P3 | mean | model top-role = chosen? | value-softmax only P1 | P2 | P3 |
|--:|---|--:|--:|--:|--:|--:|---|--:|--:|--:|
| 1 | TFF | 1/27 | 0.39 | 0.37 | 0.37 | **0.38** | Y Y Y | 0.39 | 0.37 | 0.37 |
| 2 | TMF | 2/27 | 0.77 | 0.26 | 0.65 | **0.56** | Y n Y | 0.57 | 0.47 | 0.35 |
| 3 | TMF | 1/27 | 0.89 | 0.68 | 0.83 | **0.80** | Y Y Y | 0.80 | 0.41 | 0.68 |
| 4 | TMF | 3/27 | 0.90 | 0.63 | 0.80 | **0.78** | Y Y Y | 0.82 | 0.32 | 0.63 |
| 5 | TMF | 2/27 | 0.67 | 0.62 | 0.86 | **0.72** | Y Y Y | 0.39 | 0.30 | 0.75 |

Mean P(chosen) over all player-stages: **0.65** (chance 0.33); min 0.26. Model's most likely role = chosen role in **14/15** player-stages (green carets, in-card prediction chart). Humans' own reported inferences correct in **21/24** reports (green carets, belief charts). Model's most likely inferred role (end-of-stage belief) = played role in 30/30 cells (not drawn).

| Position | participant | dominant model (R4 posterior) | P(Bayesian-Walk-BR) |
|---|---|---|--:|
| P1 | `01KQ6YNJ7QNMVJ2QBY47K2GD8M` | Top-7 | 0.01 |
| P2 | `01KQ6YSJKKMKQJ7XHX5A1BBV0S` | Contradict Others | 0.04 |
| P3 | `01KQ6YTN275Y4DTNHJ8FJ1RXAR` | Optimal | 0.02 |

### `01KRBKSTM48HJWYZ0J4SRBRN0Z` round 8 — `411_141_114`, WIN, 4 live stages

| Stage | combo | value rank | P(chosen) P1 | P2 | P3 | mean | model top-role = chosen? | value-softmax only P1 | P2 | P3 |
|--:|---|--:|--:|--:|--:|--:|---|--:|--:|--:|
| 1 | FMM | 2/27 | 0.59 | 0.30 | 0.51 | **0.47** | Y n Y | 0.59 | 0.30 | 0.51 |
| 2 | FTM | 1/27 | 0.81 | 0.40 | 0.82 | **0.68** | Y n Y | 0.66 | 0.74 | 0.67 |
| 3 | FTM | 4/27 | 0.97 | 0.74 | 0.76 | **0.82** | Y Y Y | 0.94 | 0.53 | 0.56 |
| 4 | FTM | 4/27 | 0.69 | 0.63 | 0.65 | **0.65** | Y Y Y | 0.42 | 0.31 | 0.35 |

Mean P(chosen) over all player-stages: **0.66** (chance 0.33); min 0.30. Model's most likely role = chosen role in **10/12** player-stages (green carets, in-card prediction chart). Humans' own reported inferences correct in **14/18** reports (green carets, belief charts). Model's most likely inferred role (end-of-stage belief) = played role in 24/24 cells (not drawn).

| Position | participant | dominant model (R4 posterior) | P(Bayesian-Walk-BR) |
|---|---|---|--:|
| P1 | `01KRBKYQXPWYARSJEYN9ZE9A4G` | Bayesian Thresh-PS | 0.01 |
| P2 | `01KRBKZ6Z204TWFGJKMSNVMH02` | Contradict Others | 0.00 |
| P3 | `01KRBM9V0A38W9BWWZJBX1YT1S` | Optimal | 0.00 |

### `01KQ6YDF6T28MRGBK1E911B0J4` round 6 — `222_222_222`, WIN, 4 live stages

| Stage | combo | value rank | P(chosen) P1 | P2 | P3 | mean | model top-role = chosen? | value-softmax only P1 | P2 | P3 |
|--:|---|--:|--:|--:|--:|--:|---|--:|--:|--:|
| 1 | MTF | 17/27 | 0.25 | 0.26 | 0.49 | **0.33** | n n Y | 0.25 | 0.26 | 0.49 |
| 2 | MMF | 18/27 | 0.57 | 0.13 | 0.73 | **0.48** | Y n Y | 0.21 | 0.23 | 0.51 |
| 3 | MFF | 9/27 | 0.59 | 0.25 | 0.69 | **0.51** | Y n Y | 0.24 | 0.46 | 0.43 |
| 4 | FFF | 1/27 | 0.26 | 0.73 | 0.73 | **0.58** | n Y Y | 0.49 | 0.51 | 0.50 |

Mean P(chosen) over all player-stages: **0.47** (chance 0.33); min 0.13. Model's most likely role = chosen role in **7/12** player-stages (green carets, in-card prediction chart). Humans' own reported inferences correct in **12/18** reports (green carets, belief charts). Model's most likely inferred role (end-of-stage belief) = played role in 22/24 cells (not drawn).

| Position | participant | dominant model (R4 posterior) | P(Bayesian-Walk-BR) |
|---|---|---|--:|
| P1 | `01KQ6YRNQGRKZ178DVXDETEQ0Y` | Top-7 | 0.06 |
| P2 | `01KQ6Z0AATFB9GNSGA8NBWVY78` | Bayesian Walk | 0.58 |
| P3 | `01KQ6ZNMKV8DAPT3AF6CQD00CG` | Bayesian Walk | 0.38 |
