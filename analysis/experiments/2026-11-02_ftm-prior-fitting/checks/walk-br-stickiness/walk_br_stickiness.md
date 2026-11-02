# Updaters' Walk-BR stickiness

## Walk-BR refit on updaters by objective

| Objective | τ_v | ε_s (stick) | agg-LL | mean-LL | marg-r |
|---|--:|--:|--:|--:|--:|
| agg_ll | 8.54 | 0.58 | -0.9196 | -0.8892 | 0.591 |
| mean_ll | 12.23 | 0.47 | -0.9265 | -0.8748 | 0.577 |
| marg_r | 5.44 | 0.81 | -0.9399 | -1.0028 | 0.607 |

## Switch rate (764 updater stage transitions)

Human 32.1%; Walk-BR (agg-LL fit) predicts 24.8%, at most 41.6%; Bayesian-BR (τ_v = 9.48) predicts 60.0%.

| Transition | n | human | Walk-BR | Bayesian-BR |
|---|--:|--:|--:|--:|
| 1 -> 2 | 295 | 37.3% | 23.9% | 58.2% |
| 2 -> 3 | 274 | 29.2% | 24.3% | 58.7% |
| 3 -> 4 | 157 | 26.8% | 26.6% | 64.1% |
| 4 -> 5 | 38 | 34.2% | 27.7% | 66.6% |
