# Tolerant Threshold-BR refit (λ = 0.01, no floor)

## Fitted (τ_v, δ)

| Fit on | τ_v | δ | δ plateau (equal-best) | combo-r | marg-r | agg-LL | mean-LL |
|---|--:|--:|---|--:|--:|--:|--:|
| combo_r | 3.15 | 13.75 | 13.75 to 13.75 | 0.423 | 0.363 | -3.408 | -5.009 |
| agg_ll | 11.75 | -70.00 | -70.00 to -68.50 | 0.408 | 0.188 | -2.564 | -3.107 |
| mean_ll | 15.92 | -64.00 | -64.00 to -62.00 | 0.409 | 0.184 | -2.576 | -3.091 |

δ = -70 is exactly Bayesian-BR (asserted).

## Observed switches: V(chosen) − V(previous role)

591 switches; min -61.04, max 50.52; 49.6% go to a role of no higher value than the current one.

| percentile | 1 | 5 | 25 | 50 | 75 | 95 |
|---|--:|--:|--:|--:|--:|--:|
| gap | -38.63 | -16.54 | -2.61 | 0.06 | 6.18 | 28.79 |

## Aggregate table (both tolerant threshold models at their agg-LL fits; others at paper params)

agg-LL / mean-LL: λ = 0.01 lapse, no floor. combo-r / marg-r: raw predictions. Paper columns = current paper (1e-12 / 1e-20 floors).

| Model | combo-r | marg-r | agg-LL | mean-LL | agg-LL (paper) | mean-LL (paper) |
|---|--:|--:|--:|--:|--:|--:|
| Bayesian-Walk-BR | 0.495 | 0.359 | -2.475 | -2.816 | -2.476 | -2.827 |
| Bayesian-Walk-PS | 0.463 | 0.226 | -2.512 | -2.902 | -2.512 | -2.906 |
| Bayesian-PS | 0.470 | 0.196 | -2.519 | -2.903 | -2.518 | -2.904 |
| Bayesian-BR | 0.408 | 0.188 | -2.564 | -3.108 | -2.565 | -3.118 |
| Bayesian-Threshold-BR | 0.408 | 0.188 | -2.564 | -3.107 | -9.862 | -26.934 |
| Bayesian-Threshold-PS | 0.473 | 0.171 | -2.515 | -2.912 | -8.036 | -25.154 |
| Random Walk | 0.392 | 0.265 | -2.587 | -3.008 | -2.587 | -3.016 |
| Top-7 | 0.351 | 0.521 | -3.973 | -4.488 | -12.421 | -20.744 |
| Random-to-Optimal | 0.180 | 0.354 | -3.516 | -4.258 | -5.068 | -12.683 |
| Optimal | 0.239 | 0.484 | -5.860 | -6.383 | -21.716 | -36.422 |
| Copy Others | -0.004 | 0.265 | -3.893 | -6.292 | -7.803 | -31.781 |
| Contradict Others | 0.293 | -0.215 | -2.854 | -3.885 | -2.861 | -3.949 |
| Random | 0.243 | -0.000 | -2.809 | -3.296 | -2.809 | -3.296 |

## Dominant model per participant (n=102)

| Model | new (both tolerant) | PS folder (old Thresh-BR) | paper (λ = 0.05) |
|---|--:|--:|--:|
| Bayesian-Walk-BR | 18 | 18 | 15 |
| Bayesian-Walk-PS | 7 | 7 | 6 |
| Bayesian-PS | 6 | 6 | 7 |
| Bayesian-BR | 3 | 8 | 7 |
| Bayesian-Threshold-BR | 5 | 0 | 0 |
| Bayesian-Threshold-PS | 5 | 5 | 9 |
| Random Walk | 6 | 6 | 5 |
| Top-7 | 22 | 22 | 19 |
| Random-to-Optimal | 4 | 4 | 5 |
| Optimal | 5 | 5 | 9 |
| Copy Others | 0 | 0 | 0 |
| Contradict Others | 7 | 7 | 8 |
| Random | 14 | 14 | 12 |

## δ profile (agg-LL) at τ_v = 11.75, every 10

| δ | agg-LL |
|--:|--:|
| -70 | -2.5638 |
| -60 | -2.5639 |
| -50 | -2.5643 |
| -40 | -2.5666 |
| -30 | -2.5705 |
| -20 | -2.5790 |
| -10 | -2.6321 |
| 0 | -2.9092 |
| 10 | -3.1292 |
| 20 | -3.2359 |
| 30 | -3.2945 |
| 40 | -3.3297 |
| 50 | -3.3279 |
| 60 | -3.3291 |
| 70 | -3.3310 |
