# Tolerant Threshold-PS refit (λ = 0.01, no floor)

## Fitted delta

| Fit on | delta | plateau of equal-best delta | combo-r | marg-r | agg-LL | mean-LL |
|---|--:|---|--:|--:|--:|--:|
| combo_r | -0.607 | -0.607 to -0.607 | 0.476 | 0.158 | -2.525 | -3.010 |
| agg_ll | -0.616 | -0.616 to -0.616 | 0.473 | 0.171 | -2.515 | -2.912 |
| mean_ll | -1.000 | -1.000 to -0.617 | 0.470 | 0.196 | -2.519 | -2.903 |

delta = -1 is exactly Bayesian-PS (asserted).

## Observed switches: P(chosen) − P(previous role)

591 switches; min -0.616; 94.9% go to a role no more likely than the current one.

| percentile | 1 | 5 | 25 | 50 | 75 | 95 |
|---|--:|--:|--:|--:|--:|--:|
| gap | -0.615 | -0.603 | -0.498 | -0.467 | -0.367 | 0.000 |

## Aggregate table (agg-LL-fit δ; other models at paper params)

agg-LL / mean-LL use a λ = 0.01 lapse with no floor; combo-r / marg-r are on raw predictions. Paper columns = current paper (1e-12 / 1e-20 floors).

| Model | combo-r | marg-r | agg-LL | mean-LL | agg-LL (paper) | mean-LL (paper) |
|---|--:|--:|--:|--:|--:|--:|
| Bayesian-Walk-BR | 0.495 | 0.359 | -2.475 | -2.816 | -2.476 | -2.827 |
| Bayesian-Walk-PS | 0.463 | 0.226 | -2.512 | -2.902 | -2.512 | -2.906 |
| Bayesian-PS | 0.470 | 0.196 | -2.519 | -2.903 | -2.518 | -2.904 |
| Bayesian-BR | 0.408 | 0.188 | -2.564 | -3.108 | -2.565 | -3.118 |
| Bayesian-Threshold-BR | 0.284 | 0.265 | -3.771 | -5.409 | -9.862 | -26.934 |
| Bayesian-Threshold-PS | 0.473 | 0.171 | -2.515 | -2.912 | -8.036 | -25.154 |
| Random Walk | 0.392 | 0.265 | -2.587 | -3.008 | -2.587 | -3.016 |
| Top-7 | 0.351 | 0.521 | -3.973 | -4.488 | -12.421 | -20.744 |
| Random-to-Optimal | 0.180 | 0.354 | -3.516 | -4.258 | -5.068 | -12.683 |
| Optimal | 0.239 | 0.484 | -5.860 | -6.383 | -21.716 | -36.422 |
| Copy Others | -0.004 | 0.265 | -3.893 | -6.292 | -7.803 | -31.781 |
| Contradict Others | 0.293 | -0.215 | -2.854 | -3.885 | -2.861 | -3.949 |
| Random | 0.243 | -0.000 | -2.809 | -3.296 | -2.809 | -3.296 |

## Dominant model per participant (n=102)

| Model | new (λ = 0.01) | paper (λ = 0.05, old Thresh-PS) |
|---|--:|--:|
| Bayesian-Walk-BR | 18 | 15 |
| Bayesian-Walk-PS | 7 | 6 |
| Bayesian-PS | 6 | 7 |
| Bayesian-BR | 8 | 7 |
| Bayesian-Threshold-BR | 0 | 0 |
| Bayesian-Threshold-PS | 5 | 9 |
| Random Walk | 6 | 5 |
| Top-7 | 22 | 19 |
| Random-to-Optimal | 4 | 5 |
| Optimal | 5 | 9 |
| Copy Others | 0 | 0 |
| Contradict Others | 7 | 8 |
| Random | 14 | 12 |

## delta profile (agg-LL), every 0.1

| delta | agg-LL |
|--:|--:|
| -1.00 | -2.5191 |
| -0.90 | -2.5191 |
| -0.80 | -2.5191 |
| -0.70 | -2.5191 |
| -0.60 | -2.5284 |
| -0.50 | -2.5459 |
| -0.40 | -2.9936 |
| -0.30 | -3.1711 |
| -0.20 | -3.2400 |
| -0.10 | -3.2424 |
| 0.00 | -3.2797 |
| 0.10 | -3.3365 |
| 0.20 | -3.3425 |
| 0.30 | -3.3604 |
| 0.40 | -3.3572 |
| 0.50 | -3.3503 |
| 0.60 | -3.3503 |
| 0.70 | -3.3503 |
| 0.80 | -3.3503 |
| 0.90 | -3.3503 |
| 1.00 | -3.3503 |
