# Refit of every model under a λ = 0.01 lapse, no floor

## Fitted parameters per objective

ε_s = probability of sticking with the previous role (= 1 − code's ε_switch). Random Walk ε = probability of switching.

| Model | Fit on | Params | combo-r | marg-r | agg-LL | mean-LL |
|---|---|---|--:|--:|--:|--:|
| Bayesian-BR | combo_r | τ_v=14.14 | 0.410 | 0.185 | -2.568 | -3.094 |
|  | agg_ll | τ_v=11.75 | 0.408 | 0.188 | -2.564 | -3.107 |
|  | mean_ll | τ_v=15.83 | 0.409 | 0.184 | -2.575 | -3.092 |
| Bayesian-Walk-BR | combo_r | τ_v=6.83, ε_s=0.45 | 0.505 | 0.360 | -2.505 | -2.895 |
|  | agg_ll | τ_v=11.67, ε_s=0.46 | 0.496 | 0.359 | -2.475 | -2.819 |
|  | mean_ll | τ_v=14.77, ε_s=0.39 | 0.492 | 0.348 | -2.482 | -2.803 |
| Bayesian-Threshold-BR | combo_r | τ_v=3.15, δ_BR=13.750 | 0.423 | 0.363 | -3.408 | -5.009 |
|  | agg_ll | τ_v=11.75, δ_BR=-70.000 | 0.408 | 0.188 | -2.564 | -3.107 |
|  | mean_ll | τ_v=15.92, δ_BR=-64.000 | 0.409 | 0.184 | -2.576 | -3.091 |
| Bayesian-Walk-PS | combo_r | ε_s=0.00 | 0.470 | 0.196 | -2.519 | -2.903 |
|  | agg_ll | ε_s=0.19 | 0.463 | 0.227 | -2.512 | -2.903 |
|  | mean_ll | ε_s=0.09 | 0.468 | 0.213 | -2.514 | -2.896 |
| Bayesian-Threshold-PS | combo_r | δ_PS=-0.607 | 0.476 | 0.158 | -2.525 | -3.010 |
|  | agg_ll | δ_PS=-0.616 | 0.473 | 0.171 | -2.515 | -2.912 |
|  | mean_ll | δ_PS=-1.000 | 0.470 | 0.196 | -2.519 | -2.903 |
| Top-27 | combo_r | k=4 | 0.373 | 0.507 | -4.474 | -5.190 |
|  | agg_ll | k=27 | 0.243 | -0.000 | -2.809 | -3.296 |
|  | mean_ll | k=27 | 0.243 | -0.000 | -2.809 | -3.296 |
| Random Walk | combo_r | ε=0.344 | 0.403 | 0.265 | -2.597 | -2.934 |
|  | agg_ll | ε=0.268 | 0.396 | 0.265 | -2.586 | -2.986 |
|  | mean_ll | ε=0.376 | 0.402 | 0.265 | -2.607 | -2.930 |

## Aggregate table (agg-LL fits)

New columns: every model fit and scored with λ = 0.01, no floor. `BR folder` = same scoring, paper params for non-threshold models. `paper` = current paper (floored).

| Model | Params | combo-r | marg-r | agg-LL | mean-LL | agg-LL (BR folder) | agg-LL (paper) |
|---|---|--:|--:|--:|--:|--:|--:|
| Bayesian-BR | τ_v=11.75 | 0.408 | 0.188 | -2.564 | -3.107 | -2.564 | -2.565 |
| Bayesian-Walk-BR | τ_v=11.67, ε_s=0.46 | 0.496 | 0.359 | -2.475 | -2.819 | -2.475 | -2.476 |
| Bayesian-Threshold-BR | τ_v=11.75, δ_BR=-70.000 | 0.408 | 0.188 | -2.564 | -3.107 | -2.564 | -9.862 |
| Bayesian-PS | none | 0.470 | 0.196 | -2.519 | -2.903 | -2.519 | -2.518 |
| Bayesian-Walk-PS | ε_s=0.19 | 0.463 | 0.227 | -2.512 | -2.903 | -2.512 | -2.512 |
| Bayesian-Threshold-PS | δ_PS=-0.616 | 0.473 | 0.171 | -2.515 | -2.912 | -2.515 | -8.036 |
| Optimal | none | 0.239 | 0.484 | -5.860 | -6.383 | -5.860 | -21.716 |
| Top-27 | k=27 | 0.243 | -0.000 | -2.809 | -3.296 | -3.973 | -12.421 |
| Random-to-Optimal | none | 0.180 | 0.354 | -3.516 | -4.258 | -3.516 | -5.068 |
| Copy Others | none | -0.004 | 0.265 | -3.893 | -6.292 | -3.893 | -7.803 |
| Contradict Others | none | 0.293 | -0.215 | -2.854 | -3.885 | -2.854 | -2.861 |
| Random Walk | ε=0.268 | 0.396 | 0.265 | -2.586 | -2.986 | -2.587 | -2.587 |
| Random | none | 0.243 | -0.000 | -2.809 | -3.296 | -2.809 | -2.809 |

## Dominant model per participant (n=102)

| Model | new (all refit) | BR folder | paper (λ = 0.05) |
|---|--:|--:|--:|
| Bayesian-BR | 6 | 3 | 7 |
| Bayesian-Walk-BR | 29 | 18 | 15 |
| Bayesian-Threshold-BR | 5 | 5 | 0 |
| Bayesian-PS | 6 | 6 | 7 |
| Bayesian-Walk-PS | 10 | 7 | 6 |
| Bayesian-Threshold-PS | 6 | 5 | 9 |
| Optimal | 6 | 5 | 9 |
| Top-27 | 16 | 22 | 19 |
| Random-to-Optimal | 5 | 4 | 5 |
| Copy Others | 0 | 0 | 0 |
| Contradict Others | 7 | 7 | 8 |
| Random Walk | 6 | 6 | 5 |
| Random | 0 | 14 | 12 |

## Top-k profile (agg-LL)

| k | agg-LL | combo-r |
|--:|--:|--:|
| 1 | -6.1827 | 0.220 |
| 2 | -5.4989 | 0.295 |
| 3 | -4.9689 | 0.351 |
| 4 | -4.4736 | 0.373 |
| 5 | -4.3172 | 0.370 |
| 6 | -4.0758 | 0.361 |
| 7 | -3.9733 | 0.351 |
| 8 | -3.8223 | 0.345 |
| 9 | -3.7056 | 0.336 |
| 10 | -3.6196 | 0.321 |
| 11 | -3.4587 | 0.325 |
| 12 | -3.4000 | 0.323 |
| 13 | -3.3571 | 0.311 |
| 14 | -3.3182 | 0.296 |
| 15 | -3.2914 | 0.286 |
| 16 | -3.2284 | 0.283 |
| 17 | -3.1471 | 0.287 |
| 18 | -3.0291 | 0.299 |
| 19 | -3.0063 | 0.295 |
| 20 | -2.9899 | 0.285 |
| 21 | -2.9194 | 0.283 |
| 22 | -2.9072 | 0.276 |
| 23 | -2.8697 | 0.268 |
| 24 | -2.8317 | 0.262 |
| 25 | -2.8142 | 0.258 |
| 26 | -2.8134 | 0.250 |
| 27 | -2.8085 | 0.243 |
