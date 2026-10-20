# Lapse vs floor vs none

Threshold-PS delta: none=0.255 (combo-r fit; agg-LL undefined), lapse=0.440 (plateau 0.44–1.00), floor=0.435

## No lapse, no floor

| Model | combo-r | marg-r | agg-LL | mean-LL | # dominant |
|---|--:|--:|--:|--:|--:|
| Bayesian-Walk-BR | 0.495 | 0.359 | −2.476 | −2.827 | 22 |
| Bayesian-Walk-PS | 0.463 | 0.226 | −2.512 | −2.906 | 9 |
| Bayesian-PS | 0.470 | 0.196 | −2.518 | −2.904 | 9 |
| Bayesian-BR | 0.408 | 0.188 | −2.565 | −3.118 | 8 |
| Bayesian-Threshold-BR | 0.284 | 0.265 | −∞ | −∞ | 0 |
| Bayesian-Threshold-PS | 0.355 | 0.287 | −∞ | −∞ | 1 |
| Random Walk | 0.392 | 0.265 | −2.587 | −3.016 | 5 |
| Top-7 | 0.351 | 0.521 | −∞ | −∞ | 22 |
| Random-to-Optimal | 0.180 | 0.354 | −∞ | −∞ | 5 |
| Optimal | 0.239 | 0.484 | −∞ | −∞ | 0 |
| Copy Others | −0.004 | 0.265 | −∞ | −∞ | 0 |
| Contradict Others | 0.293 | −0.215 | −2.861 | −3.949 | 7 |
| Random | 0.243 | −0.000 | −2.809 | −3.296 | 14 |
| Always Stick | 0.310 | 0.265 | −∞ | −∞ | 0 |

## Lapse (λ = 0.05)

| Model | combo-r | marg-r | agg-LL | mean-LL | # dominant |
|---|--:|--:|--:|--:|--:|
| Bayesian-Walk-BR | 0.495 | 0.359 | −2.474 | −2.802 | 15 |
| Bayesian-Walk-PS | 0.463 | 0.226 | −2.515 | −2.890 | 6 |
| Bayesian-PS | 0.468 | 0.196 | −2.523 | −2.901 | 7 |
| Bayesian-BR | 0.410 | 0.188 | −2.560 | −3.093 | 7 |
| Bayesian-Threshold-BR | 0.286 | 0.265 | −3.285 | −4.508 | 0 |
| Bayesian-Threshold-PS | 0.328 | 0.276 | −2.986 | −4.230 | 8 |
| Random Walk | 0.393 | 0.265 | −2.587 | −2.982 | 5 |
| Top-7 | 0.353 | 0.521 | −3.321 | −3.820 | 19 |
| Random-to-Optimal | 0.182 | 0.354 | −3.317 | −3.891 | 5 |
| Optimal | 0.241 | 0.484 | −4.609 | −5.124 | 9 |
| Copy Others | 0.003 | 0.265 | −3.556 | −5.219 | 0 |
| Contradict Others | 0.296 | −0.215 | −2.832 | −3.744 | 8 |
| Random | 0.243 | −0.000 | −2.809 | −3.296 | 12 |
| Always Stick | 0.312 | 0.265 | −3.041 | −4.286 | 1 |

## Floor (1e-12)

| Model | combo-r | marg-r | agg-LL | mean-LL | # dominant |
|---|--:|--:|--:|--:|--:|
| Bayesian-Walk-BR | 0.495 | 0.359 | −2.476 | −2.827 | 22 |
| Bayesian-Walk-PS | 0.463 | 0.226 | −2.512 | −2.906 | 9 |
| Bayesian-PS | 0.470 | 0.196 | −2.518 | −2.904 | 9 |
| Bayesian-BR | 0.408 | 0.188 | −2.565 | −3.118 | 8 |
| Bayesian-Threshold-BR | 0.284 | 0.265 | −9.862 | −16.540 | 0 |
| Bayesian-Threshold-PS | 0.325 | 0.276 | −8.036 | −15.455 | 1 |
| Random Walk | 0.392 | 0.265 | −2.587 | −3.016 | 5 |
| Top-7 | 0.351 | 0.521 | −12.421 | −12.893 | 22 |
| Random-to-Optimal | 0.180 | 0.354 | −5.068 | −8.616 | 5 |
| Optimal | 0.239 | 0.484 | −21.716 | −21.917 | 0 |
| Copy Others | −0.004 | 0.265 | −7.803 | −19.474 | 0 |
| Contradict Others | 0.293 | −0.215 | −2.861 | −3.949 | 7 |
| Random | 0.243 | −0.000 | −2.809 | −3.296 | 14 |
| Always Stick | 0.310 | 0.265 | −8.092 | −15.512 | 0 |
