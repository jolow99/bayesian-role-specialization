# Top-k at k = 5, and individual fits over 12 models (Fig. 5)

## Table 2 row

| Model | combo-r | marg-r | agg-LL | mean-LL |
|---|--:|--:|--:|--:|
| Top-5 | 0.370 | 0.516 | -4.317 | -4.863 |
| Top-27 (10-20 fit) | 0.243 | -0.000 | -2.809 | -3.296 |
| Random | 0.243 | -0.000 | -2.809 | -3.296 |
| Optimal | 0.239 | 0.484 | -5.860 | -6.383 |

## Individual fits (dropped: Bayesian-Threshold-BR, which collapses onto Bayesian-BR)

λ = 0.01 lapse on each player's marginal, refit (agg-LL) parameters.

| Model | Dominant for |
|---|--:|
| Bayesian-Walk-BR | 25 |
| Bayesian-Walk-PS | 7 |
| Bayesian-PS | 6 |
| Bayesian-BR | 9 |
| Bayesian-Threshold-PS | 5 |
| Random Walk | 5 |
| Top-5 | 13 |
| Random-to-Optimal | 5 |
| Optimal | 5 |
| Copy Others | 0 |
| Contradict Others | 7 |
| Random | 15 |

| Group | n | % |
|---|--:|--:|
| Bayesian | 52 | 51% |
| value-based baselines | 23 | 23% |
| noise / contrarian baselines | 27 | 26% |

Leading model posterior > 0.5: **50 of 102**
