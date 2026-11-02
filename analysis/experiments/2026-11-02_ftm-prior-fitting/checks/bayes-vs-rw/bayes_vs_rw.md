# Bayesian response models vs Random / Random Walk

Each group's own fit. TVD: mean total variation distance of predicted role distributions over the group's choices. ΔLL vs RW: per-participant log-likelihood minus Random Walk's, per choice (mean over participants).

## non-updaters (52 participants, 1099 choices)

| Model | TVD vs Random | TVD vs Random Walk | ΔLL vs RW / choice | better than RW | Wilcoxon p |
|---|--:|--:|--:|--:|--:|
| Bayesian-BR | 0.111 | 0.280 | -0.0193 | 23/52 | 0.483 |
| Bayesian-Walk-BR | 0.247 | 0.098 | +0.0437 | 39/52 | 4e-05 |
| Bayesian-Threshold-BR | 0.108 | 0.280 | -0.0210 | 23/52 | 0.466 |
| Bayesian-PS | 0.216 | 0.308 | -0.0519 | 25/52 | 0.263 |
| Bayesian-Walk-PS | 0.347 | 0.128 | -0.0060 | 26/52 | 0.722 |
| Bayesian-Threshold-PS | 0.216 | 0.308 | -0.0519 | 25/52 | 0.263 |

## updaters (50 participants, 1064 choices)

| Model | TVD vs Random | TVD vs Random Walk | ΔLL vs RW / choice | better than RW | Wilcoxon p |
|---|--:|--:|--:|--:|--:|
| Bayesian-BR | 0.190 | 0.361 | -0.0474 | 19/50 | 0.0876 |
| Bayesian-Walk-BR | 0.353 | 0.131 | +0.0768 | 39/50 | 7.95e-06 |
| Bayesian-Threshold-BR | 0.190 | 0.361 | -0.0474 | 19/50 | 0.0876 |
| Bayesian-PS | 0.139 | 0.244 | +0.0138 | 24/50 | 0.969 |
| Bayesian-Walk-PS | 0.343 | 0.038 | +0.0431 | 44/50 | 4.38e-11 |
| Bayesian-Threshold-PS | 0.202 | 0.237 | +0.0228 | 28/50 | 0.41 |
