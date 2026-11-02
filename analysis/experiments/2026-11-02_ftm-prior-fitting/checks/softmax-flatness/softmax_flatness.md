# Softmax flatness at the fitted τ_v

Per choice: value spread = best − worst expected value over the player's 3 roles; softmax at the group's fitted τ_v. Median [IQR] over the group's choices. Uniform: P(top) = 0.33, TVD = 0; deterministic: P(top) = 1, TVD = 0.67.

## updaters (1064 choices): value spread 7.3 [3.2, 13.5]

| Model | τ_v | spread / τ_v | P(worst) / P(best) | P(top role) | TVD from uniform |
|---|--:|--:|--:|--:|--:|
| Bayesian-BR | 9.48 | 0.77 [0.34, 1.43] | 0.46 [0.24, 0.71] | 0.46 [0.38, 0.60] | 0.16 [0.06, 0.27] |
| Bayesian-Walk-BR | 8.54 | 0.85 [0.37, 1.58] | 0.43 [0.21, 0.69] | 0.48 [0.39, 0.63] | 0.18 [0.07, 0.30] |
| Bayesian-Threshold-BR | 9.48 | 0.77 [0.34, 1.43] | 0.46 [0.24, 0.71] | 0.46 [0.38, 0.60] | 0.16 [0.06, 0.27] |

## non-updaters (1099 choices): value spread 7.4 [3.3, 12.6]

| Model | τ_v | spread / τ_v | P(worst) / P(best) | P(top role) | TVD from uniform |
|---|--:|--:|--:|--:|--:|
| Bayesian-BR | 16.97 | 0.44 [0.19, 0.74] | 0.65 [0.48, 0.82] | 0.41 [0.37, 0.46] | 0.09 [0.04, 0.15] |
| Bayesian-Walk-BR | 16.51 | 0.45 [0.20, 0.76] | 0.64 [0.47, 0.82] | 0.41 [0.37, 0.47] | 0.09 [0.04, 0.15] |
| Bayesian-Threshold-BR | 17.98 | 0.41 [0.18, 0.70] | 0.66 [0.50, 0.83] | 0.41 [0.37, 0.46] | 0.08 [0.04, 0.14] |
