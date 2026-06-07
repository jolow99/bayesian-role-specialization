# Epistemic rationality — human vs Bayesian-model inferences

Scope: 203 clean human team-rounds (5 exports), 3,104 inference reports. Stage-1 params: τ_prior = 4.6385, ε = 0.0624, memory = `drift_prior_0.500`. All CIs are percentile cluster bootstraps over team-rounds (10,000 resamples).

## Figure 1 — inference accuracy by game number (vs true previous role)

Overall: human **0.629**, Bayesian sampling readout **0.612** [0.602, 0.621], MAP readout **0.920**, chance 1/3. Human − sampling paired difference: +0.017 [-0.006, +0.041]. Game number = round 1-8 of the session; each participant has a unique human/bot round ordering, so each game number samples a different subset of teams.

| Game | n reports | Human acc | 95% CI | Sampling acc | 95% CI | MAP acc | 95% CI |
|---|--:|--:|---|--:|---|--:|---|
| 1 | 400 | 0.618 | [0.562, 0.673] | 0.608 | [0.588, 0.628] | 0.920 | [0.877, 0.956] |
| 2 | 426 | 0.566 | [0.500, 0.635] | 0.597 | [0.576, 0.616] | 0.927 | [0.881, 0.966] |
| 3 | 420 | 0.617 | [0.556, 0.677] | 0.623 | [0.603, 0.643] | 0.905 | [0.840, 0.956] |
| 4 | 328 | 0.649 | [0.570, 0.732] | 0.630 | [0.603, 0.656] | 0.948 | [0.915, 0.980] |
| 5 | 340 | 0.665 | [0.583, 0.746] | 0.626 | [0.601, 0.651] | 0.953 | [0.921, 0.983] |
| 6 | 458 | 0.646 | [0.594, 0.699] | 0.597 | [0.563, 0.628] | 0.873 | [0.801, 0.936] |
| 7 | 324 | 0.627 | [0.561, 0.692] | 0.596 | [0.561, 0.627] | 0.886 | [0.819, 0.942] |
| 8 | 408 | 0.657 | [0.583, 0.725] | 0.623 | [0.592, 0.652] | 0.956 | [0.925, 0.984] |

Learning trend (OLS slope, accuracy per game): human **+0.0086** [-0.0015, +0.0187], MAP readout -0.0003 [-0.0073, +0.0068].

## Figure 2 — calibration of human reports against the posterior

Every report × each role: x = posterior probability of the role, y = 1 if the human reported it.

| Metric | Value |
|--------|-------|
| Pearson r (raw pairs) | **0.463** [0.427, 0.498] (9,312 pairs) |
| Pearson r (binned means) | **0.995** |
| Pearson r — Fighter only (raw pairs) | 0.444 [0.407, 0.482] |
| Pearson r — Tank only (raw pairs) | 0.461 [0.414, 0.508] |
| Pearson r — Medic only (raw pairs) | 0.500 [0.456, 0.544] |

| Posterior bin | n | mean x | Report frequency | 95% CI |
|---------------|--:|-------:|-----------------:|--------|
| 0.1–0.2 | 4887 | 0.159 | 0.156 | [0.144, 0.168] |
| 0.2–0.3 | 964 | 0.261 | 0.273 | [0.239, 0.307] |
| 0.3–0.4 | 217 | 0.365 | 0.419 | [0.362, 0.477] |
| 0.4–0.5 | 625 | 0.433 | 0.434 | [0.395, 0.471] |
| 0.5–0.6 | 329 | 0.573 | 0.593 | [0.528, 0.656] |
| 0.6–0.7 | 1642 | 0.646 | 0.632 | [0.600, 0.663] |
| 0.7–0.8 | 648 | 0.738 | 0.745 | [0.706, 0.784] |
