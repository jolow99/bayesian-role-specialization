# R1 — Inference-accuracy numbers for paper prose

No figure (the accuracy_by_game panel is removed; report in prose). Human rounds: 3,104 reports from 203 clean team-rounds (5 exports). Bot rounds: 1,296 human reports from 101 (game, participant) clusters. Stage-1 params: tau_prior = 4.6385, epsilon = 0.0624, memory = `drift_prior_0.500`. All CIs are percentile cluster bootstraps with 10,000 resamples; human rounds clustered by team-round, bot rounds by (game_id, participant_id) — (game_id, round_number) is not unique for bot rounds. Chance accuracy = 1/3 = 0.333.

## Numbers to quote

- **Overall human inference accuracy** (vs true previous-stage role): **0.629** (95% CI [0.605, 0.653]); chance = 0.333.
- **Bayesian observer, sampling-readout accuracy** (mean posterior mass on the true role): **0.612** (95% CI [0.603, 0.621]). **Paired human - sampling difference**: **+0.017** (95% CI [-0.006, +0.041]).
- **Posterior-mode (MAP) accuracy** (ceiling reference): **0.920** (95% CI [0.901, 0.937]).
- **Learning slope of human accuracy across game number (1-8)**: **+0.0086** per game (95% CI [-0.0015, +0.0187]).
- **Bot-round human inference accuracy**: **0.579** (95% CI [0.528, 0.632]). **Human-round minus bot-round difference**: **+0.050** (95% CI [-0.007, +0.107]) — the CI straddles 0, confirming bot-round inference accuracy is statistically indistinguishable from human-round accuracy.

## Summary table

| Quantity | Value | 95% CI |
|----------|------:|--------|
| Human accuracy (human rounds) | 0.629 | [0.605, 0.653] |
| Bayesian sampling readout | 0.612 | [0.603, 0.621] |
| Human - sampling (paired) | +0.017 | [-0.006, +0.041] |
| MAP (mode) accuracy / ceiling | 0.920 | [0.901, 0.937] |
| Human learning slope (per game) | +0.0086 | [-0.0015, +0.0187] |
| Human accuracy (bot rounds) | 0.579 | [0.528, 0.632] |
| Human-round - bot-round difference | +0.050 | [-0.007, +0.107] |

