# R3 — Human-team case study (`R3_team_case`) — NEW

Human-round analogue of the bot-round adaptation storyboard
(`R3_adaptation_case`), built to illustrate the paper's claim that role
choice is a **mixture of rational value-seeking and sticky inertia**.

**Pinned case** (chosen via `case_search.py` from 143 WIN candidates, then
the `mixture_rank` sticks-then-relents filter): game
`01KRBT30X9RGSB5Q5P3PX48RYB`, round 4 — a **fully symmetric** team
(`222_222_222`). 4 logged stages, but the 4th is a spurious post-win
frame (boss already at 0 HP, one player's actions unlogged) and is
trimmed; the 3 live stages tell the whole story.

| Stage | combo | value-rank | what happens |
|---|---|---:|---|
| 1 | TMT | 26 / 27 | badly mis-coordinated; P1 & P3 both pick Tank (symmetric clash) |
| 2 | FMT | 12 / 27 | P1 best-responds immediately (→Fighter); **P2 stays Medic, P3 stays Tank though best-response says Fighter — stickiness** |
| 3 | FFF | 1 / 27 | **P2 & P3 relent → Fighter**; team hits the value-optimal combo and **WINS** |

Best-response = the role maximizing the eap-weighted expected **team**
value at stage-start HP under the fitted Bayesian observer's posterior
over the two teammates (posterior marginalized over the player's own
axis). Best-response disagrees with the played role **only** at stage 2
for P2 and P3 — the two stickiness events — so the analytic overlay is
clean: gold flags + dashed cards mark stickiness, green arrows mark the
relent. Posteriors use the 05-25 Stage-1 params (τ_prior = 4.6385,
ε = 0.0624, memory `drift_prior_0.500`), cross-checked against
common.py's 05-12 fit.

Panels (top→bottom): per-turn team/boss HP + per-stage value-rank;
one role track per human (vector Twemoji role icon + logged A/B/H per
turn + best-response/stickiness/relent overlay); one belief row per
target player (Bayesian observer posterior marginal per stage, with the
≤2 teammates' reports overlaid — caret under the named role, filled =
matches the target's true previous role); enemy-intent row; legend.

Role icons are **true-vector Twemoji** (`svg_icons.py` parses each
Twemoji SVG path + fill into matplotlib `PathPatch`es); the figure PDF
contains **0 raster image XObjects**. Icons also appear in the legend so
the role encoding does not rely on color alone. Compact full-width
(7 in) — single-column is infeasible with 3 role tracks + 3 belief rows.

---

# R1 — Calibration figure (revised headline)

Scope: 203 clean human team-rounds (5 exports), 3,104 inference reports (9,312 (report, role) pairs). Stage-1 params: tau_prior = 4.6385, epsilon = 0.0624, memory = `drift_prior_0.500`. All CIs are percentile cluster bootstraps over team-rounds (10,000 resamples).

## Headline statistic (the fix)

The figure now headlines the Pearson r over the **binned (decile) points** (mean x vs mean y), not the raw (report, role) pairs.

| Statistic | r |
|-----------|---|
| **Binned (decile) points** — headline | **0.995** |
| Raw (report, role) pairs | 0.463 [0.427, 0.498] |

## Robustness to bin count

| N_BINS | binned r | occupied bins |
|--:|--:|--:|
| 5 | 0.999 | 4 |
| 10 | 0.995 | 7 |
| 20 | 0.968 | 13 |

**Confirmation.** Over the 7 occupied decile bins the mean absolute deviation of report frequency from the posterior probability (|mean y - mean x|) is **0.016** — the binned points lie essentially on the identity line. The binned r is **0.99** (very high) while the raw-pair r is only **0.46**. The discrepancy is expected: a 0/1 report indicator has within-bin variance p(1-p) that caps the achievable per-pair Pearson r far below 1 even under perfect probability matching, so the raw-pair r understates the agreement the binned points display.

## Per-role raw-pair correlations (unchanged)

| Role | raw-pair r | 95% CI |
|------|--:|--|
| Fighter | 0.444 | [0.407, 0.482] |
| Tank | 0.461 | [0.414, 0.508] |
| Medic | 0.500 | [0.456, 0.544] |

## Per-bin detail (decile, main plotted binning)

| Posterior bin | n | mean x | Report frequency | 95% CI |
|---------------|--:|-------:|-----------------:|--------|
| 0.1-0.2 | 4887 | 0.159 | 0.156 | [0.144, 0.168] |
| 0.2-0.3 | 964 | 0.261 | 0.273 | [0.239, 0.307] |
| 0.3-0.4 | 217 | 0.365 | 0.419 | [0.362, 0.477] |
| 0.4-0.5 | 625 | 0.433 | 0.434 | [0.395, 0.471] |
| 0.5-0.6 | 329 | 0.573 | 0.593 | [0.528, 0.656] |
| 0.6-0.7 | 1642 | 0.646 | 0.632 | [0.600, 0.663] |
| 0.7-0.8 | 648 | 0.738 | 0.745 | [0.706, 0.784] |

# R2 — Individual model fitting figure (revised)

Posteriors over 13 models for 102 participants, renormalized from the per-participant log-likelihoods in `2026-05-28-paper-figures/results.json` (agg_ll-objective fits; lapse = 0.05; Stage-1: tau_prior = 4.6385, epsilon = 0.0624, memory = `drift_prior_0.500`). Mixture-PS excluded (collapses onto Bayesian Walk-PS; would double-count Walk-PS in the normalizer). The fitting logic is unchanged from `2026-06-07-instrumental-rationality`.

## Visual fixes

1. **Hatching.** Non-Bayesian baselines now carry distinct hatch patterns (Random Walk `///`, Top-7 `...`, Random-to-Optimal `xxx`, Optimal `\\\`, Copy Others `++`, Contradict Others `oo`, Random `--`); Bayesian models stay solid fills. The same hatch is shown on the legend swatches. Random Walk vs Bayesian Walk-PS and Copy Others vs Random are now separable.
2. **Negligible-baseline collapse.** A baseline is negligible iff its per-participant posterior is < 0.01 for **every** participant **and** it is dominant for 0 participants. Negligible baselines are lumped into a single thin grey "other baselines" segment. The normalizer is left intact (the bars still sum to 1 and remain an honest P(model | participant)); only the colour/hatch encoding of the lumped mass changes.

## Models collapsed into "other baselines"

| Model | max per-participant posterior | dominant for |
|-------|--:|--:|
| Copy Others | 0.0004 | 0 |

## Baselines kept as their own hatched segment

| Model | max per-participant posterior | dominant for |
|-------|--:|--:|
| Random Walk | 0.5873 | 5 |
| Top-7 | 0.9901 | 20 |
| Random-to-Optimal | 0.8628 | 5 |
| Optimal | 0.9846 | 9 |
| Contradict Others | 0.9945 | 8 |
| Random | 0.9473 | 12 |

## Dominant-model counts (recomputed, match stored `dominant_counts`)

| Model | n participants |
|-------|--:|
| Bayesian Walk | 15 |
| Bayesian Walk-PS | 6 |
| Bayesian-Belief | 7 |
| Bayesian-Value | 7 |
| Bayesian Thresh-PS | 8 |
| Random Walk | 5 |
| Top-7 | 20 |
| Random-to-Optimal | 5 |
| Optimal | 9 |
| Contradict Others | 8 |
| Random | 12 |

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
