# 2026-06-07 Epistemic rationality — are people updating beliefs in a Bayesian way?

Two PNAS-style single-column figures (3.42 in wide, no titles, no panel
letters — assembly in LaTeX) comparing **human role inferences** against the
**fitted Stage-1 Bayesian observer** on the same evidence. Scope as
05-25/05-28/06-05: 5 exports, clean teams, human rounds only — 204 team-rounds,
3,104 inference reports (203 team-rounds contribute at least one report).
Stage-1 params: τ_prior = 4.6385, ε = 0.0624, memory = `drift_prior_0.500`,
read from **`2026-05-25-full-pipeline/stage1_inference/best_inference_params.json`**
— the Stage-1 source of truth that 05-28-paper-figures also consumes (re-fit
from scratch on the 5-export scope). The 05-12 pipeline's fit, which 06-05's
`common.py` loads for its posterior machinery, is byte-identical (same 5
exports, same objective, same optimum); the script asserts the two agree at
load time so any future re-fit fails loudly instead of silently diverging.

All CIs are **percentile cluster bootstraps over team-rounds** (10,000
resamples) — reports within a team-round share evidence and observers, so
resampling individual reports would understate uncertainty.

Outputs (`.png` 300 dpi + `.pdf` each) are in `stuff to incorporate/`.

## Figure 1 — correctness (`accuracy_by_game`)

Accuracy of each inference report against the teammate's **true
previous-stage role**, by **game number** (round 1–8 of the session; each
participant has a unique human/bot round ordering, so each game number samples
a different subset of teams, 22–29 clean team-rounds each). Three lines:

- **human reports** (black) — the data;
- **Bayesian observer, sampling readout** (purple) — mean posterior mass on
  the true role = expected accuracy of an agent that samples its report from
  the posterior. This is the comparison that matters, since Figure 2 shows
  humans probability-match the posterior;
- **Bayesian observer, MAP readout** (grey dashed, thin) — posterior-mode hit
  rate, a ceiling reference.

| Game | n | Human | Sampling readout | MAP readout |
|---|--:|--:|--:|--:|
| 1 | 400 | 0.618 [0.562, 0.673] | 0.608 [0.588, 0.628] | 0.920 [0.877, 0.956] |
| 2 | 426 | 0.566 [0.500, 0.635] | 0.597 [0.576, 0.616] | 0.927 [0.881, 0.966] |
| 3 | 420 | 0.617 [0.556, 0.677] | 0.623 [0.603, 0.643] | 0.905 [0.840, 0.956] |
| 4 | 328 | 0.649 [0.570, 0.732] | 0.630 [0.603, 0.656] | 0.948 [0.915, 0.980] |
| 5 | 340 | 0.665 [0.583, 0.746] | 0.626 [0.601, 0.651] | 0.953 [0.921, 0.983] |
| 6 | 458 | 0.646 [0.594, 0.699] | 0.597 [0.563, 0.628] | 0.873 [0.801, 0.936] |
| 7 | 324 | 0.627 [0.561, 0.692] | 0.596 [0.561, 0.627] | 0.886 [0.819, 0.942] |
| 8 | 408 | 0.657 [0.583, 0.725] | 0.623 [0.592, 0.652] | 0.956 [0.925, 0.984] |

**Headline:** overall human accuracy **0.629** vs sampling readout **0.612,
95% CI [0.602, 0.621]**; the paired human − sampling difference is **+0.017,
95% CI [−0.006, +0.041]** — **statistically indistinguishable**, and every
per-game human CI contains the per-game sampling-readout mean. Humans perform
exactly as well as a Bayesian who probability-matches, well below the MAP
ceiling (0.920).

Learning trend (for the caption, not in-figure): human slope **+0.009/game,
95% CI [−0.002, +0.019]** — no significant learning effect; MAP-readout slope
−0.000 [−0.007, +0.007], so the evidence difficulty is flat across game
numbers too. Humans read teammates well above chance (1/3) from game 1.

## Figure 2 — correlation / calibration (`calibration`)

Every report crossed with each of the 3 roles (9,312 pairs); x = the Stage-1
posterior probability of that role from the reporting player's evidence,
y = 1 if the human reported it. Decile-binned mean(y) at the bin's mean x with
95% CIs, identity line, bin-occupancy histogram with per-bar counts.

| Metric | Value |
|--------|-------|
| Pearson r (raw 9,312 pairs) | **0.463**, 95% CI **[0.427, 0.498]** |
| Pearson r (binned means) | **0.995** |
| Per-role r (raw pairs) | F 0.444 [0.407, 0.482] · T 0.461 [0.414, 0.508] · M 0.500 [0.456, 0.544] |

The binned curve sits **on the identity line** — not steeper — i.e. humans
**probability-match** the Bayesian posterior rather than greedily reporting
its mode. The raw-pair r is attenuated by y being binary; the binned
calibration is the cleaner statement of Bayesian consistency. The fitted
`drift_prior_0.500` memory keeps the posterior inside ~[0.1, 0.8], so the
curve spans 7 of the 10 deciles (occupancy shown in the histogram).

## 2026-06-07 revision — what changed and what didn't

Presentation revision + one added comparison; **no previously reported
statistic changed**: human per-game accuracies, MAP per-game accuracies, both
trend slopes, all calibration rs (raw, binned, per-role) and per-bin values
are identical to the prior version. New statistics (fresh bootstrap seeds):
the sampling-readout accuracies (overall and per-game) and the paired
human − sampling difference. Style: PNAS single column (3.42 in), sans-serif,
all text ≥ 6 pt at print size, titles removed, chance line removed from the
calibration panel (identity is the reference there), histogram bars labeled
with counts, trend annotation moved here for the caption.

## Draft LaTeX caption (combined two-panel figure)

```latex
\caption{\textbf{Humans infer teammates' roles as accurately as a Bayesian
observer that probability-matches its posterior, and their reports are
calibrated to that posterior.}
(\emph{A}) Accuracy of role inferences against the teammate's true
previous-stage role, by game number within the session (8 games per
participant, unique human/bot orderings; $n$ = reports per game). Human
accuracy (black; overall 0.629) is statistically indistinguishable from the
expected accuracy of sampling from the fitted Bayesian observer's posterior
(purple; 0.612, 95\% CI [0.602, 0.621]; paired difference $+0.017$, 95\% CI
$[-0.006, +0.041]$) and shows no significant change across games (slope
$+0.009$/game, 95\% CI $[-0.002, +0.019]$). The posterior-mode (MAP) readout
(grey dashed; 0.920) marks the accuracy ceiling of the available evidence;
dotted line, chance ($1/3$).
(\emph{B}) Calibration of human reports against the Bayesian posterior. Each
of the 3{,}104 reports is crossed with the three roles (9{,}312 pairs):
$x$ = the observer-specific posterior probability of the role, $y$ = whether
the human reported it. Decile-binned report frequencies (black, 95\% CIs;
histogram shows pairs per bin) lie on the identity line (grey dashed),
indicating probability matching rather than posterior-mode responding;
$r = 0.46$, 95\% CI $[0.43, 0.50]$ across raw pairs. Error bars and CIs are
percentile cluster bootstraps over team-rounds (10{,}000 resamples).}
```

## Outputs

| File | Contents |
|------|----------|
| `stuff to incorporate/accuracy_by_game.{png,pdf}` | Human vs sampling-readout vs MAP-readout accuracy by game (1–8), 95% CIs, chance line, per-game n |
| `stuff to incorporate/calibration.{png,pdf}` | Decile calibration vs the posterior, 95% CIs, identity line, labeled histogram, raw-pair r |
| `summary.md` | All numbers: per-game accuracy table (3 readouts), trends, both rs, per-role rs, per-bin calibration table |

## How to run

```bash
cd analysis
uv run python experiments/2026-06-07-epistemic-rationality/epistemic_rationality.py
```

One script produces both figures (png + pdf). Stage-1 params come from
`2026-05-25-full-pipeline/stage1_inference/best_inference_params.json`
(asserted equal to the 05-12 fit). Posterior/data machinery is reused from
`2026-06-05-paper-figures-v2/common.py` (which imports the
`2026-05-12-current-export-metric-comparison` pipeline — same 5 exports,
same inference engine from `shared/`).

Note (per `stage1_metric.md` in 06-05): the Stage-1 fit objective is the mean
log posterior probability assigned to participants' role reports — do not call
it a log-likelihood in the paper; there is no readout β in Stage 1.
