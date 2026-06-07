# 2026-06-07 Instrumental rationality — are people choosing roles to maximize subjective expected utility?

Three PNAS-style single-column figures (3.42 in wide, no titles, no panel
letters — assembly in LaTeX) plus the aggregate model-comparison table,
parallel to the epistemic-rationality experiment
(`2026-06-07-epistemic-rationality`). Scope as 05-25/05-28/06-05: 5 exports,
clean teams, human rounds only — **204 team-rounds, 721 team-stage
observations** across 15 environments.

**Provenance.** The model-free rank statistics are recomputed here from
`compute_rank_rows()` in `2026-06-05-paper-figures-v2/common.py` (one row per
team-stage: value-rank of the played joint combo among all 27 by the
precomputed value matrices, eap-weighted, section2 conventions), on the
records from the `2026-05-12-current-export-metric-comparison` pipeline that
common.py imports — same 5 exports, same clean-teams filter. The model-based
artifacts (individual posteriors, aggregate table) are copied from
`2026-05-28-paper-figures/results.json` with **no model re-runs**; those
Bayesian fits use the 05-25 full pipeline's agg_ll objective — the
Stage-1/Stage-2 source-of-truth chain. `MODEL_ORDER`/`MODEL_COLORS` are read
from `2026-05-28-paper-figures/paper_figures.py` via AST (a plain import
would collide on the module name `pipeline`: paper_figures imports the 05-25
pipeline while common.py imports the 05-12 one).

**Recomputed vs copied.** Recomputed: all rank statistics (per-game means,
trend slope, Top-K/Bottom-K curves) and all CIs — percentile cluster
bootstraps over team-rounds, 10,000 resamples. Copied: the per-participant
log-likelihoods (renormalized into 13-model posteriors, see below) and the
aggregate-table values. Reproduction checks (all pass, printed by the
script): overall mean rank **9.71**, Top-1 **11.7%**, Top-5 **42.4%**,
Bottom-5 **8.3%** — identical to 05-28's `topk_summary.md`; recomputed
dominant-model counts match
`results.json["individual"]["dominant_counts"]`; `aggregate_table.tex` is
diff-identical to `2026-06-05-paper-figures-v2/aggregate_table.tex` (modulo
the auto-generated header comment).

**Mixture-PS: kept in the table, excluded from the individual fitting.**
Its fitted mixture weight under the agg_ll objective is **w = 1.0** — the
model collapses exactly onto its Bayesian Walk-PS component (aggregate
metrics and all 102 individual posteriors identical to Walk-PS to machine
precision), because the Thresh-PS component is near-deterministic
(fitted δ = 0) and assigns ~0 probability to switches humans actually made
(its own agg_ll is −7.42 vs Walk-PS's −2.51), so any mixture weight on it
hurts likelihood. The **table keeps the row** so the paper can show the
mixture result directly (its values being identical to Walk-PS *is* the
finding). The **individual fitting drops it**: a uniform prior over models
containing an exact duplicate double-counts Walk-PS in the model-posterior
normalizer, halving the Walk-PS family's apparent share and deflating every
other model's. Dropping it changes **0/102** dominant-model assignments
(asserted by the script); the 13-model posteriors are renormalized from the
stored log-likelihoods and asserted equal (to 1e-9) to dropping Mixture-PS
from the stored 14-model posteriors and renormalizing.

Outputs (`.png` 300 dpi + `.pdf` each) are in `stuff to incorporate/`.

## Figure 1 — value-rank by game number (`rank_by_game`, model-free, new)

Mean value-rank of the played combo (1 = best of 27, chance = 14, y-axis
inverted so up = better) by **game number** (round 1–8 of the session; each
participant has a unique human/bot round ordering, so each game number
samples a different subset of teams, 22–29 clean team-rounds each).

| Game | n stages | n team-rounds | Mean rank | 95% CI |
|---|--:|--:|--:|---|
| 1 | 94 | 27 | 9.61 | [7.71, 11.61] |
| 2 | 99 | 26 | 9.61 | [7.68, 11.69] |
| 3 | 98 | 28 | 10.37 | [8.44, 12.34] |
| 4 | 76 | 22 | 10.75 | [8.17, 13.38] |
| 5 | 81 | 24 | 7.83 | [5.98, 9.73] |
| 6 | 105 | 29 | 9.94 | [7.86, 12.09] |
| 7 | 75 | 23 | 8.99 | [6.85, 11.23] |
| 8 | 93 | 25 | 10.35 | [8.07, 12.73] |

**Trend (for the caption, not in-figure):** OLS slope **−0.011 rank/game,
95% CI [−0.342, +0.317]** — **no significant improvement across games**,
consistent with the by-stage result (06-05 task2, flat) and with the
epistemic experiment's flat accuracy-by-game. Teams sit well above chance
(rank 14) from game 1: good coordination is mostly present from the start
(stat-driven priors), it does not detectably sharpen with session experience.

## Figure 2 — Top-K / Bottom-K curves (`topk_curves`, model-free, restyle)

Cumulative fraction of team-stages whose played combo is in the value-best K
(Top-K) / value-worst K (Bottom-K), vs the uniform K/27 diagonal. Restyled
from `2026-05-28-paper-figures/section2_best_response.py` with
cluster-bootstrap 95% CI bands added; Top-1/Top-5 annotations kept.

| K | Top-K | 95% CI | Bottom-K | 95% CI | Uniform |
|--:|--:|---|--:|---|--:|
| 1 | 11.7% | [9.1%, 14.4%] | 1.8% | [0.8%, 2.9%] | 3.7% |
| 3 | 26.4% | [22.5%, 30.5%] | 5.0% | [3.3%, 6.9%] | 11.1% |
| 5 | 42.4% | [37.9%, 47.2%] | 8.3% | [6.1%, 10.7%] | 18.5% |
| 7 | 51.7% | [47.0%, 56.5%] | 13.6% | [10.5%, 16.8%] | 25.9% |
| 10 | 62.1% | [57.5%, 66.8%] | 20.9% | [17.2%, 24.9%] | 37.0% |
| 14 | 71.7% | [67.4%, 75.9%] | 30.2% | [25.9%, 34.6%] | 51.9% |

The Top-K curve sits above the diagonal and the Bottom-K curve below it at
every K, with non-overlapping CI bands through most of the range — humans
land on the exact-best combo at 3.1× the random rate and avoid the worst
combos at 0.4× the random rate.

## Figure 3 — individual model fitting (`individual_fitting`, model-based)

Per-participant posterior over 13 models (6 Bayesian + 7 baselines;
Mixture-PS excluded, see above), stacked bar only (no pie), participants
sorted by dominant model then by dominant weight. Posteriors renormalized
from the per-participant log-likelihoods in
`2026-05-28-paper-figures/results.json` (lapse = 0.05). Legend is grouped
into two titled blocks — **Bayesian models** (6) and **Non-Bayesian
baselines** (7) — placed side by side under the bars.

Dominant-model counts (for the caption; identical with or without
Mixture-PS): **Top-7: 20, Bayesian Walk: 15, Random: 12, Optimal: 9,
Bayesian Thresh-PS: 8, Contradict Others: 8, Bayesian-Belief: 7,
Bayesian-Value: 7, Bayesian Walk-PS: 6, Random Walk: 5,
Random-to-Optimal: 5** (Bayesian Threshold: 0). A Bayesian model is dominant
for **43/102** participants; the modal single model is the Top-7 heuristic
(play one of the seven value-best combos).

## Table — `aggregate_table.tex`

The full 14-model aggregate comparison (combo-r, marg-r, aggregate-LL,
mean-LL on the 204 team-rounds; the Mixture-PS row is kept here — identical
to Bayesian Walk-PS because its fitted weight is w = 1.0, see above),
regenerated from `2026-05-28-paper-figures/results.json` into
`stuff to incorporate/` so this experiment is self-contained; ported from
`2026-06-05-paper-figures-v2/task6_latex_table.py` and diff-checked
identical. Caption is inside the `.tex`.

## Draft LaTeX captions

Model-free pair (A: `topk_curves`, B: `rank_by_game`):

```latex
\caption{\textbf{Teams choose high-value role combinations far more often
than chance, from the first game onward.}
(\emph{A}) Cumulative fraction of team-stages ($n = 721$, from 204 clean
human team-rounds) whose played joint role combination ranks in the
value-best $K$ (Top-$K$, black) or value-worst $K$ (Bottom-$K$, purple) of
the 27 possible combinations, by the environment's precomputed expected
values. Teams play the single best combination in 11.7\% of team-stages
(uniform random: 3.7\%) and a top-5 combination in 42.4\% (random: 18.5\%),
while landing in the bottom 5 only 8.3\% of the time (random: 18.5\%);
grey dashed line, uniform random ($K/27$).
(\emph{B}) Mean value-rank of the played combination (1 = best of 27;
chance = 14, dotted line; $n$ = team-stages per game) by game number within
the session. Mean rank is 9.71 overall, better than chance in every game,
with no significant change across games (slope $-0.011$ rank/game, 95\% CI
$[-0.342, +0.317]$). Shaded bands and error bars are 95\% percentile cluster
bootstraps over team-rounds (10{,}000 resamples).}
```

`individual_fitting`:

```latex
\caption{\textbf{Individual-level model comparison.} Posterior probability
over 13 candidate models (6 Bayesian, 7 non-Bayesian baselines; uniform
model prior, lapse rate 0.05) given each participant's role choices across
their human rounds ($n = 102$ participants; one bar each, sorted by
best-fitting model). A Bayesian model best explains 43/102 participants;
the modal single model is the Top-7 heuristic (choosing among the seven
highest-value combinations; 20 participants), followed by Bayesian Walk
(15) and Random (12).}
```

## Outputs

| File | Contents |
|------|----------|
| `stuff to incorporate/rank_by_game.{png,pdf}` | Mean value-rank by game number (1–8), 95% CIs, chance line, per-game n |
| `stuff to incorporate/topk_curves.{png,pdf}` | Top-K / Bottom-K cumulative curves with 95% CI bands, uniform diagonal, Top-1/Top-5 annotations |
| `stuff to incorporate/individual_fitting.{png,pdf}` | Per-participant posterior over 13 models (Mixture-PS excluded), stacked bar, grouped legend, sorted by dominant model |
| `stuff to incorporate/aggregate_table.tex` | 14-model aggregate comparison table (LaTeX, caption included; Mixture-PS row kept) |
| `summary.md` | All numbers: reproduction checks, per-game table, Top-K/Bottom-K table, dominant counts |

## How to run

```bash
cd analysis
uv run python experiments/2026-06-07-instrumental-rationality/instrumental_rationality.py
```

One script produces all three figures (png + pdf), the table, and
`summary.md`, and asserts the reproduction checks listed above.
