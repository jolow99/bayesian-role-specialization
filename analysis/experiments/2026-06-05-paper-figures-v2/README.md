# 2026-06-05 Paper figures v2 (post-meeting revision)

Revised role-inference paper figures, built on the
`2026-05-12-current-export-metric-comparison` pipeline (same scope as the
05-25/05-28 experiments: 5 exports, clean teams, human-only — **204 team-rounds,
721 team-stages, 3,104 inference reports**) and the Stage-1 fitted inference
engine (τ_prior = 4.6385, ε = 0.0624, memory = `drift_prior_0.500`).

## Outputs

| Task | Script | Output |
|------|--------|--------|
| 0 | `task0_stage1_metric.py` | `stage1_metric.md` — what the Stage-1 objective actually is, fitted vs uniform baseline |
| 1 | `task1_inference_fit.py` | `figures/inference_fit.png` — model posterior binned by the human's guessed role (+ calibration scatter) |
| 2 | `task2_temporal_convergence.py` | `figures/temporal_convergence.png`, `temporal_convergence.md` — value-rank of played combo vs stage |
| 3 | `task34_qualitative.py` | `figures/qualitative_best_respond_v2.png` — MFT round, storyboard layout |
| 4 | `task34_qualitative.py` | `figures/qualitative_flip_flop_v2.png` (+ `_alt.png`) — TFF round, mirror highlights |
| 5 | `task5_example_search.py` | `example_candidates.md` — top symmetry-breaking success/failure candidates |
| 6 | `task6_latex_table.py` | `aggregate_table.tex` — 14-model × 4-metric LaTeX tabular |

Shared code: `common.py` (pipeline imports, observer-aware inference queries,
value-ranking, Apple-Color-Emoji rasterization), `qualitative_v2.py` (the
redesigned single-round renderer).

## How to run

```bash
cd analysis
uv run python experiments/2026-06-05-paper-figures-v2/task0_stage1_metric.py
uv run python experiments/2026-06-05-paper-figures-v2/task1_inference_fit.py
uv run python experiments/2026-06-05-paper-figures-v2/task2_temporal_convergence.py
uv run python experiments/2026-06-05-paper-figures-v2/task34_qualitative.py
uv run python experiments/2026-06-05-paper-figures-v2/task5_example_search.py
uv run python experiments/2026-06-05-paper-figures-v2/task6_latex_table.py
```

Each script is independent. Task 6 reads `2026-05-28-paper-figures/results.json`
(the agg_ll-objective fits); everything else recomputes from the 05-12 pipeline.
Emoji rendering requires macOS (`/System/Library/Fonts/Apple Color Emoji.ttc`);
without it the role blocks fall back to letter labels only.

## Key findings

**Task 0.** The Stage-1 objective (`pipeline.py:475-504`) is the **mean log
posterior-marginal probability of the role the human reported**, averaged over
all 3,104 reports — not the raw mean probability, and a "log-likelihood" only
under an unstated probability-matching readout. Fitted: **−0.894** (arithmetic
mean P = 0.477) vs uniform baseline **−1.099** (P = 1/3). There is **no β in
Stage 1** — the fitted triple is (τ_prior, ε, memory-drift δ); softmax β only
exists in Stage-2 behavioral models. Full details in `stage1_metric.md`.

**Task 1.** The diagonal dominates in all three bins: mean posterior on the
guessed role = 0.44 (F, n=1648), 0.50 (T, n=662), 0.53 (M, n=794) vs ~0.22–0.29
off-diagonal. Shared-evidence calibration r = 0.58.

**Task 2.** Teams play well above chance from stage 1 (mean rank ≈ 9.3 vs 14)
but there is **no improvement across stages** — flat-to-slightly-worse in both
the all-rounds series and a fixed ≥4-stage cohort (which controls for winners
exiting early). Convergence-to-top-K is not the story on this metric; the
above-chance level is there from the first choice.

**Task 5.** The current fig-3 round (`…2F8H1E` r2) is the only round that locks
onto the value-optimal combo for 3 straight stages — kept. The current fig-4
round (`…2R4QJ6` r5) mirrors once (three-medic stage) then converges and wins;
the strongest alternative (`…RBRN0Z` r3, rendered as `_alt`) mirrors in
lock-step for four stages but TIMEOUTs without converging. Candidates listed in
`example_candidates.md`.
