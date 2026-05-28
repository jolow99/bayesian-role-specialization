# 2026-05-28 Paper figures

Outputs for the paper's three results sections, all on the 2026-05-25 full-pipeline data scope (5 exports, clean games):

| Section | Quantitative | Qualitative |
|---------|--------------|-------------|
| **1. How well does the model explain human team behavior?** | `aggregate_table.md` + `figures/individual_fitting.png` | — |
| **2. Do teams (or individuals) best-respond?** | `topk_summary.md` + `figures/topk_curves.png` | `figures/qualitative_best_respond.png` |
| **3. Can the model explain human adaptation to stubborn teammates?** | `bot_adaptation_summary.md` + `figures/bot_adaptation_overview.png` | `figures/qualitative_flip_flop.png` |

Human data: **204 team-rounds, 15 envs, 102 participants** (all human rounds in 5 exports, dropout games excluded).
Bot data:   **204 player-rounds, 15 treatments, 102 participants** (all bot rounds in 5 exports, clean games).

## What this experiment runs

14 models, all evaluated on the same 204 team-rounds:

| Group | Models |
|-------|--------|
| Bayesian (from 05-25 pipeline) | Bayesian Walk, Bayesian Walk-PS, Mixture-PS, Bayesian-Belief, Bayesian-Value, Bayesian Threshold, Bayesian Thresh-PS |
| Baselines (from 03-30 summary) | Random, Top-7, Random-to-Optimal, Copy Others, Contradict Others, Random Walk, Optimal |

**Parameters.** Bayesian model params come from the 05-25 pipeline's `agg_ll`-objective fit (the proper scoring rule, per that experiment's README). Random Walk's ε is grid-searched here on the same objective. The other baselines are parameter-free.

## Outputs

| File | Contents |
|------|----------|
| `aggregate_table.md` | Markdown table — `combo_r`, `marg_r`, `agg_ll`, `mean_ll` per model, sorted by `combo_r`. The paper's aggregate sub-section uses the `combo_r` column. |
| `results.json` | Same numbers plus fitted params, per-participant log-likelihoods, posteriors over models, and dominant-model counts. |
| `figures/individual_fitting.png` | Pie chart (best-fitting model per participant) + stacked bar (posterior over models per participant). Mirrors `2026-03-30-summary-of-all-models/individual_fitting.png` but across all 14 models on the new data scope. |

## How to run

```bash
cd analysis
uv run python experiments/2026-05-28-paper-figures/paper_figures.py            # Section 1
uv run python experiments/2026-05-28-paper-figures/section2_best_response.py   # Section 2
uv run python experiments/2026-05-28-paper-figures/section3_bot_adaptation.py  # Section 3
```

Each script is independent. All three depend on `2026-05-25-full-pipeline/` for shared data-loading code, the stage-1 inference params, and (for Section 1) the fitted Bayesian-model parameters in its `results.json`. The Section 2 qualitative figure also reuses `2026-05-24_single-round-static/build_static.py` for rendering. If the upstream pipeline outputs change, re-run that pipeline first.

## Section 2 — quantitative summary

Top-K curves and per-stage rank progression on **721 team-stage observations**. Headline:

- Mean rank: **9.71** (random 14.0; one-sided z = -14.78, p < 1e-4)
- Top-1: **11.7%** (random 3.7%, 3.1× rate)
- Top-5: **42.4%** (random 18.5%)
- Bottom-5: **8.3%** (random 18.5%, 0.4× rate)
- Mean normalized optimality: **0.667** (random 0.500)

Reading: humans plan top-K — both attracted to good combos and repelled from bad ones.

## Section 3 — quantitative summary

Per-participant behavior types in bot rounds across 15 treatments (102 participants):

| Type | Criterion | N | % |
|------|-----------|--:|--:|
| Stat-adherent | stat_rate ≥ 0.70 | 46 | 45% |
| Mixed/Explorer | neither threshold met | 36 | 35% |
| Deviator | dev_rate ≥ 0.50 | 20 | 20% |

Aggregate: 60% stat-optimal play, 26% deviate-optimal. Some humans clearly adapt; others refuse to deviate.

## Outputs

| File | Section | Contents |
|------|---------|----------|
| `aggregate_table.md` | 1 | 14-model metric table (combo_r, marg_r, agg_ll, mean_ll) |
| `figures/individual_fitting.png` | 1 | Pie + stacked bar of dominant model per participant |
| `results.json` | 1 | Aggregate metrics + individual posteriors + dominant counts |
| `topk_summary.md` | 2 | Top-K headline numbers + per-stage breakdown |
| `figures/topk_curves.png` | 2 | Top-K / Bottom-K curves (overall) |
| `figures/qualitative_best_respond.png` | 2 | Single team-round converging from rank 24 → 1 (03-24 styling, no model panel) |
| `bot_adaptation_summary.md` | 3 | Headline rates + behavior-type table + per-treatment breakdown |
| `figures/bot_adaptation_overview.png` | 3 | Per-treatment role bars + behavior-type pie |
| `figures/qualitative_flip_flop.png` | 3 | Single **human** team-round (game 2R4QJ6 r5, env 141_222_222_TFF) flipping FMF → MMM → MTF → TFM → TFM — team converges to TFM in the last two stages after 7 total role switches across players. 03-24 styling, no model panel. |

## Individual fitting — method

For each of 14 models, compute a per-stage per-player role marginal (3-vector), apply a 5% lapse rate, and accumulate log-likelihoods across all of a participant's stages. Softmax over the 14 LLs gives that participant's posterior over models. The dominant model is `argmax` of that posterior.

## Top-K — method (Section 2)

For every clean human team-stage observation, compute the value-rank of the chosen 3-role combo against all 27 combos at the start-of-stage team/enemy HP. Expected value is computed as `(1 - eap) * V[combo, intent=0, thp, ehp] + eap * V[combo, intent=1, thp, ehp]` with `eap = sum(lds) / len(lds)` (per-round actual attack rate). The Top-K curve is the cumulative fraction of plays at rank ≤ K; Bottom-K is the cumulative fraction at rank > 27-K. Both are compared against the uniform-random diagonal K/27.

## Bot adaptation — method (Section 3)

Bot rounds are loaded from raw `PlayerRound`s via `shared.data_loading.load_all_exports`, then filtered to clean games (no dropouts). Each record uses `build_bot_round_layout` to resolve the human's in-game position and bot role assignments per CLAUDE.md's "Bot Round Ground Truth" (do not trust `config.humanRole`; use `pr.player_id` and the in-game-position permutation). Stat-optimal and deviate-optimal roles for the human come from `parse_stat_optimal_roles` / `parse_deviate_roles` (position 0 = human in those parsers).
