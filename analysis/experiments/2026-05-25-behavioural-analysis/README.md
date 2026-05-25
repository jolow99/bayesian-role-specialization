# 2026-05-25 Behavioural analysis

Behavioural-data figures for the PNAS paper:

| Paper fig | File | What it shows |
|-----------|------|---------------|
| Fig 2 | `figures/fig_outcome_win_rates.png` | Per-batch (per-export) win rates, human rounds vs bot rounds |
| Fig 3 | `figures/fig_human_role_behavior.png` | Per-stage stat-optimal-role rate and role-switching rate (human rounds) |
| Fig 6 | `figures/fig_bot_role_choice.png` | Per-stage deviate-optimal and stat-optimal rate in bot rounds |
| Fig 7 | `figures/fig_inference_accuracy.png` | Per-stage inference accuracy, human vs bot rounds |
| Fig 5 | `figures/fig_qualitative_by_env.png` | Per-env per-stage F/T/M role distribution: humans vs Bayesian-Walk vs Bayesian-Belief |

## Scope

- All five exports in `analysis/data/exports/`.
- Excludes games with any dropout player-round, matching the 5-export
  full-pipeline experiment so the numbers line up across figures.
- Bot rounds use the position-layout helper from CLAUDE.md
  (`pr.player_id`, `botPlayers[i].strategy.role` as int) — same gotchas as
  Stage 1 of the full pipeline.

## How to run

```bash
cd analysis/experiments/2026-05-25-behavioural-analysis

# Figs 2/3/6/7 — pure behavioural analysis
python make_figures.py

# Fig 5 — needs the full-pipeline Stage-2 fits
python qualitative_by_env.py
```

`qualitative_by_env.py` reads `../2026-05-25-full-pipeline/results.json`
for the Bayesian-Walk fit (under the `combo_r` objective, matching the
paper text), recomputes posteriors with Stage-1 params, and overlays the
Walk and Belief model marginals against the human empirical distribution.
