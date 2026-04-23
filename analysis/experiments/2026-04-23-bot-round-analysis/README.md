# 2026-04-23 Bot-Round Analysis

Behavioral-only analysis of bot rounds in the 04-23 export, with the
03-18 export as a sanity reference.

In bot rounds the human plays alongside 2 fixed-strategy AI bots that play
their deviate-optimal roles. The human's stat-suggested role differs from
the deviate-optimal slot they need to fill — the question is whether
humans (a) infer what the bots are doing and (b) deviate from their natural
stat role to fill the missing slot.

## How to run

```bash
cd analysis
uv run python experiments/2026-04-23-bot-round-analysis/analyze.py
```

Outputs:

- `comparison_table.md` — per-stage tables for outcomes, deviation,
  inference accuracy, both exports side by side.
- `figures/deviation_curve.png` — per-stage breakdown of deviate-opt vs
  stat-opt vs other for both exports.
- `figures/inference_curve.png` — per-stage human → bot inference
  accuracy with chance baseline.

## Headline results (04-23 vs 03-18)

| Metric | 04-23 | 03-18 |
|---|---|---|
| Bot-round WIN rate | **44%** | 26% |
| Ever deviated to deviate-optimal | **58%** | 37% |
| Deviated on final stage | **34%** | 17% |
| Stage-2 inference accuracy (human → bot) | **72%** | 66% |
| Stage-5 inference accuracy | 67% | 70% |

Players in 04-23 are noticeably better at the harder bot rounds —
deviating more, winning more, and inferring bot roles slightly more
accurately. Per-stage curves show a learning trajectory in both exports
(deviate-opt rises stage-over-stage, stat-opt stickiness falls), with
04-23 starting from a higher baseline.

## Notes on ground truth

This script implements the "Bot Round Ground Truth" mapping from
`CLAUDE.md`:

- Human in-game position = `pr.player_id` (NOT `config.humanRole`).
- `botPlayers[0]` → lower non-human position; `botPlayers[1]` → higher.
- Bot roles never change within a round — `bot_role_map` is computed
  once and used to score every inference.
- Inferences made at stage N are about stage N-1; bots never switch, so
  the comparison target is constant across stages.

The analysis is human-only at the metric level (we score the human's
choices and the human's inferences about the bots).
