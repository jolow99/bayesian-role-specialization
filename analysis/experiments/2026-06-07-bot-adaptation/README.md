# 2026-06-07 bot-adaptation — can humans adapt to suboptimal, stubborn, or symmetrical cases?

Paper-figure experiment for the bot-round section. In every bot round,
1 human plays with 2 fixed-strategy ("stubborn") bots that play their
deviate-optimal roles; the human's stat-optimal role always differs from
their deviate-optimal role, so adapting means abandoning the
stat-suggested default to fill the remaining deviate-optimal slot.

Also delivers the redesigned single-round storyboard (v3) replacing the
older `qualitative_best_respond{,_v2}.png`, in the PNAS style of the
06-07 epistemic/instrumental experiments, showing the 2 turns within
each stage and using the game-UI role emoji 🤺 / 💂 / 👩🏻‍⚕️.

## Scope & provenance

- **Data:** the 5 current treatment-condition exports (04-23 … 05-11),
  clean games only (any-dropout games excluded entirely):
  **204 bot rounds, 102 participants**.
- **Positional ground truth:** everything goes through
  `shared.data_loading.build_bot_round_layout` (CLAUDE.md → "Bot Round
  Ground Truth"); logical-order config is used only for the human's
  stat-optimal / deviate-optimal roles (index 0 = human).
- **Bayesian observer:** Stage-1 fitted params from
  `2026-05-25-full-pipeline/stage1_inference/best_inference_params.json`
  (τ_prior = 4.638, ε = 0.0624, memory `drift_prior_0.500`), asserted
  byte-identical to the 05-12 fit at load. Posteriors come from the
  05-12 `compute_posteriors` fed a **position-order** `role_seq` (the
  human's logged roles at their position, the constant bot roles at the
  bot positions) — its `preferred_action` reconstruction is exactly the
  bots' fixed behavior.
- **Logged vs reconstructed:** human actions and per-turn HP are logged
  (`stage.turns`); bot actions are not logged and are reconstructed via
  `preferred_action(bot_role, intent, start-of-turn logged team HP)`.
  `enemyIntentSequence` is longer than the played round and is clipped.
- **Inference timing:** a report made at stage s is about stage s−1 and
  pairs with `posteriors[s]` (the start-of-stage-s belief); bots never
  switch, so correctness = `guessed == bot_role_map[pos]`.

## Headline numbers

- Overall (participant means): **stat-optimal 60%**, **deviate-optimal
  26%** of stages.
- Behavior types (05-28 thresholds): **46 stat-adherent (45%)** /
  **36 mixed (35%)** / **20 deviator (20%)**.
- By stage: P(deviate-optimal) rises **0.225 → 0.381** (stages 1 → 5)
  while P(stat-optimal) falls **0.642 → 0.451** (see `summary.md` for
  CIs and the symmetry breakdown).

**Recomputed vs prior art:** all aggregates are recomputed from raw
exports here and asserted at runtime against
`2026-05-28-paper-figures/bot_adaptation_summary.md` (204/102 scope,
60%/26% overall rates, 46/36/20 types) and the 05-25
`fig_bot_role_choice` stage-1 values (23%/64%, ±2pp). The new analysis
adds per-stage cluster-bootstrap CIs (cluster = participant), which no
prior figure had.

## Files

| File | What it does |
|------|--------------|
| `common_bot.py` | Shared scaffolding: clean bot-round records, Stage-1 params, position-order posteriors, game-UI emoji glyphs |
| `adaptation_by_stage.py` | Figure 1 (`adaptation_by_stage.{png,pdf}`) + `summary.md`; runs the reproduction asserts |
| `case_search.py` | Scores successful-adaptation rounds → `case_candidates.md`; defines the pin rule |
| `storyboard_v3.py` | Figure 2 (`adaptation_case.{png,pdf}`) for the pinned case |
| `stuff to incorporate/` | The four figure files |

## Pinned storyboard case

`01KRBKSTM48HJWYZ0J4SRBRN0Z` r4, participant
`01KRBKZ6Z204TWFGJKMSNVMH02`, treatment `114_222_222__MFF_FMT`: human at
P2 (stats 1/1/4, stat-opt Medic, deviate-opt Fighter), bots P1 = Medic,
P3 = Tank. Trajectory **M → F → F → F → F**, WIN (boss at 0 HP on turn
9); both pre-switch bot inferences correct. Pin rule (see
`case_candidates.md`): highest-scoring candidate whose switch lands at
stage 2–3 with a ≥ 2-stage stable tail — the raw top scorer switched
only at the final stage, which doesn't illustrate *stable* adaptation.
Note (game_id, round_number) alone is ambiguous: each player in a game
has their own bot rounds, so the participant id is part of the pin.

## Emoji rendering

The medic 👩🏻‍⚕️ is a ZWJ sequence that Pillow can only shape with Raqm.
`analysis/pyproject.toml` now sets `[tool.uv] no-binary-package =
["pillow"]` so Pillow builds from source against Homebrew `libraqm`
(`brew install libraqm`, plus freetype/harfbuzz/fribidi, already
present). `common_bot.emoji_glyph` checks the rendered aspect ratio and
fails loudly (or falls back to a committed `assets/` PNG) if shaping
ever breaks — it never silently draws the broken double-width glyph.

## How to run

```bash
cd analysis
uv run python experiments/2026-06-07-bot-adaptation/adaptation_by_stage.py
uv run python experiments/2026-06-07-bot-adaptation/case_search.py
uv run python experiments/2026-06-07-bot-adaptation/storyboard_v3.py
```

## Draft LaTeX captions

**Figure 1 (`adaptation_by_stage`):**

> **Adaptation to stubborn teammates in bot rounds.** Fraction of bot
> rounds (204 rounds, 102 participants) in which the human plays the
> deviate-optimal role (green) — the true optimal, which requires
> abandoning their stat-suggested role — their stat-optimal role (red),
> or another role (grey), by stage. Error bars are cluster-bootstrap
> 95\% CIs over participants; the dotted line marks chance (1/3).
> Stat-optimal play declines from 64\% to 45\% across stages while
> deviate-optimal play rises from 23\% to 38\%: a subset of
> participants integrate their teammates' observed behavior and adapt
> (20/102 deviate reliably, 46/102 never abandon their stat-suggested
> role, 36/102 mix).

**Figure 2 (`adaptation_case`):**

> **A successful adaptation within a single bot round.** Top: team and
> boss HP after each of the two turns per stage (logged game state).
> Middle: the role played at each stage by the human (P2; stat-optimal
> Medic, deviate-optimal Fighter) and by the two fixed-role bots
> (P1 Medic, P3 Tank), with each turn's action (A/B/H = attack, block,
> heal). The human starts on their stat-optimal role, and after one
> stage of observation switches to the deviate-optimal Fighter and
> holds it through the win (boss HP reaches 0). Bottom: a Bayesian
> observer's posterior over each bot's role at the start of each stage
> (stage 1 = the stat-based prior); triangles/outlines mark the human's
> own inference reports, which track the posterior — the switch follows
> correct beliefs about both teammates' fixed roles.
