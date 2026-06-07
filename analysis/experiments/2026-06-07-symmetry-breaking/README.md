# 2026-06-07 symmetry-breaking — can humans resolve symmetrical cases in human rounds?

Human-round companion to `2026-06-07-bot-adaptation`: where the bot
figure shows adaptation to *stubborn* teammates, this one shows
resolution of *symmetrical* cases in the all-human setting. When two
teammates have identical stat profiles, their stat-based priors suggest
the same role, so the stats cannot solve the coordination problem —
someone has to deviate based on observed behavior.

## Design

One PNAS single-column figure (`symmetry_breaking.{png,pdf}`): fraction
of player pairs choosing the same role ("clash") by stage —
identical-stat pairs (black) vs stat-distinct pairs (purple, internal
control from the same games), chance = 1/3 dotted. Cluster-bootstrap
95% CIs with **game-level clusters** (the same 3 participants play
every round of a game, so team-round clusters would understate
uncertainty).

Pair sources (clean human rounds, n = 204 team-rounds):

| Profile | rounds | identical pairs/round | distinct pairs/round |
|---|--:|--:|--:|
| `114/141/411_222_222` | 122 | 1 (the 222/222 pair) | 2 |
| `222_222_222` (fully symmetric) | 40 | 3 | 0 |
| `411_141_114` | 42 | 0 | 3 |

## Headline numbers

- Pooled over stages, identical-stat pairs clash **+8.3pp**
  [+3.3, +13.6] more than stat-distinct pairs.
- Stage-1 clash: identical 0.388 [0.328, 0.453] (fully-symmetric
  rounds: **0.442**) vs distinct 0.319; both decline by stage 5
  (0.250 vs 0.162).
- Clashes don't persist: of 94 identical pairs clashing at stage 1,
  **71 (76%) split**, after 1.38 stages on average; 65% of identical
  pairs end the round on distinct roles.
- Mirror switches (both players simultaneously switching into the same
  role — the failure mode of behavioral symmetry-breaking): 70 among
  identical pairs.

Honest framing: stage-1 clash sits near chance, not far above it —
players already diversify somewhat from the start. The symmetry cost
is the persistent gap vs the stat-distinct control, and the resolution
is the within-round decline + fast splitting of initial clashes.

## Provenance

Data via `2026-06-05-paper-figures-v2/common.py::load_clean_human_teams`
(the 05-12 pipeline's 5-export clean-team scope — same 204 team-rounds
as the 06-05/06-07 instrumental/epistemic experiments; asserted at
runtime). This is a **new analysis** — no prior figure computed clash
rates; the qualitative counterpart is 06-05's `qualitative_flip_flop_v2`
(mirror boxes) and the candidates in its `example_candidates.md`.

## How to run

```bash
cd analysis
uv run python experiments/2026-06-07-symmetry-breaking/symmetry_breaking.py
```

## Draft LaTeX caption

> **Humans break symmetry using observed behavior.** Fraction of
> teammate pairs choosing the same role at each stage of clean human
> rounds, for pairs with identical stat profiles (black; n = 242
> pair-rounds, including the fully-symmetric 222\_222\_222 treatment)
> and stat-distinct pairs from the same games (purple; n = 370). Error
> bars are cluster-bootstrap 95\% CIs over games; dotted line = chance
> (1/3). Identical-stat pairs — whose stat-based priors point to the
> same role — clash 8pp more than stat-distinct pairs (pooled
> difference +0.083, 95\% CI [+0.033, +0.136]), but the conflict is
> resolved behaviorally: 76\% of pairs that clash at stage 1 split
> within 1.4 stages on average, and clash rates decline over the round
> for both pair types.
