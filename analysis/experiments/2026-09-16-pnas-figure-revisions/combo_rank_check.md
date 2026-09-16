# Combo-rank check (2026-09-16)

Question: the "combo rank k/27" header in `R3_team_case` sometimes goes UP
between stages although the team kept the same roles. Is that real, and how
big are the gaps? Scripts: `combo_rank_cases.py` (the five figure cases,
per stage) and `combo_rank_aggregate.py` (all 204 clean team-rounds, 735
team-stages). Run from `analysis/` with `uv run python <script>`.

## How the rank is computed today

`common_human.stage_value_rank`: `V` = the 27 position-ordered combos'
expected return (paper eq. `V(r, s)`, 0–100 scale, reward 100·t/H on a win)
at the STAGE-START HP, intent-averaged with the round's mean attack rate;
rank = position of the played combo in `argsort(-V)`. **Ties are broken
arbitrarily by argsort order.**

## Three separate effects

1. **Tie-break artefact (the big one).** Whenever two positions have the
   same stat profile (`114_222_222`, `141_222_222`, `222_222_222` — 4 of the
   5 figure cases), position-swapped combos have identical `V` (MFT = MTF).
   The played combo then lands at rank 2 half the time although it is tied
   for best. Over all 735 team-stages, raw rank ≠ tie-aware ("dense") rank
   in **394 (54 %)**; raw rank 1 in **12 %** of team-stages vs dense rank 1
   in **28 %**. The raw rank systematically understates how optimal teams
   were. Primary case (`2F8H1E` r2): raw 24 → 1 → 2 → 2, dense 19 → 1 → 1 → 1.
2. **End-game flatness.** When the boss is nearly dead, most combos win
   with certainty and `V` is flat at the top: in the last stage a median of
   **7** combos tie for best (2 otherwise); 119 of the 185 stages with ≥5
   tied-best combos are last stages. Raw rank there is meaningless
   (`11B0J4` r8 s4: TMF raw 10/27 while tied for best with 23 others).
3. **Genuine increases: the optimum moves under a sticky team.** With ties
   handled, consecutive-stage rank increases drop from 260 to **186**; the
   value gap to the best combo at those moments is median **13.4** points
   (IQR 6.8–24.9, max 74). In **33** of them the team kept the SAME combo
   and the state moved the optimum (e.g. `11B0J4` r8 s2–s3: after a strong
   opening the boss is low, all-Fighter becomes best, TMF trails by 12 then
   5 points). This is exactly the stickiness the Bayesian-Walk model
   predicts, so it is a feature to explain, not a blemish.

Per-stage, dense rank 1 rises from 20 % (stage 1) to 48 % (stages 4–5),
median gap to best falls 7.4 → 3.0 points.

## Canonical combos

Collapsing position-symmetric combos (`shared.parsing.canonical_combo`)
gives 18 canonical combos for `H_222_222` profiles, 10 for `222_222_222`,
27 for `411_141_114`. The canonical rank equals the dense rank in every
figure stage — canonicalisation adds nothing beyond tie handling, and its
changing denominator (18 / 10 / 27) needs a footnote. Not recommended.

## Per-stage numbers for the five cases

| case | stage | combo | HP (team, boss) | V played | V best | gap | raw | dense | canonical |
|---|--:|---|---|--:|--:|--:|--:|--:|--:|
| 2F8H1E r2 (114_222_222) | 1 | FTT | 10, 25 | 4.7 | 22.7 | 18.0 | 24 | 19 | 13/18 |
| | 2 | MFT | 5, 19 | 36.1 | 36.1 | 0 | 1 | 1 | 1/18 |
| | 3 | MFT | 10, 11 | 64.4 | 64.4 | 0 | **2** | 1 | 1/18 |
| | 4 | MFT | 4, 6 | 77.5 | 77.5 | 0 | **2** | 1 | 1/18 |
| 11B0J4 r8 (141_222_222) | 1 | TFF | 15, 25 | 40.2 | 40.2 | 0 | 1 | 1 | 1/18 |
| | 2 | TMF | 13, 17 | 47.0 | 59.4 (TFF) | 12.3 | 4 | 4 | 3/18 |
| | 3 | TMF | 15, 9 | 75.0 | 80.0 (FFF) | 5.0 | 9 | 4 | 4/18 |
| | 4 | TMF | 15, 2 | 90.0 | 90.0 | 0 | **10** | 1 (24 tied) | 1/18 |
| K72DV0 r5 (141_222_222) | 1 | TFF | 10, 25 | 12.8 | 12.8 | 0 | 1 | 1 | 1/18 |
| | 2 | TMF | 7, 16 | 30.3 | 30.3 | 0 | **2** | 1 | 1/18 |
| | 3 | TMF | 5, 12 | 40.4 | 40.4 | 0 | 1 | 1 | 1/18 |
| | 4 | TMF | 9, 6 | 73.0 | 80.0 (TFF) | 7.0 | 3 | 2 | 2/18 |
| | 5 | TMF | 7, 2 | 90.0 | 90.0 | 0 | **2** | 1 (14 tied) | 1/18 |
| RBRN0Z r8 (411_141_114) | 1 | FMM | 15, 30 | 19.0 | 31.0 (FTM) | 12.0 | 2 | 2 | 2/27 |
| | 2 | FTM | 6, 20 | 50.3 | 50.3 | 0 | 1 | 1 | 1/27 |
| | 3 | FTM | 11, 11 | 70.4 | 73.6 (FTF) | 3.2 | 4 | 3 | 3/27 |
| | 4 | FTM | 13, 3 | 90.0 | 90.0 | 0 | **4** | 1 (9 tied) | 1/27 |
| 11B0J4 r6 (222_222_222) | 1 | MTF | 15, 30 | 25.0 | 36.1 | 11.1 | 17 | 12 | 6/10 |
| | 2 | MMF | 14, 22 | 35.1 | 54.2 | 19.1 | 18 | 17 | 6/10 |
| | 3 | MFF | 12, 18 | 52.3 | 65.5 | 13.2 | 9 | 8 | 4/10 |
| | 4 | FFF | 15, 10 | 80.0 | 80.0 | 0 | 1 | 1 | 1/10 |

Bold raw ranks are pure tie-break artefacts.

## Display options

| Option | Header text (primary case) | Pros | Cons |
|---|---|---|---|
| A. dense rank (minimal fix) | `rank 19 · 1 · 1 · 1 /27` | one-line code change; kills the artefact | end-game flatness still hides that "rank 1" can mean "tied with 23 others"; gap size invisible |
| B. value vs best | `V 5 / 23 · 36 / 36 · 64 / 64 · 78 / 78` | shows the gap directly, ties disappear (equal numbers), flatness visible as V→90 | two numbers per column; needs "expected return (0–100)" in the caption |
| C. gap only | `−18 · 0 · 0 · 0` (points below best) | one number; 0 = optimal | loses the "climbing value" story |
| D. mini bar | played V as a bar with a tick at best V | visual, compact | one more graphic in a full figure |

Recommended: **B** (or A as the minimal fix). The paper already defines
`V` on a 0–100 scale, so "V 64 / 64" reads as "expected return of the
played roles / best available".
