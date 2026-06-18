# 2026-06-17 — PNAS figure revisions (round 2: R3 + R4)

Second pass of revisions on the role-self-organization PNAS figures,
following advisor (Xuan) feedback on the 06-16 versions. Only **two**
figures this time. Final artifacts go in `stuff to incorporate/` as
vector PDF (+ 300 dpi PNG), colorblind-safe, no in-figure titles.

| File | Was | What changed |
|------|-----|--------------|
| `R4_individual_fitting.pdf` | `R2_individual_fitting` (06-16) | renamed R2→R4; all 13 models shown individually (6 Bayesian + 7 baselines) under the two titled sub-legends; **hatching removed** (solid fills) — replaced by a **two-family palette** (cool Bayesian / warm-neutral baselines, hand-tuned for separation) + heavier white separators; single-column |
| `R3_team_case.pdf` | `R3_team_case` (06-16) | new 4-live-stage case ending rank 1; **leading "Start" column** (initial team/boss HP); **light role-tinted cards** (transparent fill + coloured border, no white action chips); game-UI role + action emoji on cards (no letters); compact **narrower** STR/DEF/SUP stat panel with **P1/P2/P3 at the far left**; **per-turn "turn 1/2" labels**; merged role+belief rows, boss-attacks folded into HP strip, relent arrows + best-response overlay removed; decluttered; **small** P1/P2/P3 ids + stat panels; **roomy top-left legend** (lowercase role/action keys, no group headers), paired team/boss HP bar key, correct/wrong **role-inference** markers; **per-observer belief sub-rows** (two mini posteriors per player per stage — each teammate's belief conditioned on its own role, `P(r_target | r_obs)`); inline team/boss HP labels removed; HP numbers on the first turn of each stage too |

`summary.md` collects the case-selection rationale and the (unchanged)
model-fitting numbers.

## Why each changed

**R4 — individual fitting.** All 13 models are shown individually (6
Bayesian + 7 non-Bayesian baselines) under the two titled sub-legends. The
06-16 version distinguished the baselines with **hatching**, which read
badly at single-column width — the diagonal/dot/cross lines turned to mud
in the thin bars. This revision **removes hatching** (solid fills) and
instead leans on a purpose-built **two-family palette**: Bayesian models are
a **cool** family (dark/light blue, green, teal, dark/light purple — a model
and its `-PS` variant share a hue at different lightness) and the baselines
a **warm/neutral** family (orange, magenta, brown, red, pale gold, amber,
grey). Cool-vs-warm is a redundant cue that matches the two sub-legend
headers; within each family the hues are hand-tuned for separation and
stack-adjacent segments are checked for luminance contrast. A heavier white
separator between segments keeps the bands legible without fill texture.
Kept **single-column** (3.42 in). The fitting numbers are unchanged (same
`dominant_counts` assertion, honest normalizer — each bar sums to 1; nothing
dropped or collapsed).

**R3 — team case.** Advisor asked for a 5-stage round meeting a strict
story checklist (near-worst start, sustained stickiness→relent, lands on
rank 1, monotone climb, correct beliefs) and for per-player stats. A sweep
of all 18 five-live-stage WIN rounds found **none** that meets the
checklist — the thrashing that makes a round run 5 stages is exactly what
violates it, and none ends at rank 1 (see `summary.md`). With the user, we
chose the cleanest case that does land on rank 1 with a sustained
stickiness→relent arc and correct beliefs: the 4-live-stage symmetric
round `01KQ6YDF` r6. The figure uses the **game-UI icons** (role emoji on
cards with no role letter; the action emoji ⚔️/🛡️/💚 per turn instead of
A/B/H). The role cards are now a **light role-tinted panel** (≈15% fill +
a crisp role-coloured border) rather than a solid colour block, with the
action emoji sitting directly on the card (the old white chips are gone —
they looked awkward against the solid fill). A leading **"Start" column**
(left of Stage 1) shows the **initial team / boss HP** before any turn. The
**compact, narrower** 3-row STR/DEF/SUP stat panel (label + bar of value/6 +
value) like the game's `PlayerStats` keeps the **P1/P2/P3 id at the far
left** (vertically centred on its row), leaving the legend the full top-left
block. **Per-turn "turn 1 / turn 2" labels** run under the HP strip. It is
**decluttered** — no "human" / "team's belief" / "WIN" / title labels (those
go in the caption) — and **much more compact** (merged each player's role
track with its belief row, folded boss-attacks into the HP strip, removed
the green relent arrows and the model best-response overlay). The **legend
sits in the top-left block** as a compact single column — the bottom legend
(and its whitespace) is gone and the rows are **generously spaced**. The
**role** (fighter / tank / medic mini-cards, lowercase) and **action**
(attack / block / heal) keys each occupy one row with **no group header**;
the rest key a **paired team(blue)/boss(red) HP bar** (the inline "team HP" /
"boss HP" strip labels are removed), the red ▾ **"boss attacks"** marker, a
mini bar-chart keyed as **"teammate's role belief"**, and the two
role-inference markers as their own rows: filled ▲ = **correct role
inference**, hollow △ = **wrong role inference**. The P1/P2/P3 ids and
STR/DEF/SUP stat panels are kept **small** to keep the left margin compact.
HP numbers are labeled on the **first turn of each stage** as well as the
last.

**Per-observer belief sub-rows.** Each target player's belief sub-row holds
**two** mini bar-charts — one per teammate-observer — showing that
observer's belief about the target's role, with the observer's own
correct/wrong caret over its guessed-role bar and a `Pr(Pi | Pj)` label
beneath (mild notation abuse for "Pj's belief about Pi", defined by example
in the legend) — so the P1 row reads `Pr(P1 | P2)`, `Pr(P1 | P3)`.
Each is read off the fitted **Bayesian observer model's joint posterior** by
**conditioning on the observer's own (known) role** and marginalizing the
third player: `P(r_target | r_obs = obs's role)` (`conditional_role_belief`
in `team_case.py`). This differs from the plain marginal because the fitted
memory step (`drift_prior_0.5`, a convex mix of the within-stage posterior
with the prior) makes the joint **correlated**, so conditioning on your own
role shifts your belief about teammates — the two observers therefore differ
wherever the conditioning is informative (gaps are modest in this symmetric
222 case, larger for heterogeneous-stat teams). No separate per-player model
is fitted — it's a readout of the existing joint posterior.

## Scripts

| Script | Produces |
|--------|----------|
| `individual_fitting_revision.py` | `R4_individual_fitting` (.pdf/.png) + summary section |
| `team_case.py` | `R3_team_case` (.pdf/.png) |

Run from `analysis/`:

```bash
uv run python experiments/2026-06-17-pnas-figure-revisions/individual_fitting_revision.py
uv run python experiments/2026-06-17-pnas-figure-revisions/team_case.py
```

## Provenance / dependencies

- `team_case.py` imports `common_human` (data join + posteriors +
  best-response) from `../2026-06-16-pnas-figure-revisions/` via `sys.path`
  — the cross-folder pattern that experiment already uses — and the **local
  `icons.py`** for all glyphs. `icons.py` generalizes the 06-16
  `svg_icons.py` to render any committed `assets/*.svg` as true-vector
  Twemoji PathPatches; `assets/` holds the role SVGs (🤺/💂/👩‍⚕️) plus the
  action/stat SVGs (⚔️=`action_attack`, 🛡️=`action_block`, 💚=`action_heal`,
  Twemoji 14.0.2). The PDF stays 0 raster XObjects.
- `individual_fitting_revision.py` reads `MODEL_ORDER` and the stored
  per-participant log-likelihoods from `../2026-05-28-paper-figures/`
  (`paper_figures.py`, `results.json`); it overrides only the colours.
- Stage-1 inference params: 05-25 full pipeline, cross-checked
  byte-identical to common.py's 05-12 fit.
