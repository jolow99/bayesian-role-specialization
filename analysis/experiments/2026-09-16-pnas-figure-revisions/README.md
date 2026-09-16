# 2026-09-16 — PNAS figure revisions (round 4: R3 chart + legend cleanup)

Fourth pass on the `R3_team_case` figure (paper Figure 4), following the
advisor's (Xuan) written feedback of 2026-09-14 on the 09-02/09-09 version
("tells a much better qualitative story"; a handful of presentation
changes) and the user's standardising calls of 2026-09-16. Same data, same
model, same five candidate cases as `2026-09-02-pnas-figure-revisions/`;
only the in-card model chart, the Start column, the header row, the legend
and the notation changed. **No variants** — one standard style.

Final artifacts in `stuff to incorporate/` as vector PDF (+ 300 dpi PNG),
no in-figure titles (captions live in the paper), 7 in full-width.

## Changes vs 09-02

| Ask (Xuan 09-14 / user 09-16) | Done |
|---|---|
| Remove the correct/wrong indication from the model plots (colours confusing; a near-uniform prediction can't be "wrong") | Green/red caret under the model's top pick is gone. No correctness colouring anywhere in the model chart. |
| Highlight the role actually chosen — triangle or bold outline | **Bold dark outline** round the chosen role's bar (user's call; a grey-caret alternative is one switch away, `CHOSEN_MARK`). |
| Numbers on top of each bar | Every bar carries its probability (plain 4 pt, identical style for all three; the outline alone marks the chosen role). |
| Label the model chart, but don't repeat it in every box | The chart label lives **once per row in the Start column**, at the height of the cards' white strips: a small **"Model prediction"** subheader with `π(P1 \| Model)` stacked under it, left-aligned. The cards themselves are unlabelled. |
| Subheader demarcating beliefs vs prediction | Start column carries **"Model prediction"** (card row, strip height, above the π label) and **"Model beliefs & player inferences"** (two lines above the two labelled prior charts `Pr(P1 \| P2)`, `Pr(P1 \| P3)`; belief sub-row is 0.48 units tall to fit it). |
| Player id: stats to the right of the number | `P1` sits at the left of its STR/DEF/SUP panel on the card's icon line (06-17 arrangement), not above it. Start column widened to 1.30 units. |
| Legend: model block above belief block | Reordered. |
| "Py's belief of Px" → "Py's belief about Px's role if Py were a Bayesian observer" | Done verbatim. |
| `Pr(Px \| BW)` → `Pr(Px \| Model)` → **π** for the policy | Legend keys, kept terse because the in-figure subheaders now carry the explanation: `π(Px \| Model): model's predicted role of Px`, outline = `role Px chose`, `Pr(Px \| Py): model of Py's belief about Px's role`, two keys `▲ Py correctly / wrongly infers Px's role after seeing their actions`. Xuan's longer wordings ("…if Py were a Bayesian observer", "…role selection by best-fitting model") go in the caption instead. **Main text should use π for role-selection policies and Pr for beliefs.** `P^{x\|y}(r_x)` was not adopted. |
| Legend: team/boss HP and boss attacks on one row | Done. |
| Boss-attack icon centred on top of its turn (was to the right, ambiguous) | Header row now reads, top to bottom: combo rank · boss icon (centred on the turn column, only on attack turns) · "turn j" · HP values · HP bars. |
| (not asked) | Start-column belief labels `Pr(P1 \| P2)` with spaced pipes — unspaced read as `Pr(P1IP2)`. Legend gap to the Start column widened (text is in pt, layout in data units, so the 5-stage render is the tight one). |

Belief-chart green/red carets (Py's own reported guess, correct / wrong) are
**unchanged** — the removal request was specific to the model plots.

## Timing (for the caption)

The prediction in stage column *s* is made **before** stage *s* is seen: it
uses the model's beliefs at the start of stage *s* (the posterior after
stages 1…*s*−1 — drawn in the column to the left, the Start prior for
*s* = 1), the simulated HP / boss intent entering stage *s*, and each
player's stage-(*s*−1) role for the stickiness term. The belief charts in
column *s* are the **end-of-stage** beliefs, updated on the stage-*s*
actions. So a row reads: beliefs (left) → prediction → card (what
happened) → updated beliefs. There is no Start-column prediction because a
prediction is about a stage's choice; the Start prior is the input to the
Stage 1 prediction. (Showing entering-stage beliefs in the same column as
the prediction was tried and reverted on 09-02: it loses the post-final
beliefs and misaligns the human-report carets.)

## Outputs (`stuff to incorporate/`)

| File | Case |
|---|---|
| `R3_team_case.{pdf,png}` | **Primary**: `01KQ6YDA…2F8H1E` r2 (114_222_222, FTT → MFT ×3, WIN). Identical to `R3_team_case_2F8H1E_r2.*`. |
| `R3_team_case_11B0J4_r8` | `01KQ6YDF…11B0J4` r8 (141_222_222, TFF → TMF ×3) |
| `R3_team_case_K72DV0_r5` | `01KQ6YDC…K72DV0` r5 (141_222_222, 5 stages, TFF → TMF ×4) |
| `R3_team_case_RBRN0Z_r8` | `01KRBKST…RBRN0Z` r8 (411_141_114, FMM → FTM ×3) |
| `R3_team_case_11B0J4_r6` | `01KQ6YDF…11B0J4` r6 (222_222_222, MTF → MMF → MFF → FFF; the 06-17 pin) |

Case notes (fit quality, dominant models) are in the 09-02 README; the
per-stage P(chosen) tables are regenerated in `summary.md` (numbers
unchanged from 09-02).

## Script

`team_case_model.py` — the 09-02 script with the changes above. Style
constants at the top (`NOTATION = "pi"`, `CHOSEN_MARK = "outline"`) exist
so the legend and the card renderer cannot disagree; they are not meant to
be flipped for the paper. Run from `analysis/`:

```bash
uv run python experiments/2026-09-16-pnas-figure-revisions/team_case_model.py
```

Dependencies (loaded by path, unchanged): renderer pieces + icons from
`../2026-06-17-pnas-figure-revisions/`, data scaffolding from
`../2026-06-16-pnas-figure-revisions/common_human.py`, trajectories from
`../2026-05-25-full-pipeline/pipeline.py`, fitted params + individual
posteriors from `../2026-05-28-paper-figures/results.json`.
