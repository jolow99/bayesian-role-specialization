# 2026-09-02 — PNAS figure revisions (round 3: R3 + model predictions)

Third pass on the `R3_team_case` figure (paper Figure 4), following the
advisor (Xuan) call of 2026-08-31 (Fathom recording 178159729). The 06-17
figure showed humans adapting but nothing in it showed that the
*computational model* explains that adaptation. This revision adds the
best-fitting model's per-player role-choice predictions and renders
several candidate cases, including ones that end **specialized** (three
distinct roles) rather than all-Fighter.

Final artifacts go in `stuff to incorporate/` as vector PDF (+ 300 dpi
PNG), no in-figure titles (captions live in the paper), 7 in full-width.

## What changed vs 06-17

| Change | Detail |
|---|---|
| **Model-prediction sub-row** (new) | Per player, between the role card and the belief sub-row: one mini bar-chart per stage with the **Bayesian-Walk-BR** predicted distribution over that player's role, `Pr(Px \| model)`. The bar for the role the player *actually* chose is outlined in black with its probability printed above it. |
| **Carets now score the model, not the human** | In both model rows a caret sits under the model's *most likely* role: green if it is the actual role (the chosen role in the prediction row; the played role in the belief row), red otherwise. The 06-17 carets marked whether the *human's reported* inference was right; those human reports are no longer drawn. |
| **Belief bars relabelled** | Legend key is now `Pr(Px \| Py): the model's inference of Py's belief about Px's role` — the advisor found "fitted observer's posterior" unclear about where the numbers come from. It is our Stage-1 Bayesian observer run on the actual action history, conditioned on Py's own role; not a human report. |
| **Legend moved to a left column** | The paper figure is full-width, so horizontal space is cheap: the legend is one left-aligned column to the left of the Start column (roles, actions, HP/boss, belief key, prediction key, green/red caret keys), vertically centred on the diagram. Figure is now ≈ 7 × 4.9 in (was ≈ 7 × 7.6 in with the bottom band). |
| **Case is a parameter** | `CASES` in `team_case_model.py` lists (game-id suffix, round, output stem). The first entry is also written as plain `R3_team_case.{pdf,png}`. |

Everything else (Start column, stat panels, HP strip, combo ranks,
per-observer conditional beliefs, timing alignment, vector Twemoji icons)
is inherited unchanged from 06-17 — the script imports the 06-17
`team_case.py` renderer pieces and `icons.py` by path.

## Which model, which numbers

Best-fitting model = **Bayesian Walk** (results.json key; paper name
Bayesian-Walk-BR), the top model by `combo_r` / `agg_ll` / `mean_ll` in
`2026-05-28-paper-figures/aggregate_table.md`. Params from that
`results.json` (agg_ll-objective fit): `tau_softmax = 12.094`,
`epsilon_switch = 0.5415`. Stage-1: `tau_prior = 4.6385`,
`epsilon = 0.0624`, memory `drift_prior_0.500` (05-25 full pipeline).

Prediction at stage *s* for player *i*:

```
switch_i = softmax_{tau_v}( E_{r_-i ~ posterior_s}[ V(r_i, r_-i, state_s) ] )
P(r_i)   = (1 - eps_s) * 1[r_i = r_i^{s-1}] + eps_s * switch_i      (s > 1)
P(r_i)   = switch_i                                                  (s = 1)
```

The trajectory (start-of-stage posterior, simulated HP, boss intent) is
recomputed with the 05-25 pipeline's own `precompute_trajectories`, so the
numbers are exactly what the model was scored on during fitting. The
script asserts that these start-of-stage posteriors match the 06-16
scaffolding's `human_posteriors` (max abs diff < 1e-9) and that the
pipeline's combos match the logged roles.

**Reading the row.** The chart under stage column *s* is the prediction
*for* stage *s*, made from the belief drawn in the column to its left
(end of stage *s−1*; the Start column's prior for *s = 1*) plus the
stage-*s−1* roles. Switches are always predicted at ≤ `eps_s` ≈ 0.54
total mass, so a *correctly anticipated* switch shows up as a bar around
0.2–0.4 that is nonetheless the tallest non-sticky bar; repeats show up
as 0.6–0.9. In a round's final stage the value matrix is often flat
(every combo wins), so the softmax is uniform and P(chosen) collapses to
the stickiness floor `1 − eps_s + eps_s/3 = 0.64` for a repeated role —
worth a caption sentence.

## Cases rendered (`stuff to incorporate/`)

| File | Case | Env | Trajectory | mean P(chosen) | Notes |
|---|---|---|---|--:|---|
| `R3_team_case` = `R3_team_case_2F8H1E_r2` | `01KQ6YDA…2F8H1E` r2 | 114_222_222 | FTT → MFT → MFT → MFT (rank 24 → 1 → 2 → 2), WIN | 0.54 | **Primary candidate.** Two identical-stat Tanks split; the SUP-4 player moves from Fighter to Medic; team ends specialized. P1, P3 individually Walk-dominant. |
| `R3_team_case_11B0J4_r8` | `01KQ6YDF…11B0J4` r8 | 141_222_222 | TFF → TMF ×3, WIN | 0.59 | Same team as the 06-17 case; P2 breaks the Fighter tie to Medic at stage 2. P2, P3 Walk-dominant. |
| `R3_team_case_K72DV0_r5` | `01KQ6YDC…K72DV0` r5 | 141_222_222 | TFF → TMF ×4, WIN | 0.65 | Only 5-stage candidate; best mean fit, but no participant is Bayesian-dominant. |
| `R3_team_case_RBRN0Z_r8` | `01KRBKST…RBRN0Z` r8 | 411_141_114 | FMM → FTM ×3, WIN | 0.66 | Heterogeneous stats; stat-optimal end. No participant Bayesian-dominant. |
| `R3_team_case_11B0J4_r6` | `01KQ6YDF…11B0J4` r6 | 222_222_222 | MTF → MMF → MFF → FFF, WIN | 0.47 | The 06-17 pinned case, for comparison. Both relents get ≈ 0.25; ends all-Fighter. |

Per-stage P(chosen) tables, the count of player-stages where the model's
most likely role is the chosen one (green carets, prediction row), the
model's inference accuracy (green carets, belief row), and each
participant's dominant model (from the R4 individual-fitting posteriors)
are in `summary.md`.

## Scripts

| Script | Produces |
|---|---|
| `team_case_model.py` | all `R3_team_case*.{pdf,png}` + `summary.md` |

Run from `analysis/`:

```bash
uv run python experiments/2026-09-02-pnas-figure-revisions/team_case_model.py
```

## Provenance / dependencies

- Renderer pieces + icons: `../2026-06-17-pnas-figure-revisions/{team_case.py,icons.py,assets/}`
  (loaded by path as `team_case_0617` — the 06-16 folder also has a
  `team_case.py`, and a plain import picks the wrong one).
- Data join / posteriors / value ranks: `../2026-06-16-pnas-figure-revisions/common_human.py`.
- Trajectories: `../2026-05-25-full-pipeline/pipeline.py` (loaded as
  `pipeline_0525`; the 05-12 pipeline is already imported as `pipeline`).
- Model params + individual posteriors: `../2026-05-28-paper-figures/results.json`.
