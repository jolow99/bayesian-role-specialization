# Fig. 4 (R3_team_case) legend / whitespace options, 2026-10-15

Generator: `legend_options.py` (imports the 09-16 `team_case_model.py` unchanged for data, model predictions and diagram drawing; replaces only the legend and layout). Re-run: `cd analysis && .venv/bin/python experiments/2026-10-15-fig4-legend-options/legend_options.py [--case K72DV0:5 --suffix _5stage] [--only A,D]`.
Case = the paper's 4-stage case (2F8H1E r2, values 5/36/64/78). `current.{png,pdf}` is regenerated with the unmodified 09-16 `render` and is **pixel-identical** to the paper PNG. `contact_sheet.png` compares everything at the same 7 in width.

All options: 7.00 in wide, vector PDF + 300 dpi PNG, no title, same data/styling/keys. Every option adds the header key `value (best): expected return of chosen roles (best possible)` (grey "value (best)" styled like the header). Diagram text/markers are scaled uniformly with the diagram (factor K), so the diagram looks like the current one, just larger.

**Important scale fact:** the current figure is natively 5.59 x 3.70 in (tight-bbox of a 7 in canvas); placed at 7 in width its legend text is effectively 4.9 pt x 1.25 = about 6.1 pt, numbers on bars about 5 pt. The options below set the legend at 5.1-5.6 pt (above the 4.5 pt floor but a little smaller than the current 6.1 pt effective) because the keys have to fit in two rows / a narrow column. Diagram text in the options is the same size or larger than now (numbers 5.5-6.7 pt).

| Option | File | Size (in) | Aspect (w:h) | Non-diagram area (legend + margins) | 
|---|---|---|---|---|
| Current | `current.png` (scaled to 7 in) | 7 x 4.63 | 1.51 | about 28% (about 21% pure whitespace) |
| A compact left column | `option_A` | 7 x 5.25 | 1.33 | about 19% |
| B band below (2 rows) | `option_B` | 7 x 7.03 | 1.00 | about 11% |
| C split: icon/value row above, chart keys below | `option_C` | 7 x 7.02 | 1.00 | about 11% |
| D = B + tightened vertical rhythm | `option_D` | 7 x 6.79 | 1.03 | about 11% |

(5-stage SI case K72DV0 r5: A 1.59, B 1.18, C 1.18, D 1.22 aspect; current 5-stage is `current_5stage`.)

## Options
- **A**: left column, but keys stacked/wrapped to a ~82 pt wide column (was ~37 pt gap + ~92 pt column), sections packed at the top, no gaps. Diagram gets about 81% of the width instead of about 72%, and magnified 1.38x. Con: the column's lower half is still empty (it is the least-whitespace *left* layout, not a no-whitespace one); legend text 5.6 pt.
- **B**: all keys in a hairline-ruled two-row band under the diagram, row 1 = role/action/HP/boss/value icons, row 2 = chart keys (pi, outline, Pr, carets). The two green/red caret keys are merged into one ("Py correctly / wrongly infers Px's role"; "after seeing their actions" dropped from the wording, which is the only wording loss). Diagram fills the full width (1.67x), so the figure becomes square (7 x 7.0 in). Legend 5.1 pt.
- **C**: B split by purpose: the icon/value keys sit above the stage headers (next to the header/HP rows they explain), the chart keys sit under the rows they explain. Same size as B; slightly more "keys next to what they explain", slightly more fragmented.
- **D**: B plus a tighter diagram: HP bars shortened (0.5 to 0.3 units), belief row slightly shorter. 7 x 6.79 in. A smaller win than hoped: the diagram's height is set by 3 player rows x (card + belief chart), so a full-width diagram is inherently near-square.

Keys: none dropped. Merged: the two caret keys into one row in B/C/D (kept as two rows in A). Wording shortened in A/B/C/D: "pi(Px | Model): model's predicted role of Px" -> "pi(Px | Model): predicted role of Px" in the bands.

## Trade-off and recommendation
A full-width diagram with the legend moved out of the left column necessarily goes to about 1:1 (6.8-7.0 in tall) because all the width the legend used is returned to a diagram that is then magnified. That is the "figure gets tall" worry, confirmed. If a 7 x 7 in float is acceptable (a PNAS page holds about 7 x 9 in plus caption), **D** (or C) removes the most whitespace and gives the biggest, most legible diagram. If the 3:2-ish aspect matters more, **A** (1.33:1) is the safe incremental option, but it keeps an empty block in the lower left. My pick: **D** if height is OK, otherwise **A**.

## 5-stage SI cases
Tested with K72DV0 r5 (`*_5stage.png`): all of A, B, C, D render cleanly, no overlap/clipping (legend rows have a little less slack in the 4-stage width because band width follows the diagram; at 5 stages bands have more room). Limitation: the band-row justification and the left-column wrapping are tuned for the 4/5-stage widths used here; other cases were not individually inspected. A 4-stage band row is tight (row 1 has about 2 pt gaps between keys at 5.1 pt, 0.5 pt of nominal overflow tolerance).

---
# Follow-up: D2 (grouped legend) and D3 (shorter figure)  (`grouped_variants.py`)
Earlier A-D files are untouched. New: `option_D2.{png,pdf}`, `option_D3.{png,pdf}`, `option_D2_5stage.*`, `option_D3_5stage.*`, `contact_sheet_D_variants.png` (current vs D vs D2 vs D3, same 7 in width; the older `contact_sheet.png` is unchanged).

**D2** = D layout + grouped legend, 7 x 6.81 in (aspect 1.03, about 12% non-diagram). Row 1: G1 fighter/tank/medic | G2 attack/block/heal | G3 team-boss HP + boss attacks | trailing value key. Row 2: G4 pi(Px|Model) + "role Px chose" | G5 Pr(Px|Py) + merged correct/wrong carets. Within a group 6 pt apart, between groups 26 pt. Legend text 5.6 pt (was 5.1). All keys kept.

**D3** = D2 + all height levers, 7 x 6.09 in (aspect 1.15, about 13% non-diagram; diagram is 87% of the area). 5-stage (K72DV0 r5): D2 7 x 5.77 (1.21), D3 7 x 5.17 (1.35); both render cleanly.

Height saved per lever, each applied alone to D2 (6.81 in):
| Lever | Saving |
|---|---|
| HP bars 0.30 -> 0.22 units | 0.09 in |
| Header block: boss icon moved inline beside "turn j", value row closer | 0.18 in |
| Stage titles closer to the value row | 0.07 in |
| Card height 0.76 -> 0.72 | 0.13 in |
| Belief row 0.45 -> 0.40 (tighter subheader/labels) | 0.16 in |
| Row padding (top gap .05 -> .03, group gap .16 -> .15) | 0.09 in |
| **All together (D3)** | **0.72 in** |
Not worth it / not done: margins are already 0.03 in; the card cannot go below about 0.70 (role icon meets the prediction strip); the two-line "Model beliefs & player inferences" subheader must stay two lines (one line is wider than the Start column), so the belief row cannot shrink much more.

Style decisions kept (outline on chosen bar, numbers on all bars, no correctness colouring on model plots, pi/Pr notation, Start-column subheaders). Diagram text stays about the same on-page size as the current figure or bigger.

**Recommendation:** D3 (7 x 6.1 in, 1.15:1). Caveats: the boss icons now sit left of "turn j" (check they do not read as a legend glyph), and the group gaps are fixed at 26 pt so the legend rows are left-aligned and end slightly short of the right edge.

---
# D4 = D3 + one-line Start-column subheaders (`option_D4.*`, `option_D4_5stage.*`)
- 4-stage: **7 x 5.89 in (aspect 1.19)**, diagram 87% of area (D3: 6.09). 5-stage (K72DV0 r5): 7 x 5.00 in (1.40) (D3: 5.17). Both view clean. `contact_sheet_D_variants.png` now shows current / D / D2 / D3 / D4.
- Subheaders are one line and centred in the Start column: "Model prediction  π(Px | Model)" (at the height of the card's white strip) and "Model beliefs & player inferences" (at the top of the belief row). Wording and notation unchanged; Pr(Px | Py) chart labels kept. The widest line (beliefs) just fits the 1.30-unit Start column, so I took the first fallback only: font 4.6 -> 3.9 nominal (about 7.7 -> 6.5 pt on the 7 in page, so about 1.2 pt smaller); no column widening and no shortened wording needed.
- Freed vertical space: belief row 0.40 -> 0.34 units, saving about 0.2 in.
- Clipped caret under the Pr key fixed (more bbox padding for marker glyphs).
- Legend rows: the groups now spread to the right edge (row 1 gaps about 49 pt between groups; row 2 uses the same gap and ends a little short). Legend text 5.6 pt.

---
# D5 = D4 + merged stage/value header + centred boss icons (`option_D5.*`, `option_D5_5stage.*`)
- 4-stage: **7 x 5.88 in (aspect 1.19)**; 5-stage: 7 x 5.00 in (1.40). `contact_sheet_D_variants.png` now shows current / D3 / D4 / D5.
- Header: "Stage j" (bold, dark) and "value v (best b)" (regular, grey) on one line per column, "Start" on the same line. Sizes on the 7 in page: 4-stage stage title 10.0 pt, value 6.5 pt; 5-stage the text is auto-shrunk to fit the column: 8.6 pt / 5.6 pt (value stays above 5.5 pt). The 5-stage header fills about 99% of the column width, so neighbouring headers nearly touch (no overlap).
- Boss icon is centred above "turn j" on attack turns again (slightly smaller, 0.115 vs 0.13 units). Net height effect: merging the header lines saves what putting the icons back above the turn labels costs, so D5 is 5.88 in vs D4 5.89 in (about 0.01 in).
