# R3 — Human-team case study (`R3_team_case`) — compact revision

Successor to the 06-16 `R3_team_case`. Two advisor requests addressed:
a case with **≥ 4 live stages**, and a **much more compact** figure.

**New pinned case:** game `01KQ6YDF6T28MRGBK1E911B0J4`, round 6 — a fully
symmetric team (`222_222_222`), **4 live stages**, WIN. (The 06-16 case
`01KRBT30…X48RYB` r4 logged 4 stages but the 4th was a spurious post-win
duplicate frame, so it trimmed to only 3 live stages.)

| Stage | combo | value-rank | what happens |
|---|---|---:|---|
| 1 | MTF | 17 / 27 | mis-coordinated start |
| 2 | MMF | 18 / 27 | P3 already best-responding (Fighter); **P1 stays Medic though best-response says Fighter — stickiness** |
| 3 | MFF | 9 / 27 | **P2 relents → Fighter**; P1 still stuck on Medic |
| 4 | FFF | 1 / 27 | **P1 finally relents → Fighter**; value-optimal combo, **WINS** |

P3 is the rational anchor (Fighter throughout); P1 is the sticky laggard
who holds Medic against the value signal for two stages then relents — the
visual signature of a rational + sticky *mixture* (cf. Bayesian Walk-PS,
the strong fit in `R4_individual_fitting`). The per-stage **combo value
rank** (1 = best of 27) carries this story in-figure: the team climbs
17 → 18 → 9 → 1, and P1's Medic card simply persisting across stages 2–3
before flipping to Fighter shows the stickiness directly. (Case selection
used an offline best-response check — the role maximizing the eap-weighted
expected **team** value at stage-start HP under the fitted Bayesian
observer's posterior — to confirm P1 was holding *against* a clear Fighter
signal; that overlay is **no longer drawn** on the figure, only the value
ranks and the card trajectory.) Posteriors use the 05-25 Stage-1 params
(τ_prior = 4.6385, ε = 0.0624, memory `drift_prior_0.500`), cross-checked
against common.py's 05-12 fit.

**Per-player stats** are shown as a 3-row panel mirroring the game UI's
`PlayerStats` component — one row per STR / DEF / SUP (text label) with a
bar of value/6 and the value — colour-linked to the role each stat favours
(STR→Fighter red, DEF→Tank blue, SUP→Medic green). The panel is **compact**
(short bars, tight spacing) so it no longer eats the left margin; the
reclaimed left whitespace lets `x_lo` pull in. This team is symmetric
(all 2/2/2), so capabilities give no role hint and reaching the optimum is
a pure coordination problem (state this in the caption — there is no
in-figure title/note).

**Game-icon encoding.** Role cards show the game's role emoji (🤺 / 💂 /
👩‍⚕️) with **no** role-letter label, and each turn's action is the game's
action emoji (⚔️ attack / 🛡️ block / 💚 heal) instead of an A/B/H letter.
The cards are now drawn as a **light role-tinted panel** (≈15% role colour
fill + a crisp role-coloured border) rather than a solid colour block, so
the role/action icons read on a near-white ground — the action emoji sit
directly on the card (the old white chips behind them are gone, which
looked awkward against the solid fill). The legend keys both the role and
action icons, each laid out on a single horizontal "Roles" / "Actions" row.

**Decluttered** per the latest pass: dropped the "human" sub-label, the
"team's belief" row label, the "turn" index row, the in-figure "WIN" tag,
and the top title/note (all belong in the caption). The **model
best-response overlay was removed entirely** — the gold ▾ best-response
flags, the "best-resp: X" text, and the dashed gold "stuck" card outlines
are gone (the value ranks + the card trajectory already carry the
rational + sticky story without the extra ink).

The **legend sits in the empty top-left block** (left of the Start column,
above the P1 group) as a compact single column — no bottom legend, so the
freed bottom whitespace is cut. The **Roles** key (Fighter / Tank / Medic
mini-cards matching the light-tinted stage cards) and the **Actions** key
(attack / block / heal) each occupy one **horizontal row**; the remaining
rows key a **paired team(blue)/boss(red) HP bar** (the inline "team HP" /
"boss HP" strip labels are removed — this is now the only HP key), the red ▾
**"boss attacks"** marker, a worked mini bar-chart for the posterior bars,
and the two role-inference markers as their own rows (no sub-header): filled
▲ = **correct role inference**, hollow △ = **wrong role inference**. The
per-stage carets sit on the inferred-role bar and are **labelled with the
reporting player (P1/P2/P3)** beneath, so each marker reads as "P_n inferred
this player's role correctly/wrongly".

**Why a 4-stage (not 5-stage) case.** The advisor's hard requirements were
a 5-stage round that (i) starts near-worst, (ii) shows sustained
stickiness then relents, (iii) lands on the single best combo (rank 1),
(iv) climbs monotonically, and (v) has correct beliefs at the decision
points. A full sweep of all 18 five-live-stage WIN rounds found **none**
that satisfies the set: the thrashing that makes a round last 5 stages is
exactly what violates it (every 5-stage round spikes back toward rank ~27
mid-round, and **none ends at rank 1** — the best ends at rank 2). The
closest 5-stage candidate (`01KQ6YDD23…` r4, heterogeneous stats,
beliefs 88% correct) ends rank 3, is non-monotone, and its final
stickiness margin is a near-tie (0.82), which would undercut the
"sticky despite a clear value signal" claim. The 4-stage `01KQ6YDF` r6
was chosen as the cleanest case that actually lands on rank 1 with a
sustained stickiness-then-relent arc and correct beliefs.

**Compactness changes (less whitespace):**

1. **Merged rows.** Each player's role track is merged with the "belief
   about that player" row into one labelled group — was 3 role tracks +
   3 belief rows (6 rows); now 3 stacked groups (role card on top, the
   team's posterior belief + teammates' reports directly beneath).
2. **Folded boss-attack markers** into the team/boss HP strip (red carets
   in the strip's lower margin) instead of a separate row. The inline
   "boss attacks" row label is dropped — it is now a legend item.
3. **Removed the green relent arrows** — the role card simply changing in
   the next stage already shows the relent.
4. **Legend moved to the empty top-left** (was a multi-row block under the
   last player group); the bottom whitespace it occupied is removed.
5. **Narrower STR/DEF/SUP stat panel** (shorter bars, tighter spacing). The
   **player id (P1/P2/P3) sits at the far left**, vertically centred on its
   row, leaving the legend the full top-left block.
6. **Per-turn labels** ("turn 1" / "turn 2") are printed under the HP strip
   so each stage's turn columns are explicit, and inter-group / sub-row gaps
   were tightened to cut the remaining slack.

**Start column.** A leading **"Start"** column (left of Stage 1) shows the
**initial team and boss HP** (full bars, labelled "initial") so the reader
sees the starting state before any turn. The team/boss HP numbers are also
printed on the **first turn of each stage as well as the last** (single-turn
stages collapse to one label), so every stage's entering and exiting HP is
legible. The inline "team HP" / "boss HP" strip labels are gone — the paired
blue/red HP bar key in the legend carries that now.

All icons (role, action, stat) are **true-vector Twemoji** via the local
`icons.py` (generalizes the 06-16 `svg_icons.py` to any committed
`assets/*.svg`; action/stat glyphs ⚔️/🛡️/💚 fetched as Twemoji SVGs); the
figure PDF contains **0 raster image XObjects**. Full-width (7 in).

# R4 — Individual model fitting figure (revised palette)

Posteriors over 13 models for 102 participants, renormalized from the per-participant log-likelihoods in `2026-05-28-paper-figures/results.json` (agg_ll-objective fits; lapse = 0.05; Stage-1: tau_prior = 4.6385, epsilon = 0.0624, memory = `drift_prior_0.500`). Mixture-PS excluded (collapses onto Bayesian Walk-PS; would double-count Walk-PS in the normalizer). The fitting numbers are unchanged from `2026-06-07-instrumental-rationality`; only the colour encoding changed from the 06-16 version (R2 -> renamed R4).

## This revision

All 13 models are shown individually (6 Bayesian + 7 non-Bayesian baselines) under two titled sub-legends. The 06-16 version distinguished the baselines with **hatching**, which read badly at single-column width — the diagonal/dot/cross lines turned to mud in the thin bars. Fixes:

1. **No hatching.** Every segment is a solid fill.
2. **Two-family palette.** Bayesian models are a COOL family (dark/light blue, green, teal, dark/light purple — a model and its `-PS` variant share a hue at different lightness); non-Bayesian baselines are a WARM/NEUTRAL family (orange, magenta, brown, red, pale gold, amber, grey). Cool-vs-warm is a redundant cue matching the two sub-legend headers; within each family the hues are hand-tuned for separation and stack-adjacent segments are checked for luminance contrast.
3. **White separators.** A slightly heavier white edge keeps the bands legible without any fill texture. Single-column (3.42 in).

The honest normalizer (each bar sums to 1) is unchanged — nothing is dropped or collapsed.

## All seven non-Bayesian baselines (shown individually)

| Model | colour | max per-participant posterior | dominant for |
|-------|--------|--:|--:|
| Random Walk | `#e6550d` | 0.5873 | 5 |
| Top-7 | `#c51b7d` | 0.9901 | 20 |
| Random-to-Optimal | `#8c510a` | 0.8628 | 5 |
| Optimal | `#b2182b` | 0.9846 | 9 |
| Copy Others | `#fff7bc` | 0.0004 | 0 |
| Contradict Others | `#fb9a29` | 0.9945 | 8 |
| Random | `#969696` | 0.9473 | 12 |

## Dominant-model counts (recomputed, match stored `dominant_counts`)

| Model | n participants |
|-------|--:|
| Bayesian Walk | 15 |
| Bayesian Walk-PS | 6 |
| Bayesian-Belief | 7 |
| Bayesian-Value | 7 |
| Bayesian Thresh-PS | 8 |
| Random Walk | 5 |
| Top-7 | 20 |
| Random-to-Optimal | 5 |
| Optimal | 9 |
| Contradict Others | 8 |
| Random | 12 |
