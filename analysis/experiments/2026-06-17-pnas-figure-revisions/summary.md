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
(short bars, tight spacing) and now lives **inside the "Start" column** (at
each player's row, below the initial-HP bars) rather than in the old
negative-x left margin; `START_W` was widened 0.65 → 0.95 so the bars +
numeric values fit, and `x_lo` pulls in to ≈ 0. This team is symmetric
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
action icons (lowercase names, in the bottom legend band).

**Decluttered** per the latest pass: dropped the "human" sub-label, the
"team's belief" row label, the "turn" index row, the in-figure "WIN" tag,
and the top title/note (all belong in the caption). The **model
best-response overlay was removed entirely** — the gold ▾ best-response
flags, the "best-resp: X" text, and the dashed gold "stuck" card outlines
are gone (the value ranks + the card trajectory already carry the
rational + sticky story without the extra ink).

The **legend sits in a horizontal band along the bottom** of the figure (it
was a tall top-left column) laid out as **two centered rows**, which —
together with the stats moving into the Start column — frees the left and top
whitespace and tightens the figure. Row 1: the **role** key (fighter / tank /
medic mini-cards matching the light-tinted stage cards, lowercase) + the
**action** key (attack / block / heal), then the **green "Py correctly infers
Px"** caret. Row 2: a **paired team(blue)/boss(red) HP bar** key (the inline
"team HP" / "boss HP" strip labels are removed — this is now the only HP key),
the **👹 "boss attacks"** marker, a worked mini bar-chart for the belief bars
keyed generically as **`Pr(Px | Py) = Py's belief of Px`** (spaces kept around
the pipe — a bare `|` reads like an "I" at small sizes), then the **red "Py
wrongly infers Px"** caret. The two inference carets are **colour-coded
(green = correct, red = wrong)** — replacing the earlier filled/hollow pair —
and the same green/red carets annotate the belief charts; the boss-attack
marker is the **👹 vector Twemoji** (was a red ▾) both in the HP strip and the
legend. Within row 1 the **fighter/tank/medic and attack/block/heal icons are
packed into tight clusters**, and **both rows are justified to a common width**
(`leg_w`) so their left/right edges align; the two inference-caret clusters
are given an identical (max) width so the green and red carets **left-align in
the same column** (start at the same x). The whole legend band is dropped a
little lower for clearer separation from the diagram.

**Dashed group separators.** Two dashed horizontal lines sit in the gaps
between the P1/P2 and P2/P3 groups (full grid width). Because each player's
belief sub-row sits *below* its own role card, the separators keep readers
from mis-grouping a player's card with the beliefs printed above it — the line
makes "card + the beliefs beneath it" read as one unit.

**What the belief sub-rows show (per-observer).** For each target player the
sub-row holds **two** mini bar-charts — **one per teammate-observer** —
giving that observer's belief about the target's role, with the observer's
own correct/wrong inference caret over its guessed-role bar and a
**`Pr(Pᵢ | Pⱼ)`** label.

**Timing alignment.** The chart under stage column `s` is the belief at the
**END of stage s** (the joint after observing stage s's actions, conditioned
on the observer's role *during* stage s), and its caret is the human report
**about stage s** — which the game logs at the *next* stage (an inference made
at stage N is about stage N−1). The leading **Start column** carries the
**initial belief** (the prior, `posteriors[0]`, uniform `[.33,.33,.33]` for
this symmetric team) before any actions, with no caret. The **last stage**
therefore shows its end-of-stage posterior but no caret (no inference is ever
reported after it). Earlier the charts were offset by one (column `s` showed
the *start*-of-stage-`s` belief about stage `s−1`), which both mislabeled the
timing and dropped the final stage's posterior; the readout is now
left-shifted by one stage so each column shows that stage's own outcome.

The `Pr(Pᵢ | Pⱼ)` label is a mild notation abuse: "Pᵢ's role given Pⱼ
knows their own role" = Pⱼ's belief about Pᵢ; the bottom legend defines it
generically as `Pr(Px | Py) = Py's belief of Px`). So the P1 row holds
`Pr(P1 | P2)` and `Pr(P1 | P3)` — the two
teammates' beliefs about P1 — not P1's beliefs about others. Each observer's belief is read off the **fitted Bayesian
observer model's joint posterior** by **conditioning on that observer's own
(known) role** and marginalizing the third player:
`P(r_target | r_obs = obs's role)`. This is *not* the same as the plain
marginal: although the prior (`exp(Σᵢ statᵢ(rᵢ)/τ)`) and action likelihood
(`∏ᵢ P(actionᵢ | roleᵢ)`) each factorize, the fitted memory step
(`drift_prior_0.5`, a convex mix of the within-stage posterior with the
prior — `pipeline.apply_boundary`) makes the joint **correlated**, so
conditioning on your own role shifts your belief about teammates. The two
observers therefore differ wherever the conditioning is informative (e.g.
this case's end-of-Stage-4 belief about P2: P1→P2 `[.76,.12,.12]` vs
P3→P2 `[.78,.11,.11]`); they
coincide only when conditioning happens to be uninformative. The
per-observer gaps are **modest in this symmetric 222 case** (the only
coupling is the drift mixing) and would be larger for heterogeneous-stat
teams. The conditioning is a pure readout of the existing joint posterior —
no separate per-player model is fitted (`conditional_role_belief` in
`team_case.py`).

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
2. **Folded boss-attack markers** into the team/boss HP strip (👹 emoji
   in the strip's lower margin) instead of a separate row. The inline
   "boss attacks" row label is dropped — it is now a legend item.
3. **Removed the green relent arrows** — the role card simply changing in
   the next stage already shows the relent.
4. **Legend moved to a horizontal band along the bottom** (was a tall
   single column in the top-left margin), laid out as two rows justified to a
   common width with tight F/T/M and A/B/H clusters — so the left margin is no
   longer reserved for it and `x_lo` pulls in to ≈ 0. Its posterior-bar key is
   relabeled generically `Pr(Px | Py) = Py's belief of Px`; the inference
   carets are the green "correctly infers" / red "wrongly infers" keys, and
   the boss-attack key is the 👹 emoji.
5. **Narrower STR/DEF/SUP stat panel + player id moved into the "Start"
   column** (at each player's row, x ∈ [0, START_W]) instead of the old
   negative-x left margin; `START_W` widened 0.65 → 0.95 so the bars +
   values fit, and the three stat rows are spaced tighter (`stat_dy = 0.105`,
   centred on the role-track row). This is the main left-whitespace reclaim.
6. **Per-turn labels** ("turn 1" / "turn 2") are printed under the HP strip
   so each stage's turn columns are explicit, and inter-group / sub-row gaps
   were tightened to cut the remaining slack.
7. **Dropped the outer column-separator lines** — only the interior
   Start/stage boundaries are drawn, so no vertical rule crosses the player
   ids at the left edge or hangs off the right edge.
8. **Dashed separators between the player groups** (P1/P2 and P2/P3), so each
   player's belief sub-row reads with its own card above it rather than with
   the player above.

**Start column.** A leading **"Start"** column (left of Stage 1) reads
top-to-bottom as: the **initial team and boss HP** (full bars, labelled
"initial") so the reader sees the starting state before any turn, then each
player's **P1/P2/P3 id + STR/DEF/SUP stat panel** at that player's row. The
team/boss HP numbers are also
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
