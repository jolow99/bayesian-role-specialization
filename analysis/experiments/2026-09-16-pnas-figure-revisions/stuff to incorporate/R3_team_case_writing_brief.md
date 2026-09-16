# Brief for the paper-writing agent: Figure `R3_team_case` (2026-09-16)

You are updating the PNAS manuscript (`pnas_paper/Research_report.tex`) to
match the revised case-study figure `R3_team_case.pdf` in this folder. Use
the facts below verbatim where numbers are involved; do not invent numbers.
The four other `R3_team_case_*.pdf` files are alternate cases for an
appendix / SI if wanted.

## 1. What the figure shows (one team-round, all three players)

Game `01KQ6YDA7NHWHSTQNZB82F8H1E`, human round 2, stat profile
`114_222_222` (P1: STR 1 / DEF 1 / SUP 4; P2 and P3 both 2/2/2). Four
stages, WIN. Role trajectory (P1 P2 P3):

| Stage | roles | value played (best) | boss attacks on |
|--:|---|---|---|
| 1 | Fighter Tank Tank | 5 (23) | turn 1 |
| 2 | Medic Fighter Tank | 36 (36) | – |
| 3 | Medic Fighter Tank | 64 (64) | turns 1, 2 |
| 4 | Medic Fighter Tank | 78 (78) | – |

Story: an uncoordinated opening (two Tanks, the support-heavy P1 attacking)
is worth 5 points against a best of 23; after one stage of observing each
other, P1 moves to Medic and P2 breaks the Tank tie by switching to
Fighter; the team lands on an optimal assignment at Stage 2 and holds it.
The team ends *specialised* (three distinct roles).

## 2. Layout, top to bottom (for the caption)

- **Header row** per stage: `value v (best b)`, the boss-attack icon
  centred over the turns on which the boss attacked, `turn 1 / turn 2`
  labels, and team (blue) / boss (red) HP bars with numbers. The Start
  column shows initial HP (10 / 25).
- **One block per player** (P1, P2, P3), separated by dashed lines. Left
  Start column: player id, STR/DEF/SUP bars, then two subheaders with the
  row labels.
- **Role card** per stage: tinted by role (red Fighter, blue Tank, green
  Medic), role icon centred, the player's per-turn action icons (swords =
  attack, shield = block, heart = heal) aligned under the turn columns.
- **White strip inside the card = "Model prediction", label `π(Px | Model)`**:
  the best-fitting role-selection model's predicted distribution over the
  three roles for that player at that stage. Every bar shows its
  probability; a bold outline marks the role the player actually chose.
- **Sub-row under the card = "Model beliefs & player inferences", labels
  `Pr(Px | Py)`, `Pr(Px | Pz)`**: the Bayesian inference engine's posterior
  over Px's role held by each teammate (conditioned on that teammate's own
  role), one small bar chart per teammate. A green triangle marks the role
  that teammate actually *reported* for Px in the inference probe when the
  report was correct, red when wrong. The Start column shows the prior
  (from stat profiles; no report exists there or after the final stage).

## 3. Timing (must be stated in the caption; readers will ask)

- The prediction in stage column *s* is made **before stage s is seen**:
  it uses the engine's posterior at the *start* of stage *s* (the beliefs
  drawn in the column to the left; the Start-column prior for *s* = 1),
  the simulated HP and boss intent entering stage *s*, and each player's
  stage *s*−1 role for the stickiness term.
- The belief charts in column *s* are the **end-of-stage-s** posteriors,
  updated on the stage-*s* actions. A row therefore reads: beliefs →
  prediction → what happened (card) → updated beliefs.
- Human inference probes at stage *s*+1 ask about stage *s*, so the
  triangles over column *s* score the report made at *s*+1.
- There is no Start-column prediction because a prediction is about a
  stage's choice; the Start prior is the input to the Stage 1 prediction.

## 4. The model and its numbers

- Predictions come from **Bayesian-Walk** (BR variant), the best-fitting
  Stage-2 model in Table 2; aggregate-LL fit: τ_v = 12.09, ε_s = 0.54.
  Stage-1 engine parameters (shared by all models): τ_p = 4.638,
  ε = 0.0624, β = 0.50 drift-to-prior.
- Prediction rule: `switch_i = softmax_{τ_v}( E_{r_−i ~ posterior}[V(r_i, r_−i, s)] )`,
  `P(r_i) = (1 − ε_s)·1[r_i = r_i^{prev}] + ε_s·switch_i` (Stage 1:
  `P = switch_i`).
- Model probability of the role actually chosen, per player per stage:

| Stage | P1 | P2 | P3 | mean |
|--:|--:|--:|--:|--:|
| 1 | .24 | .30 | .30 | .28 |
| 2 | .36 | .19 | .62 | .39 |
| 3 | .91 | .67 | .61 | .73 |
| 4 | .93 | .70 | .61 | .75 |

  Mean over the 12 player-stages **0.54** (chance 0.33). The model's most
  likely role equals the chosen role in 7/12 player-stages. Switches are
  capped at ε_s ≈ 0.54 total mass by construction, so a *correctly
  anticipated* switch appears as a 0.2–0.4 bar that is nevertheless the
  tallest non-sticky bar (P1 → Medic at Stage 2: .36 vs .55 for staying
  Fighter). Repeats appear as 0.6–0.9.
- Human inference probes correct in **9/18** reports in this round; the
  engine's most likely inferred role equals the played role in 22/24
  (observer, target, stage) cells (not drawn).
- Individual model dominance (from the individual-fitting posteriors): P1
  and P3 are Bayesian-Walk-dominant (posterior .33 and .55); P2 is Top-7
  dominant (.12 on Bayesian-Walk). Say "two of the three participants" if
  this is mentioned.

## 5. Notation to harmonise in the main text

- Use **π** for role-selection policies (the figure's `π(Px | Model)`) and
  **Pr** for beliefs (`Pr(Px | Py)` = Py's belief about Px's role, computed
  by the Bayesian inference engine, not reported by humans). Adjust the
  Computational Models section so the symbol for a role-selection
  distribution is π, not P.
- Do **not** use the `P^{x|y}(r_x)` notation; it was considered and dropped.
- Do not call the beliefs "fitted observer"; say the (Bayesian) inference
  engine's posterior / model beliefs.

## 6. "value (best)" header: definition and why it replaced "combo rank"

- **value** = the expected return `V(r, s)` of eq. (value) in Methods,
  0–100 scale (reward 100·(remaining fraction of the 10-turn horizon) on
  a win, 0 otherwise), for the roles the team played, evaluated at the
  stage-start HP and averaged over boss intent with the round's mean
  attack rate. The value table has no turn counter (computed as if 10
  turns remain), so it measures how good state + roles are, not the
  literal final score.
- **best** = the maximum of the same quantity over all 27 role
  assignments at that state. Equal numbers = the team's assignment was
  optimal; the difference is the shortfall in points.
- Earlier drafts showed "combo rank k/27" (position in an argsort of the
  27 values). That ranking broke ties arbitrarily: whenever two players
  share a stat profile, position-swapped assignments have identical V,
  and the played one was reported at rank 2 while tied for best (in this
  case Stages 3–4 read "rank 2/27"). Across all 735 team-stages the raw
  rank disagreed with a tie-aware rank in 54 %; raw rank 1 in 12 % of
  team-stages vs tie-aware rank 1 in 28 %. If the manuscript anywhere
  reports "rank 1" frequencies computed from the old function, they must
  be recomputed tie-aware or replaced by value statements. Details:
  `../combo_rank_check.md`.
- Genuine value gaps that persist while roles are unchanged (e.g. the
  5-stage alternate case, Stage 4: 73 vs best 80) are the optimum moving
  under a sticky team as HP changes — exactly the stickiness
  Bayesian-Walk models. Frame them that way if discussed.

## 7. Suggested caption skeleton (edit freely, keep the facts)

> **Turn-by-turn adaptation of one human team is captured by Bayesian
> adaptation.** One round (stat profile 1/1/4, 2/2/2, 2/2/2; four stages,
> won). *Top:* expected return of the played role assignment and the best
> achievable at each stage start (V, 0–100, Methods), boss attacks, and
> team/boss HP per turn. *Rows:* for each player, the role played and the
> actions taken; inside each card, the Bayesian-Walk model's predicted
> role distribution for that stage, computed before the stage from the
> beliefs to its left (outline = role chosen; numbers = probabilities);
> below, each teammate's model-inferred belief about the player's role at
> the end of the stage, with the teammate's actual report marked green
> (correct) or red (wrong). The team opens uncoordinated (value 5 of 23),
> then P1 moves to Medic and P2 to Fighter, reaching an optimal
> assignment by Stage 2 and holding it; the model assigns the chosen roles
> mean probability 0.54 (chance 0.33), rising from 0.28 at Stage 1 to 0.75
> at Stage 4.

## 8. Alternate cases available (same style) for SI

| file | profile | trajectory | mean P(chosen) | note |
|---|---|---|--:|---|
| `R3_team_case_11B0J4_r8` | 141_222_222 | TFF → TMF ×3 | 0.59 | P2 breaks the Fighter tie to Medic at Stage 2 |
| `R3_team_case_K72DV0_r5` | 141_222_222 | TFF → TMF ×4 (5 stages) | 0.65 | Stage 4 value 73 vs best 80: optimum moved, team stayed |
| `R3_team_case_RBRN0Z_r8` | 411_141_114 | FMM → FTM ×3 | 0.66 | heterogeneous stats, stat-optimal ending |
| `R3_team_case_11B0J4_r6` | 222_222_222 | MTF → MMF → MFF → FFF | 0.47 | former pinned case; ends all-Fighter |
