# Single-Round Static Figure (prototype)

Tufte-style static visualization for a single human round. Companion to
the GIFs in `2026-05-24_single-team-gifs/`, but flattens the stage-by-
stage animation into one page so a reader can take in the whole round
at a glance.

## Status

Prototype on one round. Not yet wired up to render the full export.

## Design

Four panels share a turn-level x-axis (stage boundaries marked as faint
vertical guides). Convention: **Fighter at top, Tank middle, Medic
bottom**.

1. **Roles + actions** *(tall)*
   * 3 step-lines for player roles across stages
   * Per-turn markers shaped by action (▲ ATTACK, ■ BLOCK, ▼ HEAL)
     positioned **on the role line** — no separate action panel,
     because action is a deterministic function of role + intent + HP
   * Enemy-attack vertical bands behind the chart
2. **Inferences** *(compact)*
   * 6 observer→target pairs arranged in 2 columns × 3 rows
   * One letter per stage (the inferred role); green = correct relative
     to the target's role in the *previous* stage, red = wrong
3. **HP** — team / enemy areas (range-framed, 0–max), enemy-attack
   bands
4. **Model P(combo) per stage** — joint P(team's actual combo) solid
   and P(optimal combo) dashed for each of {Belief, Value, Walk}; 1/27
   chance reference. This panel carries *all* model-prediction
   information; the role panel stays clean.

## Run

```bash
cd analysis
.venv/bin/python experiments/2026-05-24_single-round-static/build_static.py
```

Picks the first clean human round with ≥4 stages and some role
variation, then writes `figures/<env>__<game>_<round>.png`.

## Key params

Tunable model params lifted verbatim from
`2026-05-12-current-export-metric-comparison/` (the published numbers):

| Param            | Value     |
|------------------|-----------|
| `tau_prior`      | 4.638     |
| `epsilon`        | 0.0624    |
| memory strategy  | `drift_prior_0.500` |
| `tau_softmax` (Value) | 13.716 |
| `tau_softmax` (Walk)  | 7.207  |
| `epsilon_switch` | 0.559     |

## Open design questions

* The model bars only encode P(actual choice). If the model put high
  mass on a *different* role, that mass is invisible. A future
  iteration could add a faint "down-bar" pointing to the model's MAP
  role when MAP ≠ actual.
* Inferences only show observer's letter + correctness, not the full
  per-role posterior. Adding model-predicted inferences as overlay
  markers is possible if we re-import the Stage-1 posterior pipeline.
