# 2026-04-12 — Aggregate-LL pipeline

Rebuild of the two-stage Bayesian role-specialization fitting pipeline with
four orthogonal changes against the 2026-04-09 baseline:

1. **Bot-round ground truth fixed.** Stage 1 and Stage 2 now use
   `shared.data_loading.build_bot_round_layout(pr)` to resolve the human's
   in-game position (`pr.player_id`, **not** `config.humanRole`), to read
   bot roles directly from `botPlayers[i]["strategy"]["role"]` as integers
   (**not** through `GAME_ROLE_TO_IDX`, which is keyed by strings and
   silently returns an empty map), and to permute `player_stats` from
   logical to in-game-position order. See CLAUDE.md → "Bot Round Ground
   Truth" for the full three-bug trap.
2. **Stage 2 is fit on pooled (human + bot) data** rather than human-only.
   Bot rounds and human rounds share a single parameter vector; metrics are
   reported disaggregated per subset so any spread between them becomes the
   headline finding.
3. **Objective changed from `combo_r` to aggregate cross-entropy.** Stage 2
   now maximises `agg_ll_pooled`, an aggregate-distribution proper scoring
   rule (per-sample mean cross-entropy between empirical canonical-combo
   frequencies and the model's teacher-forced marginal per (env, stage)).
   Pearson `combo_r` is still reported but no longer optimised.
4. **Expanded Stage 1 strategy space.** In addition to `full` and
   `window_{1..4}` from 04-09 and `drift_prior_δ` from 04-09, this
   experiment adds `drift_uniform_δ` and `temper_γ` from 04-11-memory
   strategies, with a denser grid around the 04-11 peak at γ = 0.41.

Plus: game-level leave-one-out CV (stratified so every fold has both round
types), richer diagnostics (clip-floor histograms, "rerun-under-fix"
numbers for the 04-09 and 04-11 winners), and a bootstrap 95% CI on the
pooled `agg_ll` for each final fit.

## Layout

```
2026-04-12-aggregate-ll-pipeline/
├── README.md
├── audit/value_matrix_audit.py        # blocking gate 0
├── stage2_common.py                   # pooled loading + disaggregated metrics
├── memory_strategies.py               # consolidated strategy space
├── shared_utils.py                    # checkpoint/resume (verbatim 04-09 copy)
├── cv_utils.py                        # stratified LOO helpers
├── stage1_inference/tune.py           # bot-corrected stage 1
├── bayesian_belief/tune.py            # 7 Stage-2 models
├── bayesian_value/tune.py
├── bayesian_walk/tune.py
├── bayesian_thresh/tune.py
├── bayesian_walk_ps/tune.py
├── bayesian_thresh_ps/tune.py
├── mixture_ps/tune.py                 # w-only fit, walk/thresh frozen
├── cv/run_cv.py                       # game-level LOO over all 7 models
└── summary/build_summary.py           # comparison table + scatter
```

## Value-matrix audit

Run `python audit/value_matrix_audit.py` before anything else.

**Verdict for 2026-04-12:** Hypothesis A holds. `values.npy` is a
`(27, 2, H, W)` joint-combo value table indexed by `(combo, intent,
team_hp, enemy_hp)`; the table depends on the env's `player_stats`,
`boss_damage`, and HP grids, all fixed per `env_id`. For bot rounds we
use `data/envs/<env_id>/values.npy` (the existing fallback in
`online_model_sim.py` and in this experiment's `stage2_common` loader).
All 11 bot `env_id`s have matrices on disk — no regeneration required.

## Running

The pipeline is sequential but resumable at every phase via
`shared_utils` checkpoints. Order:

```bash
cd analysis/experiments/2026-04-12-aggregate-ll-pipeline

# 1. Audit
python audit/value_matrix_audit.py

# 2. Stage 1 (bot-corrected, writes best_inference_params.json)
python stage1_inference/tune.py

# 3. Stage 2 (order matters: mixture_ps reads the standalone _ps fits)
python bayesian_belief/tune.py
python bayesian_value/tune.py
python bayesian_walk/tune.py
python bayesian_thresh/tune.py
python bayesian_walk_ps/tune.py
python bayesian_thresh_ps/tune.py
python mixture_ps/tune.py

# 4. Cross-validation (optional, slow)
python cv/run_cv.py

# 5. Summary table + scatter plot
python summary/build_summary.py
```

## Bot-round fix — verification checkpoints

| Check | Expected | Where |
|-------|----------|-------|
| Unit test: `bot_role_map[others[i]] == deviate_roles[i+1]` on one bot round | passes 208/208 checks | `shared/data_loading.py::build_bot_round_layout` |
| Stage 1 bot-round query count | **578** (CLAUDE.md ground truth) | `stage1_inference/tune.py` startup assertion |
| `load_team_rounds(include_bot_rounds=True)` | ≥1 bot record, non-empty `botPlayers` | assertion inside `load_team_rounds` |
| Belief on pooled records | `agg_ll ≥ mean_ll` | smoke-tested |

Under the buggy variants the stage-1 bot-query count was `~210`
(`config.humanRole` bug) or `~0` (`GAME_ROLE_TO_IDX` bug on an integer
role field); the 578 gate rejects both.

## Smoke-test observations (pre-full-run)

These are from spot-checks run during implementation using a placeholder
Stage 1 seed (`tau_prior=3.0, epsilon=0.3, memory_strategy=full`) on the
pooled 189-record dataset. They are not final fitted numbers — re-run the
full Stage 1 tune before trusting anything here.

| Model          | Tuned params                     | Pooled agg_ll | Pooled mean_ll | Pooled combo_r |
|----------------|----------------------------------|--------------:|---------------:|---------------:|
| Belief         | — (no params)                    |        -2.240 |         -2.971 |          0.563 |
| Walk-PS (fit)  | ε_switch ≈ 0.614                 |        -2.159 |         -3.022 |          0.616 |
| Value (eval)   | τ_softmax=20 (untuned)           |        -3.062 |         -4.039 |          0.033 |
| Thresh (eval)  | τ_softmax=5, δ=1.0 (untuned)     |       -13.225 |            n/a |            n/a |

**Structural finding for Thresh.** Under `agg_ll` optimisation the
threshold model still plateaus at much worse than Belief, because its
choice rule is *all-or-nothing*: at each stage the model either places
100% on the previous role or 0% on it (fully committing to a switch
candidate). This structurally breaks calibration whenever the human
actually switches, and no choice of (τ, δ) fully compensates. The 04-09
pipeline hid this behind `combo_r` optimisation, which is scale-invariant
and ignored the overconfidence. Under a proper scoring rule the
pathology is still present — but now visible in the fitted `agg_ll`
rather than confined to `mean_ll`. Interpret thresh as a
reference-only model; walk (and walk_ps) is the better-specified
stick-vs-switch formulation.

**Value-matrix ordering in bot rounds.** ``_precompute_bot`` uses the
stat-profile (*logical*) ordering of ``player_stats`` because
``values.npy`` was solved against ``config.py``'s PLAYER_STATS which is
also in logical order. Using `build_bot_round_layout` directly (as in
Stage 1) would put the human at the in-game position and corrupt the
flat-index lookup into `values[flat_idx, ...]` used by
`softmax_role_dist`, `bayesian_value`, `bayesian_walk`, and
`bayesian_thresh`. Stage 1 uses position order correctly, because
`stage.inferred_roles` keys are in-game positions; Stage 2 uses
logical order correctly, because `values` and `canonical_combo` are
both logical-order quantities. The two orderings are NOT
interchangeable — do not refactor this without reading the comment at
the top of `_precompute_bot`.

## Headline numbers (full pipeline run)

**Stage 1 winner:** `window_1`, τ_prior = 5.056, ε = 0.547.
Inference LL = -0.981 (human -0.972, bot -0.999 — balanced under the fix;
compare to 04-09's `window_1` at τ = 6.80, ε = 0.56, LL = -1.010 fit on
corrupted bot-round posteriors).

**Stage 2 ranking (pooled `agg_ll`, lower = worse):**

| Rank | Model            | Params                          | Pooled agg_ll | CV mean ± std  | Human/Bot spread |
|-----:|------------------|---------------------------------|--------------:|---------------:|-----------------:|
|  1   | mixture_ps       | w = 0.899 (walk-dominant)       |        -2.141 | -2.334 ± 0.503 |            -1.18 |
|  2   | walk_ps          | ε_switch = 0.236                |        -2.142 | -2.314 ± 0.488 |            -1.15 |
|  3   | walk             | τ = 20.00, ε_switch = 0.165     |        -2.176 | -2.350 ± 0.480 |            -1.06 |
|  4   | belief           | — (no choice params)            |        -2.491 | -2.447 ± 0.187 |            -0.37 |
|  5   | value            | τ = 50.0 (bound)                |        -2.918 | -2.935 ± 0.087 |            +0.12 |
|  6   | thresh_ps        | ε_switch = 0.278, δ = 0.030     |        -4.711 | -10.33 ± 3.70  |            -5.19 |
|  7   | thresh           | τ = 20.00, δ = 0.0              |       -15.827 | -18.17 ± 1.70  |            +9.01 |

(Spread = `agg_ll_human − agg_ll_bot`. Negative means the model fits
human rounds *worse* than bot rounds — expected because bot rounds have
two-of-three roles deterministic and therefore a lower-entropy ground
truth.)

**Observations**

- **Walk-family wins decisively.** `walk_ps`, `walk`, and `mixture_ps`
  (which collapses to ~90% walk_ps) all land within 0.04 nats of each
  other. Adding the value matrix to walk buys essentially nothing — the
  fitted softmax temperature ran to the upper bound (τ = 20) for `walk`
  and τ = 50 for `value`, meaning the softmax contribution is flat.
  Bayesian-value alone (no stickiness) is *worse* than
  Bayesian-belief, which just reports posterior marginals.
- **Belief is the baseline to beat.** Pooled `agg_ll = −2.491` with no
  choice parameters, tighter CV bounds (std ≈ 0.19 vs walk's ≈ 0.49),
  and a much smaller human-bot spread. Walk's spread (-1.15) is mostly
  bot rounds being easier, not walk fitting humans better — the CV
  confirms this is stable.
- **Thresh models remain pathological.** The all-or-nothing commit of
  the threshold switch rule drives `pooled agg_ll` to -4.71 (thresh_ps)
  and -15.83 (thresh), even under proper-scoring-rule optimisation. The
  fitted τ_softmax pushes against the upper bound (value model wants
  a flat softmax); thresh's fitted δ runs toward 0 (trying to avoid
  triggering switches). CV variance on thresh_ps is huge (std ≈ 3.70).
  This is a structural finding about the model, not an optimisation
  problem — walk's ε-blend is the right stick-vs-switch parameterisation.
- **Stage-1 "flat optimum" finding from 04-11 is partly resolved.** The
  04-11 tempered-posterior result was suspect because it was fit on
  corrupt bot-round posteriors. Under the fix, `window_1` wins again
  but `temper_0.410` comes second at -0.983 vs -0.981 — the optimum is
  *narrow-but-shallow*, with several strategies within 0.01 nats. See
  `stage1_inference/clip_floor_diagnostics.json` for the "rerun under
  fix" comparison against the 04-09 and 04-11 winners.
