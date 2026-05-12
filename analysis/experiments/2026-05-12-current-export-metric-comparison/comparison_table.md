# Metric Comparison (all 5 current exports, human-only, clean teams)

Fitted on 204 clean human team-rounds from the five current treatment-condition exports (04-23, 04-27a, 04-27b, 05-11a, 05-11b — see `pipeline.load_human_team_records`). Every team-round must have a matching value matrix in `human_envs_value_matrices/<stat_profile>__<role_combo>`; missing matrices raise rather than silently dropping records. Each cell shows the metric value when the model was fit under that objective.

## Eval metric: `combo_r`

| Model | Fit: combo_r | Fit: agg_ll | Fit: mean_ll |
|-------|--------:|--------:|--------:|
| bayesian_walk_ps | 0.4703 | 0.4635 | 0.4687 |
| bayesian_walk | 0.5047 | 0.4949 | 0.4901 |
| mixture_ps | 0.4722 | 0.4635 | 0.4687 |
| bayesian_thresh_ps | 0.3545 | 0.3449 | 0.3442 |
| bayesian_thresh | 0.3213 | 0.2836 | 0.2967 |
| bayesian_belief | 0.4703 | 0.4703 | 0.4703 |
| bayesian_value | 0.4096 | 0.4081 | 0.4081 |

## Eval metric: `marg_r`

| Model | Fit: combo_r | Fit: agg_ll | Fit: mean_ll |
|-------|--------:|--------:|--------:|
| bayesian_walk_ps | 0.1961 | 0.2258 | 0.2114 |
| bayesian_walk | 0.3576 | 0.3588 | 0.3479 |
| mixture_ps | 0.2171 | 0.2258 | 0.2114 |
| bayesian_thresh_ps | 0.2886 | 0.2841 | 0.2837 |
| bayesian_thresh | 0.3495 | 0.2651 | 0.2812 |
| bayesian_belief | 0.1961 | 0.1961 | 0.1961 |
| bayesian_value | 0.1852 | 0.1878 | 0.1831 |

## Eval metric: `agg_ll`

| Model | Fit: combo_r | Fit: agg_ll | Fit: mean_ll |
|-------|--------:|--------:|--------:|
| bayesian_walk_ps | -2.5181 | -2.5117 | -2.5136 |
| bayesian_walk | -2.5073 | -2.4756 | -2.4846 |
| mixture_ps | -2.5118 | -2.5117 | -2.5136 |
| bayesian_thresh_ps | -8.1085 | -7.4217 | -7.4219 |
| bayesian_thresh | -10.5305 | -9.8620 | -9.9329 |
| bayesian_belief | -2.5181 | -2.5181 | -2.5181 |
| bayesian_value | -2.5675 | -2.5655 | -2.5793 |

## Eval metric: `mean_ll`

| Model | Fit: combo_r | Fit: agg_ll | Fit: mean_ll |
|-------|--------:|--------:|--------:|
| bayesian_walk_ps | -2.9042 | -2.9064 | -2.8981 |
| bayesian_walk | -2.9254 | -2.8269 | -2.8068 |
| mixture_ps | -2.9063 | -2.9064 | -2.8981 |
| bayesian_thresh_ps | -24.9757 | -24.2088 | -24.2088 |
| bayesian_thresh | -27.1870 | -26.9340 | -26.8541 |
| bayesian_belief | -2.9042 | -2.9042 | -2.9042 |
| bayesian_value | -3.0992 | -3.1175 | -3.0934 |

## Fitted Parameters

| Model | Objective | Params |
|-------|-----------|--------|
| bayesian_walk_ps | combo_r | epsilon_switch=1.0000 |
| bayesian_walk_ps | agg_ll | epsilon_switch=0.8231 |
| bayesian_walk_ps | mean_ll | epsilon_switch=0.9188 |
| bayesian_walk | combo_r | tau_softmax=7.2065, epsilon_switch=0.5590 |
| bayesian_walk | agg_ll | tau_softmax=12.0943, epsilon_switch=0.5415 |
| bayesian_walk | mean_ll | tau_softmax=15.7357, epsilon_switch=0.6135 |
| mixture_ps | combo_r | w=0.9042 |
| mixture_ps | agg_ll | w=1.0000 |
| mixture_ps | mean_ll | w=1.0000 |
| bayesian_thresh_ps | combo_r | epsilon_switch=0.9992, delta=0.2500 |
| bayesian_thresh_ps | agg_ll | epsilon_switch=0.3862, delta=0.0000 |
| bayesian_thresh_ps | mean_ll | epsilon_switch=0.3618, delta=0.0000 |
| bayesian_thresh | combo_r | tau_softmax=2.4709, delta=0.4643 |
| bayesian_thresh | agg_ll | tau_softmax=18.5785, delta=0.4286 |
| bayesian_thresh | mean_ll | tau_softmax=10.0496, delta=0.4643 |
| bayesian_belief | combo_r | (none) |
| bayesian_belief | agg_ll | (none) |
| bayesian_belief | mean_ll | (none) |
| bayesian_value | combo_r | tau_softmax=13.7160 |
| bayesian_value | agg_ll | tau_softmax=11.6222 |
| bayesian_value | mean_ll | tau_softmax=16.8575 |

## Stage 1 inference parameters (best)

- `tau_prior=4.6385`, `epsilon=0.0624`, memory strategy `drift_prior_0.500`
- Per-query inference log-likelihood = -0.894 (vs chance log(1/3) ≈ -1.099)
- Per-query argmax accuracy = 91.98% over 3104 inference queries from
  204 clean human teams. No floor hits.

## Interpretation

### Scope and dropout exclusion

- All five current treatment-condition exports are loaded: 04-23, 04-27a,
  04-27b, 05-11a, 05-11b. 1200 player-rounds in total.
- `pipeline.discover_dropout_games` flags **16 of 50 games** as dropout
  (≥1 `isDropout=True` row). Those games are removed entirely before any
  human-team is built. The remaining 34 games yield **204 complete clean
  human team-rounds** (≈6 rounds/game), the universe for Stage 2.
- Every record is paired with a value matrix at
  `human_envs_value_matrices/<stat_profile>__<role_combo>`. Missing
  matrices would raise — no silent drops occurred this run.

### Stage 1 — what the inference parameters say

- The winning memory strategy is again `drift_prior_0.500`: every stage,
  the posterior is pulled 50% of the way back to the original prior.
  Compared with the 3-export run (tau=3.40, eps=0.001), the larger pool
  (3104 vs 1672 queries) shifts the noise estimate from "effectively no
  noise" to ε≈0.06 and the prior temperature from 3.4 to 4.6 (slightly
  sharper). The qualitative story — forgetful, soft prior, fairly clean
  posterior updates — is unchanged.
- 92% argmax accuracy at the inference targets is high; the marginal-posterior
  inference engine is a strong predictor of what humans guess about
  teammates. That sets the floor for the Stage-2 PS-style models.

### Stage 2 — model ranking (combo_r objective)

```
bayesian_walk(0.505) > mixture_ps(0.472) > bayesian_belief(0.470)
  > bayesian_walk_ps(0.470) > bayesian_value(0.410)
  > bayesian_thresh_ps(0.355) > bayesian_thresh(0.321)
```

- **`bayesian_walk` is the only model that beats `bayesian_belief`
  across every eval metric** (combo_r 0.505 vs 0.470; agg_ll -2.48 vs
  -2.52; mean_ll -2.81 vs -2.90). Its win comes from combining (a) value-aware
  softmax over EVs given the posterior with (b) ε-mixture stickiness on
  the previous role. Fitted ε_switch settles at 0.54–0.61, i.e. about
  half the probability mass goes to "stay with what I picked last time"
  — consistent with the 35–45% switch rate observed in the behavioral
  experiment.
- **PS-style siblings collapse to the belief baseline.** `walk_ps`
  fits ε_switch→1 under combo_r (which makes it identical to the pure
  posterior marginal) and only modestly stickier under the LL objectives.
  `mixture_ps` keeps the walk-PS branch with w≈0.9–1.0, so it inherits the
  same metric. Without the value matrix, posterior-sampling can't
  distinguish "I should switch towards a higher-EV role" from "the
  marginal says my old role is still likely," so the value-aware walk model
  pulls ahead.
- **`bayesian_value` underperforms `bayesian_belief`** (0.41 vs 0.47 combo_r).
  Pure softmax over EVs without stickiness over-predicts switching;
  humans are clearly less responsive to EV than a free softmax assumes.
- **Threshold models do worst.** Both `bayesian_thresh` and
  `bayesian_thresh_ps` fit small deltas (δ≈0 under LL, δ=0.25–0.46 under
  combo_r) but suffer catastrophic log-likelihood (-8 to -27) because the
  threshold gating zeros out switches the model considers "not worth
  it" but that humans actually take. This drives floor hits and is the
  reason the agg_ll / mean_ll cells look orders of magnitude worse than
  walk/belief.
- **Marginal r ≠ combo r.** `bayesian_walk` and `bayesian_thresh` lead on
  marg_r (0.36 and 0.35) — meaning models that explicitly value-rank or
  threshold-rank roles track the human marginal role distribution better
  even when their full-combo predictions are mediocre. The PS-only
  models barely move on marg_r (0.19) because the unconditional
  posterior marginal is too flat to capture the bimodal stat-optimal /
  deviate-from-stat pull.

### How the 5-export sample changes the picture vs 2026-05-11 (3 exports)

| Metric (combo_r-fit, eval=combo_r) | 3 exports (n=114) | 5 exports (n=204) |
|---|---|---|
| bayesian_walk | 0.4348 | **0.5047** |
| bayesian_belief / walk_ps | 0.3977 | 0.4703 |
| bayesian_value | 0.3717 | 0.4096 |
| bayesian_thresh | 0.2623 | 0.3213 |

Adding the two new treatment conditions raises every model's combo_r by
~0.04–0.07; the ranking is preserved. `bayesian_walk` is the biggest
beneficiary (+0.07), which is consistent with the new data being noisier
on stat-optimal but cleaner on stage-to-stage stickiness — exactly the
signal the walk model is built to use.

### Bottom line

- The value-aware sticky walker (`bayesian_walk`) is the best fit and
  remains the clear winner now that we have all five treatment conditions
  (combo_r 0.50, agg_ll -2.48 at fitted ε_switch ≈ 0.55, τ ≈ 7–12).
- Posterior-only models track the posterior marginal well (combo_r 0.47)
  but cannot exploit the value matrix and so plateau below the walk model.
- Threshold gating is the wrong inductive bias on this dataset: humans
  switch more freely than the gate allows, which the log-likelihood
  metrics punish heavily.
- Stage-1 strategy (forgetful posterior, drift-to-prior 0.5) is stable
  across sample sizes, suggesting the memory model is identifying a real
  property of how humans update beliefs.
