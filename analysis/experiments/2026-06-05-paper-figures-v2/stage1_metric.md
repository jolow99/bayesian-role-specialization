# Task 0 — the Stage-1 inference-fit objective

## What the code actually optimizes

`stage1_evaluate` (`2026-05-12-current-export-metric-comparison/pipeline.py:475-504`), maximized over (τ_prior, ε, memory strategy) by `stage1_inference/tune.py` (coarse grid → refined grid → L-BFGS-B):

```
inference_ll = mean over all human inference reports of
               log( marginal posterior probability of the role the
                    human reported for that teammate )
```

Per report (observer i, stage s ≥ 2, target j): the joint Bayesian posterior at the *start* of stage s is marginalized over everyone but j, and the log of the probability mass on the human's reported role is averaged across all reports (floor 1e-20; 0 floor hits at the optimum).

**Three clarifications relative to what we suspected:**

1. It is the mean **log** posterior probability of the reported role, not the mean raw probability — both are reported below.
2. Whether to call it a log-likelihood: it equals a per-report log-likelihood *only* under the auxiliary assumption that humans sample reports from the model's posterior marginal (probability-matching readout). No readout/response model (no β, no lapse) is actually specified or fitted, so 'mean log posterior probability of the reported role' is the accurate name. Safest paper phrasing: *mean log posterior probability assigned to participants' role reports*.
3. **There is no β in Stage 1.** The three fitted quantities are τ_prior (prior temperature), ε (action noise in the likelihood), and the memory strategy (a stage-boundary drift-back-to-prior rate δ; fitted: `drift_prior_0.500`). β (a softmax temperature, `tau_softmax`) only appears in the Stage-2 behavioral models.

## Values on the 204 clean human team-rounds (3104 inference reports)

| Quantity | Fitted (τ=4.638, ε=0.0624, δ=0.5) | Uniform posterior (1/3) |
|---|---:|---:|
| Mean log P(reported role) — the tuned objective | **-0.8940** | -1.0986 (= ln 1/3) |
| Geometric mean P(reported role) | 0.4090 | 0.3333 |
| Arithmetic mean P(reported role) | **0.4773** | 0.3333 |
| Posterior mode = human's report | 63.9% | — |
| Posterior mode = target's true previous role | 92.0% | 33.3% (chance) |
| Human report = target's true previous role | 62.9% | — |

(The `accuracy` field stored in `best_inference_params.json` (0.9198) is mode-vs-**true**-role accuracy, not agreement with the human report.)
