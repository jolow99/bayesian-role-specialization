"""Task 0 — what exactly is the Stage-1 inference-fit objective?

Recomputes the Stage-1 objective at the fitted parameters and reports it
alongside a uniform-posterior baseline (1/3 per role), in both log and
raw-probability form, plus mode-pick accuracies. Writes stage1_metric.md.
"""

from __future__ import annotations

import numpy as np

from common import (
    SCRIPT_DIR, MemoryStrategy, compute_posteriors, load_stage1,
    load_human_inference_records, stage1_evaluate, target_marginal,
)

OUT_PATH = SCRIPT_DIR / "stage1_metric.md"


def evaluate(prepared, posterior_fn):
    """Per-report stats under an arbitrary posterior source.

    posterior_fn(data) -> list of (3,3,3) joints, one per stage boundary.
    Returns dict with mean log P(report), mean P(report), mode-vs-report
    and mode-vs-true accuracies, n.
    """
    log_p, raw_p = [], []
    mode_eq_report = mode_eq_true = report_eq_true = 0
    n = 0
    for data in prepared:
        posteriors = posterior_fn(data)
        for _obs, si, target_pos, guessed, true_prev in data["queries"]:
            if si >= len(posteriors):
                continue
            marg = target_marginal(posteriors[si], target_pos)
            log_p.append(np.log(max(marg[guessed], 1e-20)))
            raw_p.append(marg[guessed])
            mode = int(np.argmax(marg))
            n += 1
            mode_eq_report += mode == guessed
            mode_eq_true += mode == true_prev
            report_eq_true += guessed == true_prev
    return {
        "mean_log_p": float(np.mean(log_p)),
        "mean_p": float(np.mean(raw_p)),
        "geo_mean_p": float(np.exp(np.mean(log_p))),
        "mode_eq_report": mode_eq_report / n,
        "mode_eq_true": mode_eq_true / n,
        "report_eq_true": report_eq_true / n,
        "n": n,
    }


def main():
    s1, strat = load_stage1()
    print(f"Fitted Stage-1 params: tau_prior={s1['tau_prior']:.4f} "
          f"epsilon={s1['epsilon']:.6f} memory={strat.name}")

    prepared = load_human_inference_records()

    # The pipeline's own number, to confirm we're reading the same objective.
    pipe = stage1_evaluate(prepared, s1["tau_prior"], s1["epsilon"], strat)
    print(f"pipeline stage1_evaluate: inference_ll={pipe['inference_ll']:.6f} "
          f"(stored {s1['inference_ll']:.6f}), n={pipe['n']}")
    assert abs(pipe["inference_ll"] - s1["inference_ll"]) < 1e-9

    # NOTE: prepare via the pipeline's own queries (observer-free) so n
    # matches exactly; evaluate() only needs (si, target, guessed, true).
    prepared_q = [dict(d, queries=[(None, *q) for q in d["queries"]])
                  for d in prepared]

    fitted = evaluate(
        prepared_q,
        lambda d: compute_posteriors(d, s1["tau_prior"], s1["epsilon"], strat))
    uniform_post = [np.ones((3, 3, 3)) / 27.0] * 10
    unif = evaluate(prepared_q, lambda d: uniform_post)

    lines = [
        "# Task 0 — the Stage-1 inference-fit objective",
        "",
        "## What the code actually optimizes",
        "",
        "`stage1_evaluate` "
        "(`2026-05-12-current-export-metric-comparison/pipeline.py:475-504`), "
        "maximized over (τ_prior, ε, memory strategy) by "
        "`stage1_inference/tune.py` (coarse grid → refined grid → L-BFGS-B):",
        "",
        "```",
        "inference_ll = mean over all human inference reports of",
        "               log( marginal posterior probability of the role the",
        "                    human reported for that teammate )",
        "```",
        "",
        "Per report (observer i, stage s ≥ 2, target j): the joint Bayesian "
        "posterior at the *start* of stage s is marginalized over everyone "
        "but j, and the log of the probability mass on the human's reported "
        "role is averaged across all reports (floor 1e-20; 0 floor hits at "
        "the optimum).",
        "",
        "**Three clarifications relative to what we suspected:**",
        "",
        "1. It is the mean **log** posterior probability of the reported "
        "role, not the mean raw probability — both are reported below.",
        "2. Whether to call it a log-likelihood: it equals a per-report "
        "log-likelihood *only* under the auxiliary assumption that humans "
        "sample reports from the model's posterior marginal (probability-"
        "matching readout). No readout/response model (no β, no lapse) is "
        "actually specified or fitted, so 'mean log posterior probability "
        "of the reported role' is the accurate name. Safest paper phrasing: "
        "*mean log posterior probability assigned to participants' role "
        "reports*.",
        "3. **There is no β in Stage 1.** The three fitted quantities are "
        "τ_prior (prior temperature), ε (action noise in the likelihood), "
        "and the memory strategy (a stage-boundary drift-back-to-prior "
        "rate δ; fitted: `drift_prior_0.500`). β (a softmax temperature, "
        "`tau_softmax`) only appears in the Stage-2 behavioral models.",
        "",
        "## Values on the 204 clean human team-rounds "
        f"({fitted['n']} inference reports)",
        "",
        "| Quantity | Fitted (τ={:.3f}, ε={:.4f}, δ=0.5) | Uniform posterior (1/3) |"
        .format(s1["tau_prior"], s1["epsilon"]),
        "|---|---:|---:|",
        f"| Mean log P(reported role) — the tuned objective | "
        f"**{fitted['mean_log_p']:.4f}** | {unif['mean_log_p']:.4f} "
        f"(= ln 1/3) |",
        f"| Geometric mean P(reported role) | {fitted['geo_mean_p']:.4f} | "
        f"{unif['geo_mean_p']:.4f} |",
        f"| Arithmetic mean P(reported role) | **{fitted['mean_p']:.4f}** | "
        f"{unif['mean_p']:.4f} |",
        f"| Posterior mode = human's report | {fitted['mode_eq_report']:.1%} "
        f"| — |",
        f"| Posterior mode = target's true previous role | "
        f"{fitted['mode_eq_true']:.1%} | 33.3% (chance) |",
        f"| Human report = target's true previous role | "
        f"{fitted['report_eq_true']:.1%} | — |",
        "",
        "(The `accuracy` field stored in `best_inference_params.json` "
        f"({s1['accuracy']:.4f}) is mode-vs-**true**-role accuracy, not "
        "agreement with the human report.)",
    ]
    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_PATH}")
    for ln in lines:
        print(ln)


if __name__ == "__main__":
    main()
