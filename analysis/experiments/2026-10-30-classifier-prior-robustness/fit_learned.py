"""Robustness check on the 10-27 two-type classifier: non-Bayesians use a learned role distribution.

Same model as 2026-10-27-inference-soft-classifier/fit.py, except the
non-Bayesian (D) class reports F / T / M with fixed probabilities
(a, b, 1 − a − b), learned, ignoring stats and the target player:
    L_i = (1 − θ)·∏_q P_N(r_q) + θ·∏_q [(1 − λ)·p_{r_q} + λ/3]
Bayesians are unchanged (stat-based prior at tau, unmixed P_N, drift_prior
beta, λ = 0.01 lapse). (a, b, 1 − a − b) = softmax(z_F, z_T, 0).

Fit: for each (tau, eps, beta), P_N is computed once and (theta, a, b) are
maximized on it (cheap: the D class only needs each participant's F/T/M
report counts). Coarse tau x eps x beta grid (same as the 10-27 fit), then
L-BFGS-B polish of (tau, eps, beta) with (theta, a, b) re-maximized inside.

Outputs (this folder): best_params_learned.json, participant_types_learned.csv.

Run from analysis/:
    uv run python experiments/2026-10-30-classifier-prior-robustness/fit_learned.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_expit, log_softmax

SCRIPT_DIR = Path(__file__).resolve().parent
CLASSIFIER_DIR = SCRIPT_DIR.parent / "2026-10-27-inference-soft-classifier"
sys.path.insert(0, str(CLASSIFIER_DIR))

import pipeline  # noqa: E402
from pipeline import _answer_probs, _participant_logprob, load_tagged_answers  # noqa: E402

assert Path(pipeline.__file__).parent == CLASSIFIER_DIR, pipeline.__file__

# Same grids and bounds as the 10-27 fit.py
TAU_VALS = np.linspace(0.5, 15.0, 8)
EPS_VALS = np.linspace(0.05, 0.9, 8)
BETA_VALS = np.linspace(0.0, 1.0, 5)
TAU_BOUNDS = (0.05, 30.0)
EPS_BOUNDS = (0.001, 0.999)
BETA_BOUNDS = (0.0, 1.0)
INNER_BOUNDS = [(-10.0, 10.0)] * 3  # logit theta, z_F, z_T
ROLES = "FTM"
OUTPUT_PATH = SCRIPT_DIR / "best_params_learned.json"
TYPES_PATH = SCRIPT_DIR / "participant_types_learned.csv"
STAT_PRIOR_TYPES_PATH = CLASSIFIER_DIR / "participant_types.csv"
UNIFORM_TYPES_PATH = SCRIPT_DIR / "participant_types_uniform.csv"


def report_counts(answers):
    """(n_participants, 3) counts of F / T / M reports."""
    inferred = np.asarray(answers["inferred"])
    return np.array([np.bincount(inferred[ix], minlength=3)
                     for ix in answers["by_participant"]])


def role_probs(z):
    return np.exp(log_softmax(np.r_[z[0], z[1], 0.0]))


def mixture_terms(x, log_B, counts):
    """x = (logit θ, z_F, z_T) -> per-participant log L_i, log D_i, P(D | answers)."""
    lam = pipeline.INFERENCE_LAPSE
    p = role_probs(x[1:])
    log_D = counts @ np.log((1.0 - lam) * p + lam / 3.0)
    log_L = np.logaddexp(log_expit(-x[0]) + log_B, log_expit(x[0]) + log_D)
    return log_L, log_D, np.exp(log_expit(x[0]) + log_D - log_L)


def fit_inner(log_B, counts, n_answers, x0=(0.0, 0.0, 0.0)):
    opt = minimize(lambda x: -mixture_terms(x, log_B, counts)[0].sum() / n_answers,
                   np.asarray(x0, float), method="L-BFGS-B", bounds=INNER_BOUNDS,
                   options={"maxiter": 500, "ftol": 1e-13, "gtol": 1e-10})
    return opt


def evaluate(prepared, answers, counts, tau, eps, beta):
    p_B, _ = _answer_probs(prepared, answers, float(tau), float(eps), float(beta))
    log_B = _participant_logprob(p_B, answers)
    n = len(p_B)
    best = min((fit_inner(log_B, counts, n, x0) for x0 in [(0, 0, 0), (-2, 0, 0), (2, 0, 0)]),
               key=lambda o: o.fun)
    p = role_probs(best.x[1:])
    return {"ll_per_answer": float(-best.fun), "ll_total": float(-best.fun * n), "n": n,
            "tau_prior": float(tau), "epsilon": float(eps), "beta": float(beta),
            "theta": float(np.exp(log_expit(best.x[0]))),
            "p_F": float(p[0]), "p_T": float(p[1]), "p_M": float(p[2]),
            "_x": best.x, "_log_B": log_B}


def main():
    print("=" * 66)
    print("Two-type classifier, learned (a, b, 1-a-b) non-Bayesian class")
    print("=" * 66)

    prepared, answers = load_tagged_answers()
    counts = report_counts(answers)

    print("\n--- Coarse tau x eps x beta grid ((theta, a, b) maximized at each) ---")
    coarse = [evaluate(prepared, answers, counts, tp, eps, b)
              for tp in TAU_VALS for eps in EPS_VALS for b in BETA_VALS]
    start = max(coarse, key=lambda r: r["ll_per_answer"])
    print(f"Grid best: tau={start['tau_prior']:.3f} eps={start['epsilon']:.3f} "
          f"beta={start['beta']:.2f} LL={start['ll_per_answer']:.4f}", flush=True)

    print("\n--- L-BFGS-B polish of (tau, eps, beta) ---")
    opt = minimize(lambda v: -evaluate(prepared, answers, counts, *v)["ll_per_answer"],
                   [start["tau_prior"], start["epsilon"], start["beta"]],
                   method="L-BFGS-B", bounds=[TAU_BOUNDS, EPS_BOUNDS, BETA_BOUNDS],
                   options={"maxiter": 100, "ftol": 1e-9})
    best = evaluate(prepared, answers, counts, *opt.x)
    if best["ll_per_answer"] < start["ll_per_answer"]:
        best = start

    log_L, log_D, resp = mixture_terms(best["_x"], best["_log_B"], counts)
    output = {k: v for k, v in best.items() if not k.startswith("_")}
    output.update(mean_resp_prior=float(resp.mean()), n_free_params=6,
                  non_bayesian_class="learned (a, b, 1-a-b)",
                  inference_target="unmixed", inference_lapse=pipeline.INFERENCE_LAPSE)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    with open(TYPES_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["participant_id", "n_answers", "n_F", "n_T", "n_M",
                    "log_B", "log_D", "p_learned_type"])
        for i, pid in enumerate(answers["participant_ids"]):
            w.writerow([pid, counts[i].sum(), *counts[i],
                        f"{best['_log_B'][i]:.4f}", f"{log_D[i]:.4f}", f"{resp[i]:.4f}"])

    cls = resp > 0.5
    ids = answers["participant_ids"]
    emp = counts[cls].sum(axis=0) / counts[cls].sum()
    print(f"\nBest: theta={output['theta']:.3f} tau={output['tau_prior']:.4f} "
          f"eps={output['epsilon']:.4f} beta={output['beta']:.3f} "
          f"LL={output['ll_per_answer']:.4f}")
    print(f"Learned non-Bayesian role distribution: "
          + "  ".join(f"{r}={output['p_' + r]:.3f}" for r in ROLES))
    print(f"Empirical report frequencies of classified non-Bayesians: "
          + "  ".join(f"{r}={e:.3f}" for r, e in zip(ROLES, emp)))
    for name, path, col in [("stat-prior", STAT_PRIOR_TYPES_PATH, "p_prior_type"),
                            ("uniform", UNIFORM_TYPES_PATH, "p_uniform_type")]:
        with open(path) as f:
            other = {r["participant_id"]: float(r[col]) > 0.5 for r in csv.DictReader(f)}
        o = np.array([other[pid] for pid in ids])
        print(f"Non-Bayesians (P > 0.5): learned {cls.sum()}, {name} {o.sum()}, "
              f"both {(cls & o).sum()}, agree on {(cls == o).sum()} of {len(ids)}")
    print(f"Saved to {OUTPUT_PATH} and {TYPES_PATH}")


if __name__ == "__main__":
    main()
