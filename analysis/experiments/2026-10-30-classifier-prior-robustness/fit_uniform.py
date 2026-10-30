"""Robustness check on the 10-27 two-type classifier: non-Bayesians answer uniformly.

Same model and fitting procedure as 2026-10-27-inference-soft-classifier/fit.py,
except the non-Bayesian (D) class reports each role with probability 1/3
instead of drawing from the stat-based prior P_0:
    L_i = (1 − θ)·∏_q P_N(r_q) + θ·(1/3)^{n_i}
Bayesians are unchanged (stat-based prior at tau, unmixed P_N, drift_prior
beta, λ = 0.01 lapse). The uniform class needs no lapse ((1 − λ)/3 + λ/3 = 1/3).

If the fit (and the classification) matches the stat-prior version, the
non-Bayesians' answers are not detectably using the stat-based prior.

1. Theta profile: for each theta on a grid, coarse tau x eps x beta grid then
   L-BFGS-B polish of (tau, eps, beta).
2. Joint L-BFGS-B over (tau, eps, beta, theta) from the best profile point;
   theta is rounded to 3 decimals and (tau, eps, beta) re-polished at that value.

Outputs (this folder): theta_profile_uniform.json, best_params_uniform.json,
participant_types_uniform.csv.

Run from analysis/:
    uv run python experiments/2026-10-30-classifier-prior-robustness/fit_uniform.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
CLASSIFIER_DIR = SCRIPT_DIR.parent / "2026-10-27-inference-soft-classifier"
sys.path.insert(0, str(CLASSIFIER_DIR))

import pipeline  # noqa: E402
from pipeline import _answer_probs, _participant_logprob, load_tagged_answers  # noqa: E402

assert Path(pipeline.__file__).parent == CLASSIFIER_DIR, pipeline.__file__

# Same grids and bounds as the 10-27 fit.py
THETA_VALS = np.round(np.linspace(0.0, 1.0, 21), 3)
TAU_VALS = np.linspace(0.5, 15.0, 8)
EPS_VALS = np.linspace(0.05, 0.9, 8)
BETA_VALS = np.linspace(0.0, 1.0, 5)
TAU_BOUNDS = (0.05, 30.0)
EPS_BOUNDS = (0.001, 0.999)
BETA_BOUNDS = (0.0, 1.0)
PROFILE_PATH = SCRIPT_DIR / "theta_profile_uniform.json"
OUTPUT_PATH = SCRIPT_DIR / "best_params_uniform.json"
TYPES_PATH = SCRIPT_DIR / "participant_types_uniform.csv"
STAT_PRIOR_TYPES_PATH = CLASSIFIER_DIR / "participant_types.csv"


def mixture_loglik_uniform(prepared, answers, tau_prior, epsilon, beta, theta):
    """As pipeline.mixture_loglik, but the D class answers uniformly (1/3)."""
    p_B, _ = _answer_probs(prepared, answers, tau_prior, epsilon, beta)
    log_B = _participant_logprob(p_B, answers)
    log_D = np.array([len(ix) for ix in answers["by_participant"]]) * np.log(1.0 / 3.0)

    with np.errstate(divide="ignore"):
        log_w_B = np.log1p(-theta)
        log_w_D = np.log(theta)
    log_L = np.logaddexp(log_w_B + log_B, log_w_D + log_D)
    ll_total = float(log_L.sum())
    return {
        "ll_total": ll_total,
        "ll_per_answer": ll_total / len(p_B),
        "log_B": log_B,
        "log_D": log_D,
        "resp_prior": np.exp(log_w_D + log_D - log_L),  # P(t_i = D | answers)
        "n": len(p_B),
    }


def evaluate(prepared, answers, tau, eps, beta, theta):
    m = mixture_loglik_uniform(prepared, answers, float(tau), float(eps),
                               float(beta), float(theta))
    return {
        "ll_per_answer": m["ll_per_answer"],
        "ll_total": m["ll_total"],
        "mean_resp_prior": float(m["resp_prior"].mean()),
        "n": m["n"],
        "tau_prior": float(tau),
        "epsilon": float(eps),
        "beta": float(beta),
        "theta": float(theta),
    }


def polish(prepared, answers, theta, tau0, eps0, beta0):
    def obj(params):
        return -mixture_loglik_uniform(prepared, answers, params[0], params[1],
                                       params[2], theta)["ll_per_answer"]

    opt = minimize(obj, [tau0, eps0, beta0], method="L-BFGS-B",
                   bounds=[TAU_BOUNDS, EPS_BOUNDS, BETA_BOUNDS],
                   options={"maxiter": 100, "ftol": 1e-9})
    return evaluate(prepared, answers, *opt.x, theta)


def profile_theta(prepared, answers, theta):
    coarse = [evaluate(prepared, answers, tp, eps, b, theta)
              for tp in TAU_VALS for eps in EPS_VALS for b in BETA_VALS]
    best = max(coarse, key=lambda r: r["ll_per_answer"])
    return polish(prepared, answers, theta,
                  best["tau_prior"], best["epsilon"], best["beta"])


def fit_joint(prepared, answers, start):
    def obj(params):
        return -mixture_loglik_uniform(prepared, answers, *params)["ll_per_answer"]

    x0 = [start["tau_prior"], start["epsilon"], start["beta"], start["theta"]]
    opt = minimize(obj, x0, method="L-BFGS-B",
                   bounds=[TAU_BOUNDS, EPS_BOUNDS, BETA_BOUNDS, (0.0, 1.0)],
                   options={"maxiter": 100, "ftol": 1e-9})
    theta = round(float(opt.x[3]), 3)
    return polish(prepared, answers, theta, opt.x[0], opt.x[1], opt.x[2])


def main():
    print("=" * 66)
    print("Two-type classifier, uniform non-Bayesian class: tau, eps, beta, theta")
    print("=" * 66)

    prepared, answers = load_tagged_answers()

    print("\n--- Theta profile (tau, eps, beta polished at each theta) ---")
    profile = []
    for theta in THETA_VALS:
        r = profile_theta(prepared, answers, theta)
        profile.append(r)
        print(f"  theta={theta:.3f}  LL={r['ll_per_answer']:.4f}  "
              f"tau={r['tau_prior']:.3f}  eps={r['epsilon']:.4f}  "
              f"beta={r['beta']:.3f}  mean P(D)={r['mean_resp_prior']:.3f}", flush=True)
    with open(PROFILE_PATH, "w") as f:
        json.dump(profile, f, indent=2)

    start = max(profile, key=lambda r: r["ll_per_answer"])
    print(f"\nProfile best: theta={start['theta']:.3f} LL={start['ll_per_answer']:.4f}")

    print("\n--- Joint L-BFGS-B over (tau, eps, beta, theta) ---")
    best = fit_joint(prepared, answers, start)
    if best["ll_per_answer"] < start["ll_per_answer"]:
        best = start

    output = dict(best, non_bayesian_class="uniform", inference_target="unmixed",
                  inference_lapse=pipeline.INFERENCE_LAPSE)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    m = mixture_loglik_uniform(prepared, answers, best["tau_prior"], best["epsilon"],
                               best["beta"], best["theta"])
    with open(TYPES_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["participant_id", "n_answers", "log_B", "log_D", "p_uniform_type"])
        for i, pid in enumerate(answers["participant_ids"]):
            w.writerow([pid, len(answers["by_participant"][i]),
                        f"{m['log_B'][i]:.4f}", f"{m['log_D'][i]:.4f}",
                        f"{m['resp_prior'][i]:.4f}"])

    # Compare the classification with the stat-prior (10-27) classifier
    with open(STAT_PRIOR_TYPES_PATH) as f:
        stat = {r["participant_id"]: float(r["p_prior_type"]) for r in csv.DictReader(f)}
    uni = m["resp_prior"] > 0.5
    old = np.array([stat[pid] > 0.5 for pid in answers["participant_ids"]])
    print(f"\nBest: theta={best['theta']:.3f} tau={best['tau_prior']:.4f} "
          f"eps={best['epsilon']:.4f} beta={best['beta']:.3f} "
          f"LL={best['ll_per_answer']:.4f} mean P(D)={best['mean_resp_prior']:.3f}")
    print(f"Non-Bayesians (P > 0.5): uniform class {uni.sum()}, stat-prior class {old.sum()}, "
          f"both {(uni & old).sum()}, agree on {(uni == old).sum()} of {len(uni)}")
    print(f"Saved to {OUTPUT_PATH}, {PROFILE_PATH}, {TYPES_PATH}")


if __name__ == "__main__":
    main()
