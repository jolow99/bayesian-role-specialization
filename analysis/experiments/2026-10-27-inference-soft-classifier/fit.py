"""Fit the per-participant Bayesian / prior-defaulter mixture: (tau, eps, beta, theta).

Uses this folder's pipeline.py: mixture_loglik scores each participant's
inference answers as
    L_i = (1 − θ)·∏_q P_N(r_q) + θ·∏_q P_0(r_q)
where P_N is the unmixed Bayesian posterior (drift_prior beta carried between
stages) and P_0 the stat-based prior at the same tau. λ = 0.01 lapse on both.

Human-only, clean-teams-only. Objective: total mixture log-likelihood, reported
per answer (ll_per_answer) to compare with the 05-25 / 10-27 Stage-1 fits.

1. Theta profile: for each theta on a grid, coarse tau x eps x beta grid then
   L-BFGS-B polish of (tau, eps, beta).
2. Joint L-BFGS-B over (tau, eps, beta, theta) from the best profile point;
   theta is rounded to 3 decimals and (tau, eps, beta) re-polished at that value.

Outputs (this folder): theta_profile.json, best_params.json,
participant_types.csv (per-participant log B_i, log D_i, P(t_i = D | answers)).

Run from analysis/:
    uv run python experiments/2026-10-27-inference-soft-classifier/fit.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import pipeline  # noqa: E402
from pipeline import load_tagged_answers, mixture_loglik  # noqa: E402

assert Path(pipeline.__file__).parent == SCRIPT_DIR, pipeline.__file__

THETA_VALS = np.round(np.linspace(0.0, 1.0, 21), 3)
TAU_VALS = np.linspace(0.5, 15.0, 8)
EPS_VALS = np.linspace(0.05, 0.9, 8)
BETA_VALS = np.linspace(0.0, 1.0, 5)
TAU_BOUNDS = (0.05, 30.0)
EPS_BOUNDS = (0.001, 0.999)
BETA_BOUNDS = (0.0, 1.0)
PROFILE_PATH = SCRIPT_DIR / "theta_profile.json"
OUTPUT_PATH = SCRIPT_DIR / "best_params.json"
TYPES_PATH = SCRIPT_DIR / "participant_types.csv"


def evaluate(prepared, answers, tau, eps, beta, theta):
    m = mixture_loglik(prepared, answers, float(tau), float(eps), float(beta), float(theta))
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
        return -mixture_loglik(prepared, answers, params[0], params[1],
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
        return -mixture_loglik(prepared, answers, *params)["ll_per_answer"]

    x0 = [start["tau_prior"], start["epsilon"], start["beta"], start["theta"]]
    opt = minimize(obj, x0, method="L-BFGS-B",
                   bounds=[TAU_BOUNDS, EPS_BOUNDS, BETA_BOUNDS, (0.0, 1.0)],
                   options={"maxiter": 100, "ftol": 1e-9})
    theta = round(float(opt.x[3]), 3)
    return polish(prepared, answers, theta, opt.x[0], opt.x[1], opt.x[2])


def main():
    print("=" * 66)
    print("Soft-classifier fit: tau, eps, beta, theta (per-participant mixture)")
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

    output = dict(best, inference_target="unmixed",
                  inference_lapse=pipeline.INFERENCE_LAPSE)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    m = mixture_loglik(prepared, answers, best["tau_prior"], best["epsilon"],
                       best["beta"], best["theta"])
    with open(TYPES_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["participant_id", "n_answers", "log_B", "log_D", "p_prior_type"])
        for i, pid in enumerate(answers["participant_ids"]):
            w.writerow([pid, len(answers["by_participant"][i]),
                        f"{m['log_B'][i]:.4f}", f"{m['log_D'][i]:.4f}",
                        f"{m['resp_prior'][i]:.4f}"])

    print(f"\nBest: theta={best['theta']:.3f} tau={best['tau_prior']:.4f} "
          f"eps={best['epsilon']:.4f} beta={best['beta']:.3f} "
          f"LL={best['ll_per_answer']:.4f} mean P(D)={best['mean_resp_prior']:.3f}")
    print(f"Saved to {OUTPUT_PATH}, {PROFILE_PATH}, {TYPES_PATH}")


if __name__ == "__main__":
    main()
