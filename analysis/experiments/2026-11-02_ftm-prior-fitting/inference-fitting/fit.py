"""Two-type classifier with a learned (p_F, p_T, p_M) non-updater class.

    L_i = (1 − θ)·∏_q P_N(r_q) + θ·∏_q [(1 − λ)·p_{r_q} + λ/3]

Updaters: stat-based prior at tau, unmixed P_N, drift_prior beta, lapse λ.
Non-updaters: (p_F, p_T, p_M) = softmax(z_F, z_T, 0), ignoring stats and target.
For each (tau, eps, beta), (θ, z_F, z_T) are maximized on the participants'
F/T/M counts. Coarse grid, then L-BFGS-B polish of (tau, eps, beta).

Run from analysis/:
    uv run python experiments/2026-11-02_ftm-prior-fitting/inference-fitting/fit.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_expit, log_softmax

from pipeline import INFERENCE_LAPSE, answer_probs, load_answers, participant_logprob

OUT_DIR = Path(__file__).resolve().parent
TAU_VALS = np.linspace(0.5, 15.0, 8)
EPS_VALS = np.linspace(0.05, 0.9, 8)
BETA_VALS = np.linspace(0.0, 1.0, 5)
OUTER_BOUNDS = [(0.05, 30.0), (0.001, 0.999), (0.0, 1.0)]  # tau, eps, beta
INNER_BOUNDS = [(-10.0, 10.0)] * 3  # logit θ, z_F, z_T


def role_probs(z):
    return np.exp(log_softmax(np.r_[z[0], z[1], 0.0]))


def mixture_terms(x, log_B, counts):
    """x = (logit θ, z_F, z_T) -> per-participant log L_i, log D_i, P(non-updater)."""
    p = role_probs(x[1:])
    log_D = counts @ np.log((1.0 - INFERENCE_LAPSE) * p + INFERENCE_LAPSE / 3.0)
    log_L = np.logaddexp(log_expit(-x[0]) + log_B, log_expit(x[0]) + log_D)
    return log_L, log_D, np.exp(log_expit(x[0]) + log_D - log_L)


def evaluate(teams, answers, counts, tau, eps, beta):
    log_B = participant_logprob(answer_probs(teams, answers, tau, eps, beta), answers)
    n = int(counts.sum())
    best = min((minimize(lambda x: -mixture_terms(x, log_B, counts)[0].sum() / n, x0,
                         method="L-BFGS-B", bounds=INNER_BOUNDS,
                         options={"maxiter": 500, "ftol": 1e-13, "gtol": 1e-10})
                for x0 in [(0.0, 0.0, 0.0), (-2.0, 0.0, 0.0), (2.0, 0.0, 0.0)]),
               key=lambda o: o.fun)
    return {"ll": -best.fun, "tau": tau, "eps": eps, "beta": beta, "x": best.x, "log_B": log_B}


def main():
    teams, answers = load_answers()
    counts = np.array([np.bincount(answers["inferred"][ix], minlength=3)
                       for ix in answers["by_participant"]])

    grid = [evaluate(teams, answers, counts, t, e, b)
            for t in TAU_VALS for e in EPS_VALS for b in BETA_VALS]
    start = max(grid, key=lambda r: r["ll"])
    print(f"Grid best: tau={start['tau']:.3f} eps={start['eps']:.3f} "
          f"beta={start['beta']:.2f} LL={start['ll']:.4f}", flush=True)

    opt = minimize(lambda v: -evaluate(teams, answers, counts, *v)["ll"],
                   [start["tau"], start["eps"], start["beta"]],
                   method="L-BFGS-B", bounds=OUTER_BOUNDS, options={"maxiter": 100, "ftol": 1e-9})
    best = max([start, evaluate(teams, answers, counts, *opt.x)], key=lambda r: r["ll"])

    _, log_D, p_nu = mixture_terms(best["x"], best["log_B"], counts)
    p = role_probs(best["x"][1:])
    params = {"tau_prior": float(best["tau"]), "epsilon": float(best["eps"]),
              "beta": float(best["beta"]), "theta": float(np.exp(log_expit(best["x"][0]))),
              "p_F": float(p[0]), "p_T": float(p[1]), "p_M": float(p[2]),
              "inference_lapse": INFERENCE_LAPSE,
              "ll_per_answer": float(best["ll"]), "n_answers": int(counts.sum()),
              "n_participants": len(counts)}
    with open(OUT_DIR / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    with open(OUT_DIR / "participant_types.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["participant_id", "n_answers", "n_F", "n_T", "n_M",
                    "log_B", "log_D", "p_non_updater"])
        for i, pid in enumerate(answers["participant_ids"]):
            w.writerow([pid, counts[i].sum(), *counts[i],
                        f"{best['log_B'][i]:.4f}", f"{log_D[i]:.4f}", f"{p_nu[i]:.4f}"])

    print(json.dumps(params, indent=2))
    print(f"Non-updaters (P > 0.5): {(p_nu > 0.5).sum()} of {len(counts)}")


if __name__ == "__main__":
    main()
