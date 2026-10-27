"""Fit per-participant epsilon with a discretized Beta prior: (tau, beta, a, b).

Each participant i has their own ε_i = (2/3)·ε'_i with ε'_i ~ Beta(a, b),
discretized on 21 nodes (pipeline.EPS_PRIME_NODES, bin masses from the Beta
CDF). ε' = 1 is uniform action noise, i.e. no updating (P_N = P_0), so the
prior-default type of the two-type mixture is the top node. Answers are scored
on the unmixed P_N with the λ = 0.01 lapse; tau and beta are shared.

    log L_i = logsumexp_k [log w_k(a, b) + Σ_q log p(r_q | ε_k, tau, beta)]

Fit: profile over (tau, beta). For each (tau, beta) the 102 x 21 node table is
computed once (~0.8 s) and (log a, log b) are maximized on it from several
starts (U-shaped, flat, humped, skewed). Coarse tau x beta grid, then
L-BFGS-B polish of (tau, beta).

Outputs (this folder): best_params_eps_pooled.json, participant_eps.csv.

Run from analysis/:
    uv run python experiments/2026-10-27-inference-soft-classifier/fit_eps_pooled.py
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
from pipeline import (  # noqa: E402
    EPS_NODES, eps_node_loglik_table, load_tagged_answers, pooled_loglik,
)

assert Path(pipeline.__file__).parent == SCRIPT_DIR, pipeline.__file__

TAU_VALS = np.linspace(0.5, 15.0, 8)
BETA_VALS = np.linspace(0.0, 1.0, 5)
TAU_BOUNDS = (0.05, 30.0)
BETA_BOUNDS = (0.0, 1.0)
LOG_AB_BOUNDS = (np.log(0.05), np.log(50.0))
AB_STARTS = [(0.5, 0.5), (1.0, 1.0), (3.0, 3.0), (1.0, 3.0), (3.0, 1.0)]
OUTPUT_PATH = SCRIPT_DIR / "best_params_eps_pooled.json"
EPS_PATH = SCRIPT_DIR / "participant_eps.csv"


def fit_ab(table, n):
    """Maximize pooled LL over (log a, log b) on a fixed node table."""
    def obj(x):
        return -pooled_loglik(table, np.exp(x[0]), np.exp(x[1]), n)["ll_per_answer"]

    best = None
    for a0, b0 in AB_STARTS:
        opt = minimize(obj, np.log([a0, b0]), method="L-BFGS-B",
                       bounds=[LOG_AB_BOUNDS, LOG_AB_BOUNDS],
                       options={"maxiter": 200, "ftol": 1e-12})
        if best is None or opt.fun < best.fun:
            best = opt
    return float(np.exp(best.x[0])), float(np.exp(best.x[1])), float(-best.fun)


def evaluate(prepared, answers, tau, beta):
    n = len(answers["participant"])
    table = eps_node_loglik_table(prepared, answers, float(tau), float(beta))
    a, b, ll = fit_ab(table, n)
    return {"ll_per_answer": ll, "ll_total": ll * n, "n": n,
            "tau_prior": float(tau), "beta": float(beta), "a": a, "b": b,
            "mean_eps": float((2.0 / 3.0) * a / (a + b))}


def main():
    print("=" * 66)
    print("Pooled-epsilon fit: tau, beta, a, b (discretized Beta prior on ε_i)")
    print("=" * 66)

    prepared, answers = load_tagged_answers()

    print("\n--- Coarse tau x beta grid ((a, b) maximized at each point) ---")
    coarse = []
    for tau in TAU_VALS:
        for beta in BETA_VALS:
            r = evaluate(prepared, answers, tau, beta)
            coarse.append(r)
            print(f"  tau={tau:6.3f} beta={beta:.2f}  LL={r['ll_per_answer']:.4f}  "
                  f"a={r['a']:.3f} b={r['b']:.3f}", flush=True)
    start = max(coarse, key=lambda r: r["ll_per_answer"])
    print(f"\nGrid best: tau={start['tau_prior']:.3f} beta={start['beta']:.2f} "
          f"LL={start['ll_per_answer']:.4f}")

    print("\n--- L-BFGS-B polish of (tau, beta) ---")
    opt = minimize(lambda x: -evaluate(prepared, answers, x[0], x[1])["ll_per_answer"],
                   [start["tau_prior"], start["beta"]], method="L-BFGS-B",
                   bounds=[TAU_BOUNDS, BETA_BOUNDS],
                   options={"maxiter": 60, "ftol": 1e-9})
    best = evaluate(prepared, answers, opt.x[0], opt.x[1])
    if best["ll_per_answer"] < start["ll_per_answer"]:
        best = start

    table = eps_node_loglik_table(prepared, answers, best["tau_prior"], best["beta"])
    m = pooled_loglik(table, best["a"], best["b"], best["n"])
    best["population_weights"] = pipeline.beta_grid_weights(best["a"], best["b"]).tolist()
    best["eps_nodes"] = EPS_NODES.tolist()
    best["inference_target"] = "unmixed"
    best["inference_lapse"] = pipeline.INFERENCE_LAPSE
    with open(OUTPUT_PATH, "w") as f:
        json.dump(best, f, indent=2)

    with open(EPS_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["participant_id", "n_answers", "post_mean_eps", "p_no_update"]
                   + [f"p_eps_{e:.4f}" for e in EPS_NODES])
        for i, pid in enumerate(answers["participant_ids"]):
            w.writerow([pid, len(answers["by_participant"][i]),
                        f"{m['post_mean_eps'][i]:.4f}", f"{m['p_no_update'][i]:.4f}"]
                       + [f"{p:.5f}" for p in m["node_post"][i]])

    print(f"\nBest: tau={best['tau_prior']:.4f} beta={best['beta']:.3f} "
          f"a={best['a']:.3f} b={best['b']:.3f} mean eps={best['mean_eps']:.3f} "
          f"LL={best['ll_per_answer']:.4f}")
    print(f"Saved to {OUTPUT_PATH} and {EPS_PATH}")


if __name__ == "__main__":
    main()
