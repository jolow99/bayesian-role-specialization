"""Stage 1 refit: (beta, tau_prior, epsilon) with inference scored on the unmixed posterior.

Uses this folder's pipeline.py, where compute_posteriors stores each stage's
posterior before the drift_prior boundary (P_N), while the mixed belief
(1 - beta) * P_N + beta * P_0 is still carried into the next stage.

Human-only, clean-teams-only. Objective: per-query mean log-likelihood of
the human's inferred role under the target's marginal, with a λ = 0.01
lapse (1 − λ)·p + λ/3 and no floor (05-25 tune.py used a 1e-20 floor).

1. Beta profile: for each beta on a grid, coarse tau x eps grid (same ranges
   as tune.py) then L-BFGS-B polish of (tau, eps).
2. Joint L-BFGS-B over (tau, eps, beta) from the best profile point; beta is
   rounded to 3 decimals and (tau, eps) re-polished at that value.

Outputs (this folder): beta_profile.json, best_inference_params.json (same
schema as 05-25 stage1_inference/best_inference_params.json).

Run from analysis/:
    uv run python experiments/2026-10-27-inference-on-unmixed-posterior/fit_stage1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import pipeline  # noqa: E402
from pipeline import MemoryStrategy, load_human_inference_records, stage1_evaluate  # noqa: E402

assert Path(pipeline.__file__).parent == SCRIPT_DIR, pipeline.__file__

BETA_VALS = np.round(np.linspace(0.0, 1.0, 21), 3)
TAU_VALS = np.linspace(0.5, 15.0, 8)
EPS_VALS = np.linspace(0.05, 0.9, 8)
TAU_BOUNDS = (0.05, 30.0)
EPS_BOUNDS = (0.001, 0.999)
PROFILE_PATH = SCRIPT_DIR / "beta_profile.json"
OUTPUT_PATH = SCRIPT_DIR / "best_inference_params.json"


def drift(beta):
    beta = float(beta)
    return MemoryStrategy(f"drift_prior_{beta:.3f}", "drift_prior", beta)


def evaluate(prepared, tau, eps, beta):
    r = stage1_evaluate(prepared, float(tau), float(eps), drift(beta))
    r.update({"tau_prior": float(tau), "epsilon": float(eps), "beta": float(beta)})
    return r


def polish_tau_eps(prepared, beta, tau0, eps0):
    strat = drift(beta)

    def obj(params):
        return -stage1_evaluate(prepared, params[0], params[1], strat)["inference_ll"]

    opt = minimize(obj, [tau0, eps0], method="L-BFGS-B",
                   bounds=[TAU_BOUNDS, EPS_BOUNDS],
                   options={"maxiter": 100, "ftol": 1e-9})
    return evaluate(prepared, opt.x[0], opt.x[1], beta)


def profile_beta(prepared, beta):
    coarse = [evaluate(prepared, tp, eps, beta) for tp in TAU_VALS for eps in EPS_VALS]
    best = max(coarse, key=lambda r: r["inference_ll"])
    return polish_tau_eps(prepared, beta, best["tau_prior"], best["epsilon"])


def fit_joint(prepared, start):
    def obj(params):
        return -evaluate(prepared, *params)["inference_ll"]

    opt = minimize(obj, [start["tau_prior"], start["epsilon"], start["beta"]],
                   method="L-BFGS-B", bounds=[TAU_BOUNDS, EPS_BOUNDS, (0.0, 1.0)],
                   options={"maxiter": 100, "ftol": 1e-9})
    beta = round(float(opt.x[2]), 3)
    return polish_tau_eps(prepared, beta, opt.x[0], opt.x[1])


def main():
    print("=" * 66)
    print("Stage 1 refit: beta, tau_prior, epsilon (inference on unmixed P_N)")
    print("=" * 66)

    prepared = load_human_inference_records()

    print("\n--- Beta profile (tau, eps polished at each beta) ---")
    profile = []
    for beta in BETA_VALS:
        r = profile_beta(prepared, beta)
        profile.append(r)
        print(f"  beta={beta:.3f}  LL={r['inference_ll']:.4f}  acc={r['accuracy']:.4f}  "
              f"tau={r['tau_prior']:.3f}  eps={r['epsilon']:.4f}", flush=True)
    with open(PROFILE_PATH, "w") as f:
        json.dump(profile, f, indent=2)

    start = max(profile, key=lambda r: r["inference_ll"])
    print(f"\nProfile best: beta={start['beta']:.3f} LL={start['inference_ll']:.4f}")

    print("\n--- Joint L-BFGS-B over (tau, eps, beta) ---")
    best = fit_joint(prepared, start)
    if best["inference_ll"] < start["inference_ll"]:
        best = start
    strat = drift(best["beta"])

    output = {
        "tau_prior": best["tau_prior"],
        "epsilon": best["epsilon"],
        "memory_strategy": strat.name,
        "kind": strat.kind,
        "param": strat.param,
        "window": None,
        "drift_delta": strat.param,
        "accuracy": best["accuracy"],
        "inference_ll": best["inference_ll"],
        "n": best["n"],
        "inference_target": "unmixed",
        "inference_lapse": pipeline.INFERENCE_LAPSE,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nBest: beta={strat.param:.3f} tau={best['tau_prior']:.4f} "
          f"eps={best['epsilon']:.4f} LL={best['inference_ll']:.4f} "
          f"acc={best['accuracy']:.4f}")
    print(f"Saved to {OUTPUT_PATH} and {PROFILE_PATH}")


if __name__ == "__main__":
    main()
