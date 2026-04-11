"""Refined sweep around the tempering winner.

Denser γ grid (0.30..0.60 in 0.02 steps), denser (tau_prior, epsilon) grid
around the coarse-sweep winner (3.4, 0.42), followed by L-BFGS-B polish on
all three params jointly.
"""

import sys
import json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import tune

REFINED_PATH = SCRIPT_DIR / 'refined_results.json'
REFINED_SUMMARY_PATH = SCRIPT_DIR / 'refined_summary.json'


def main():
    print("=" * 60)
    print("Refined sweep: tempering near γ=0.45")
    print("=" * 60)

    prepared = tune.load_prepared()

    gamma_vals = np.round(np.linspace(0.30, 0.60, 16), 3)
    tau_vals = np.round(np.linspace(1.5, 6.0, 10), 3)
    eps_vals = np.round(np.linspace(0.20, 0.65, 10), 3)

    total = len(gamma_vals) * len(tau_vals) * len(eps_vals)
    print(f"Grid: {len(gamma_vals)} γ x {len(tau_vals)} τ x {len(eps_vals)} ε "
          f"= {total} points", flush=True)

    results = []
    if REFINED_PATH.exists():
        with open(REFINED_PATH) as f:
            results = json.load(f)
    done = {(r['param'], r['tau_prior'], r['epsilon']) for r in results}

    count = len(done)
    for gi, g in enumerate(gamma_vals):
        strat = {'name': f'temper_{g:.3f}', 'kind': 'temper', 'param': float(g)}
        added = False
        for tp in tau_vals:
            for eps in eps_vals:
                key = (float(g), float(tp), float(eps))
                if key in done:
                    continue
                res = tune.evaluate(prepared, float(tp), float(eps), strat)
                results.append({
                    'strategy': strat['name'],
                    'kind': 'temper',
                    'param': float(g),
                    'tau_prior': float(tp),
                    'epsilon': float(eps),
                    **res,
                })
                count += 1
                added = True
        if added:
            with open(REFINED_PATH, 'w') as f:
                json.dump(results, f, indent=2)
        rs = [r for r in results if abs(r['param'] - float(g)) < 1e-9]
        best = max(rs, key=lambda r: r['inference_ll'])
        print(f"  [{gi+1}/{len(gamma_vals)}] γ={g:.3f}  "
              f"LL={best['inference_ll']:.5f}  acc={best['accuracy']:.3f}  "
              f"tp={best['tau_prior']:.2f}  eps={best['epsilon']:.2f}",
              flush=True)

    # L-BFGS-B polish on (γ, τ_prior, ε)
    grid_best = max(results, key=lambda r: r['inference_ll'])
    print(f"\nGrid best: γ={grid_best['param']:.4f}  "
          f"τ={grid_best['tau_prior']:.4f}  ε={grid_best['epsilon']:.4f}  "
          f"LL={grid_best['inference_ll']:.5f}", flush=True)

    def objective(params):
        g, tp, eps = params
        strat = {'name': 'polish', 'kind': 'temper', 'param': float(g)}
        return -tune.evaluate(prepared, float(tp), float(eps), strat)['inference_ll']

    x0 = [grid_best['param'], grid_best['tau_prior'], grid_best['epsilon']]
    bounds = [(0.05, 0.95), (0.1, 20.0), (0.01, 0.99)]
    print("\nL-BFGS-B polishing...", flush=True)
    opt = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 80, 'ftol': 1e-7})
    opt_g, opt_tp, opt_eps = opt.x
    opt_strat = {'name': f'temper_{opt_g:.4f}_polished',
                 'kind': 'temper', 'param': float(opt_g)}
    opt_res = tune.evaluate(prepared, float(opt_tp), float(opt_eps), opt_strat)
    polished = {
        'strategy': opt_strat['name'],
        'kind': 'temper',
        'param': float(opt_g),
        'tau_prior': float(opt_tp),
        'epsilon': float(opt_eps),
        **opt_res,
    }
    print(f"Polished: γ={opt_g:.5f}  τ={opt_tp:.5f}  ε={opt_eps:.5f}  "
          f"LL={polished['inference_ll']:.5f}  acc={polished['accuracy']:.3f}",
          flush=True)

    final = grid_best if grid_best['inference_ll'] > polished['inference_ll'] else polished
    summary = {
        'grid_best': grid_best,
        'polished': polished,
        'final': final,
        'coarse_stage1_winner_reference': {
            'strategy': 'window_1', 'tau_prior': 6.8031, 'epsilon': 0.5614,
            'inference_ll': -1.00959, 'accuracy': 0.9265, 'n': 1374,
        },
    }
    with open(REFINED_SUMMARY_PATH, 'w') as f:
        json.dump(summary, f, indent=2)

    # Top 10 from the refined grid
    results_sorted = sorted(results, key=lambda r: -r['inference_ll'])[:10]
    print(f"\n{'='*60}\nTop 10 refined grid points:")
    print(f"{'γ':>8}{'τ_prior':>10}{'ε':>8}{'LL':>11}{'acc':>8}")
    for r in results_sorted:
        print(f"{r['param']:>8.3f}{r['tau_prior']:>10.3f}{r['epsilon']:>8.3f}"
              f"{r['inference_ll']:>11.5f}{r['accuracy']:>8.3f}")

    print(f"\nFinal winner: γ={final['param']:.4f}  "
          f"τ_prior={final['tau_prior']:.4f}  ε={final['epsilon']:.4f}")
    print(f"  LL={final['inference_ll']:.5f}  acc={final['accuracy']:.3f}")
    print(f"  vs stage1 window_1: ΔLL = "
          f"{final['inference_ll'] - (-1.00959):+.5f} nats/query")


if __name__ == '__main__':
    main()
