"""Stage 1: tune (tau_prior, epsilon, memory_strategy) on the 04-23 export.

Human-only, clean-teams-only. No imports from any other experiment.

Objective: per-query mean log-likelihood of human inferred roles under the
marginal of the Bayesian posterior at inference time.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from pipeline import (
    MemoryStrategy, build_strategy_grid, load_human_inference_records,
    stage1_evaluate, load_checkpoint, save_checkpoint, get_completed_keys,
    pick_best,
)

CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
COARSE_PATH = CHECKPOINT_DIR / "coarse_results.json"
REFINED_PATH = CHECKPOINT_DIR / "refined_results.json"
POLISHED_PATH = CHECKPOINT_DIR / "polished_results.json"
OUTPUT_PATH = SCRIPT_DIR / "best_inference_params.json"


# ──────────────────────────────────────────────────────────────────────
# Search phases
# ──────────────────────────────────────────────────────────────────────

def run_coarse(prepared):
    strategies = build_strategy_grid()
    tau_vals = np.linspace(0.5, 15.0, 8)
    eps_vals = np.linspace(0.05, 0.9, 8)

    results = load_checkpoint(str(COARSE_PATH))
    done = get_completed_keys(results, ["strategy_name", "tau_prior", "epsilon"])
    total = len(strategies) * len(tau_vals) * len(eps_vals)
    print(f"Coarse: {len(strategies)} strategies x {len(tau_vals)} tau "
          f"x {len(eps_vals)} eps = {total} points (done {len(done)})")

    count = len(done)
    for si, strat in enumerate(strategies):
        added = False
        for tp in tau_vals:
            for eps in eps_vals:
                key = (strat.name, float(tp), float(eps))
                if key in done:
                    continue
                r = stage1_evaluate(prepared, float(tp), float(eps), strat)
                r.update({
                    "strategy_name": strat.name,
                    "kind": strat.kind,
                    "param": strat.param,
                    "tau_prior": float(tp),
                    "epsilon": float(eps),
                })
                results.append(r)
                added = True
                count += 1
        if added:
            save_checkpoint(str(COARSE_PATH), results)
            rs = [r for r in results if r["strategy_name"] == strat.name]
            best = max(rs, key=lambda x: x["inference_ll"])
            print(f"  [{si+1}/{len(strategies)}] {strat.name:<22} "
                  f"LL={best['inference_ll']:.4f} "
                  f"tp={best['tau_prior']:.2f} eps={best['epsilon']:.2f} "
                  f"[{count}/{total}]", flush=True)
    return results


def run_refined(prepared, coarse_results):
    best = pick_best(coarse_results, "inference_ll")
    print(f"\nCoarse best: strategy={best['strategy_name']} "
          f"tau={best['tau_prior']:.3f} eps={best['epsilon']:.3f} "
          f"LL={best['inference_ll']:.4f}")

    strat = MemoryStrategy(best["strategy_name"], best["kind"], best["param"])
    tau_step = (15.0 - 0.5) / 7
    eps_step = (0.9 - 0.05) / 7
    tau_vals = np.linspace(max(0.05, best["tau_prior"] - tau_step),
                            min(30.0, best["tau_prior"] + tau_step), 11)
    eps_vals = np.linspace(max(0.001, best["epsilon"] - eps_step),
                            min(0.999, best["epsilon"] + eps_step), 11)

    results = load_checkpoint(str(REFINED_PATH))
    done = get_completed_keys(results, ["strategy_name", "tau_prior", "epsilon"])
    total = len(tau_vals) * len(eps_vals)
    print(f"Refined: {len(tau_vals)} x {len(eps_vals)} = {total} points (done {len(done)})")

    count = len(done)
    for tp in tau_vals:
        added = False
        for eps in eps_vals:
            key = (strat.name, float(tp), float(eps))
            if key in done:
                continue
            r = stage1_evaluate(prepared, float(tp), float(eps), strat)
            r.update({
                "strategy_name": strat.name,
                "kind": strat.kind,
                "param": strat.param,
                "tau_prior": float(tp),
                "epsilon": float(eps),
            })
            results.append(r)
            added = True
            count += 1
        if added:
            save_checkpoint(str(REFINED_PATH), results)
            print(f"  [{count}/{total}] ...", flush=True)
    return results


def run_polish(prepared, all_results):
    best = pick_best(all_results, "inference_ll")
    strat = MemoryStrategy(best["strategy_name"], best["kind"], best["param"])
    print(f"\nPolish from: strategy={strat.name} "
          f"tau={best['tau_prior']:.4f} eps={best['epsilon']:.4f}")

    def obj(params):
        return -stage1_evaluate(prepared, params[0], params[1], strat)["inference_ll"]

    opt = minimize(obj, [best["tau_prior"], best["epsilon"]],
                   method="L-BFGS-B", bounds=[(0.05, 30.0), (0.001, 0.999)],
                   options={"maxiter": 50, "ftol": 1e-6})
    r = stage1_evaluate(prepared, float(opt.x[0]), float(opt.x[1]), strat)
    r.update({
        "strategy_name": strat.name, "kind": strat.kind, "param": strat.param,
        "tau_prior": float(opt.x[0]), "epsilon": float(opt.x[1]),
    })
    save_checkpoint(str(POLISHED_PATH), [r])
    return r


def main():
    print("=" * 66)
    print("Stage 1 (04-23, human-only, clean teams): inference tuning")
    print("=" * 66)

    prepared = load_human_inference_records()
    n_queries = sum(len(d["queries"]) for d in prepared)
    print(f"Prepared {len(prepared)} records, {n_queries} inference queries")

    def top5(results):
        by_strat = defaultdict(list)
        for r in results:
            by_strat[r["strategy_name"]].append(r)
        best_per = [max(rs, key=lambda x: x["inference_ll"]) for rs in by_strat.values()]
        best_per.sort(key=lambda r: -r["inference_ll"])
        return best_per[:5]

    print("\n--- Phase 1: Coarse grid ---")
    coarse_results = run_coarse(prepared)
    print("\n--- Phase 2: Refined grid ---")
    refined_results = run_refined(prepared, coarse_results)

    all_results = coarse_results + refined_results
    print("\n--- Phase 3: L-BFGS-B polish ---")
    polished = run_polish(prepared, all_results)
    all_results.append(polished)

    best = pick_best(all_results, "inference_ll")
    print("\nTop-5 strategies:")
    for r in top5(all_results):
        print(f"  {r['strategy_name']:<24} LL={r['inference_ll']:.4f} "
              f"tau={r['tau_prior']:.3f} eps={r['epsilon']:.3f}")

    output = {
        "tau_prior": best["tau_prior"],
        "epsilon": best["epsilon"],
        "memory_strategy": best["strategy_name"],
        "kind": best["kind"],
        "param": best["param"],
        "window": best["param"] if best["kind"] == "window" else None,
        "drift_delta": best["param"] if best["kind"] == "drift_prior" else 0.0,
        "accuracy": best["accuracy"],
        "inference_ll": best["inference_ll"],
        "n": best["n"],
        "floor_hits": best["floor_hits"],
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nBest: {best['strategy_name']} tau={best['tau_prior']:.4f} "
          f"eps={best['epsilon']:.4f} LL={best['inference_ll']:.4f}")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
