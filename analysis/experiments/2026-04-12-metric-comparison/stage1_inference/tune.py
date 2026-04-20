"""Stage 1 (human-only): tune (tau_prior, epsilon, memory_strategy) on inference data.

Thin wrapper around the 04-12 pipeline's Stage 1. Two differences:
1. Only loads the 03-06 and 03-18 exports (excludes 02-13 pilot).
2. Filters to human-origin records only.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent

# Import from the 04-12 pipeline.
PIPELINE_DIR = EXPERIMENT_DIR.parent / "2026-04-12-aggregate-ll-pipeline"
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(PIPELINE_DIR / "stage1_inference"))

from stage1_inference.tune import (
    _prepare_human_team,
    _prepare_bot_record,
    evaluate,
    compute_posteriors,
)
from shared_utils import load_checkpoint, save_checkpoint, get_completed_keys, pick_best
from memory_strategies import MemoryStrategy, build_strategy_grid, apply_boundary

from shared import EXPORTS_DIR
from shared.data_loading import load_all_exports

import numpy as np
from collections import defaultdict
from scipy.optimize import minimize

# Override checkpoint/output paths to write into THIS experiment's directory.
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
COARSE_PATH = CHECKPOINT_DIR / "coarse_results.json"
REFINED_PATH = CHECKPOINT_DIR / "refined_results.json"
POLISHED_PATH = CHECKPOINT_DIR / "polished_results.json"
OUTPUT_PATH = SCRIPT_DIR / "best_inference_params.json"

# Only use the two later exports (exclude 02-13 pilot).
DATA_DIRS = [
    EXPORTS_DIR / "bayesian-role-specialization-2026-03-06-09-54-19",
    EXPORTS_DIR / "bayesian-role-specialization-2026-03-18-15-47-09",
]


def load_prepared():
    """Load 03-06 + 03-18 exports, human-origin only."""
    records = load_all_exports(data_dirs=DATA_DIRS)
    human_teams = defaultdict(list)
    for pr in records:
        if pr.is_dropout:
            continue
        if pr.round.round_type == "human":
            human_teams[(pr.game_id, pr.round.round_number)].append(pr)
    human_teams = {k: sorted(v, key=lambda p: p.player_id)
                   for k, v in human_teams.items() if len(v) == 3}
    print(f"Loaded {len(human_teams)} human teams (human-only, 03-06 + 03-18)")
    prepared = [_prepare_human_team(t) for t in human_teams.values()]
    return prepared


# ──────────────────────────────────────────────────────────────────────
# Search phases (same logic as 04-12 but with local checkpoint paths)
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
                r = evaluate(prepared, float(tp), float(eps), strat)
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
                  f"(H={best['human_inference_ll']:.3f}) "
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
            r = evaluate(prepared, float(tp), float(eps), strat)
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
        return -evaluate(prepared, params[0], params[1], strat)["inference_ll"]

    opt = minimize(obj, [best["tau_prior"], best["epsilon"]],
                   method="L-BFGS-B", bounds=[(0.05, 30.0), (0.001, 0.999)],
                   options={"maxiter": 50, "ftol": 1e-6})
    r = evaluate(prepared, float(opt.x[0]), float(opt.x[1]), strat)
    r.update({
        "strategy_name": strat.name, "kind": strat.kind, "param": strat.param,
        "tau_prior": float(opt.x[0]), "epsilon": float(opt.x[1]),
    })
    save_checkpoint(str(POLISHED_PATH), [r])
    return r


def main():
    print("=" * 66)
    print("Stage 1 (human-only): inference parameter tuning")
    print("=" * 66)

    prepared = load_prepared()
    n_queries = sum(len(d["queries"]) for d in prepared)
    n_bot_queries = sum(len(d["queries"]) for d in prepared if d["origin"] == "bot")
    n_human_queries = n_queries - n_bot_queries
    print(f"Prepared {len(prepared)} records, {n_queries} inference queries "
          f"({n_human_queries} human, {n_bot_queries} bot)")

    assert n_bot_queries == 0, (
        f"Expected 0 bot queries in human-only mode, got {n_bot_queries}"
    )
    print(f"  [OK] human-only queries = {n_human_queries} (0 bot)")

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
        "human_inference_ll": best["human_inference_ll"],
        "bot_inference_ll": best["bot_inference_ll"],
        "n": best["n"],
        "n_human": best["n_human"],
        "n_bot": best["n_bot"],
        "floor_hits": best["floor_hits"],
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nBest: {best['strategy_name']} tau={best['tau_prior']:.4f} "
          f"eps={best['epsilon']:.4f} LL={best['inference_ll']:.4f}")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
