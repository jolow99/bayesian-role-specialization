"""Stage 2: Bayesian Walk PS — tunes epsilon_switch on pooled agg_ll."""

import sys
import json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from shared_utils import load_checkpoint, save_checkpoint, get_completed_keys, pick_best
from stage2_common import (
    load_stage1_params, load_records, precompute_trajectories,
    compute_pooled_metric, compute_disaggregated_metrics,
    build_joint_dist, posterior_marginal,
)
from memory_strategies import strategy_from_params

CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = SCRIPT_DIR / "params.json"
METRIC = "agg_ll"


def predict_from_trajectory_walk_ps(trajectory, epsilon_switch):
    results = []
    for stage in trajectory:
        prior = stage["prior"]
        prev_roles = stage["prev_roles"]
        switch = [posterior_marginal(prior, i) for i in range(3)]
        per_agent = []
        for i in range(3):
            if prev_roles is None:
                per_agent.append(switch[i])
            else:
                stick = np.zeros(3)
                stick[prev_roles[i]] = 1.0
                per_agent.append((1.0 - epsilon_switch) * stick
                                 + epsilon_switch * switch[i])
        results.append({
            "predicted_dist": build_joint_dist(per_agent),
            "human_combo": stage["human_combo"],
            "model_marginal": np.mean(per_agent, axis=0),
        })
    return results


def make_factory(epsilon_switch):
    def factory(records_subset, traj_subset):
        idx_map = {id(r): i for i, r in enumerate(records_subset)}
        def predict_fn(record):
            i = idx_map[id(record)]
            return predict_from_trajectory_walk_ps(traj_subset[i], epsilon_switch)
        return predict_fn
    return factory


def evaluate_pooled(records, trajectories, epsilon_switch):
    factory = make_factory(epsilon_switch)
    val = compute_pooled_metric(records, trajectories, factory, metric=METRIC)
    return {"epsilon_switch": float(epsilon_switch), METRIC: val}


def main():
    print("=" * 66)
    print("Stage 2: Bayesian Walk PS")
    print("=" * 66)

    s1 = load_stage1_params(EXPERIMENT_DIR)
    strat = strategy_from_params(
        s1.get("memory_strategy", "full"), s1.get("window"),
        s1.get("drift_delta", 0.0))
    print(f"Stage 1: tau_prior={s1['tau_prior']:.4f} epsilon={s1['epsilon']:.6f} "
          f"strategy={strat.name}")

    records = load_records(include_bot_rounds=True)
    print("Precomputing trajectories...")
    trajectories = precompute_trajectories(
        records, s1["tau_prior"], s1["epsilon"], memory_strategy=strat)

    coarse_path = str(CHECKPOINT_DIR / "coarse_results.json")
    results = load_checkpoint(coarse_path)
    done = get_completed_keys(results, ["epsilon_switch"])
    es_vals = np.linspace(0.0, 1.0, 21)
    print(f"\nCoarse: {len(es_vals)} points (done {len(done)})")
    for es in es_vals:
        if (float(es),) in done:
            continue
        r = evaluate_pooled(records, trajectories, es)
        results.append(r)
        save_checkpoint(coarse_path, results)

    best = pick_best(results, METRIC)
    print(f"Coarse best: eps_switch={best['epsilon_switch']:.4f} "
          f"{METRIC}={best[METRIC]:.4f}")

    refined_path = str(CHECKPOINT_DIR / "refined_results.json")
    refined = load_checkpoint(refined_path)
    done_r = get_completed_keys(refined, ["epsilon_switch"])
    step = 1.0 / 20
    fine_es = np.linspace(max(0.0, best["epsilon_switch"] - step),
                           min(1.0, best["epsilon_switch"] + step), 11)
    print(f"\nRefined: {len(fine_es)} points")
    for es in fine_es:
        if (float(es),) in done_r:
            continue
        r = evaluate_pooled(records, trajectories, es)
        refined.append(r)
        save_checkpoint(refined_path, refined)

    all_results = results + refined
    best = pick_best(all_results, METRIC)

    def obj(params):
        return -evaluate_pooled(records, trajectories, params[0])[METRIC]

    print(f"\nPolish from es={best['epsilon_switch']:.4f}")
    opt = minimize(obj, [best["epsilon_switch"]], method="L-BFGS-B",
                   bounds=[(0.0, 1.0)],
                   options={"maxiter": 50, "ftol": 1e-6})
    r = evaluate_pooled(records, trajectories, float(opt.x[0]))
    all_results.append(r)
    best = pick_best(all_results, METRIC)

    eval_blocks = compute_disaggregated_metrics(
        records, trajectories, make_factory(best["epsilon_switch"]))

    output = {
        "metric_optimized": METRIC,
        "stage1_params": {
            "tau_prior": s1["tau_prior"], "epsilon": s1["epsilon"],
            "memory_strategy": strat.name,
        },
        "tuned_params": {"epsilon_switch": float(best["epsilon_switch"])},
        "eval": eval_blocks,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")
    for subset, m in eval_blocks.items():
        print(f"  {subset}: n={m['n_records']} agg_ll={m['agg_ll']:.4f} "
              f"combo_r={m['combo_r']:.4f} mean_ll={m['mean_ll']:.4f}")


if __name__ == "__main__":
    main()
