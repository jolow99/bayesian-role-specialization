"""Stage 2: Bayesian Belief — no tunable params, baseline evaluation.

Fits on pooled (human + bot) records, reports disaggregated eval.
"""

import sys
import json
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from stage2_common import (
    load_stage1_params, load_records, precompute_trajectories,
    compute_disaggregated_metrics, build_joint_dist, posterior_marginal,
)
from memory_strategies import strategy_from_params

OUTPUT_PATH = SCRIPT_DIR / "params.json"
METRIC = "agg_ll"


def predict_from_trajectory_belief(trajectory):
    results = []
    for stage in trajectory:
        prior = stage["prior"]
        per_agent = [posterior_marginal(prior, i) for i in range(3)]
        results.append({
            "predicted_dist": build_joint_dist(per_agent),
            "human_combo": stage["human_combo"],
            "model_marginal": np.mean(per_agent, axis=0),
        })
    return results


def make_predict_fn(records_subset, traj_subset):
    idx_map = {id(r): i for i, r in enumerate(records_subset)}
    def predict_fn(record):
        return predict_from_trajectory_belief(traj_subset[idx_map[id(record)]])
    return predict_fn


def main():
    print("=" * 66)
    print("Stage 2: Bayesian Belief (pooled fit, disaggregated eval)")
    print("=" * 66)

    s1 = load_stage1_params(EXPERIMENT_DIR)
    strat = strategy_from_params(
        s1.get("memory_strategy", "full"),
        s1.get("window"),
        s1.get("drift_delta", 0.0),
    )
    print(f"Stage 1: tau_prior={s1['tau_prior']:.4f} epsilon={s1['epsilon']:.6f} "
          f"strategy={strat.name}")

    records = load_records(include_bot_rounds=True)
    print("Precomputing trajectories...")
    trajectories = precompute_trajectories(
        records, s1["tau_prior"], s1["epsilon"], memory_strategy=strat)

    eval_blocks = compute_disaggregated_metrics(
        records, trajectories, make_predict_fn)

    output = {
        "metric_optimized": METRIC,
        "stage1_params": {
            "tau_prior": s1["tau_prior"], "epsilon": s1["epsilon"],
            "memory_strategy": strat.name,
        },
        "tuned_params": {},
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
