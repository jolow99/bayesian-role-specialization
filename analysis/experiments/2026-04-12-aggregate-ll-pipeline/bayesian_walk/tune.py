"""Stage 2: Bayesian Walk — tunes tau_softmax, epsilon_switch on pooled agg_ll."""

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
    build_joint_dist,
)
from memory_strategies import strategy_from_params
from shared.inference import softmax_role_dist

CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = SCRIPT_DIR / "params.json"
METRIC = "agg_ll"


def predict_from_trajectory_walk(trajectory, values, tau_softmax, epsilon_switch):
    results = []
    for stage in trajectory:
        prior = stage["prior"]
        intent, thp, ehp = stage["intent"], stage["thp"], stage["ehp"]
        prev_roles = stage["prev_roles"]
        switch = [softmax_role_dist(i, intent, thp, ehp, prior, values, tau_softmax)
                  for i in range(3)]
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


def make_factory(tau_softmax, epsilon_switch):
    def factory(records_subset, traj_subset):
        idx_map = {id(r): i for i, r in enumerate(records_subset)}
        def predict_fn(record):
            i = idx_map[id(record)]
            return predict_from_trajectory_walk(
                traj_subset[i], record["env_config"]["values"],
                tau_softmax, epsilon_switch)
        return predict_fn
    return factory


def evaluate_pooled(records, trajectories, tau_softmax, epsilon_switch):
    factory = make_factory(tau_softmax, epsilon_switch)
    val = compute_pooled_metric(records, trajectories, factory, metric=METRIC)
    return {
        "tau_softmax": float(tau_softmax),
        "epsilon_switch": float(epsilon_switch),
        METRIC: val,
    }


def main():
    print("=" * 66)
    print("Stage 2: Bayesian Walk")
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
    done = get_completed_keys(results, ["tau_softmax", "epsilon_switch"])

    ts_vals = np.linspace(0.1, 20.0, 20)
    es_vals = np.linspace(0.0, 1.0, 21)
    total = len(ts_vals) * len(es_vals)
    print(f"\nCoarse: {len(ts_vals)} x {len(es_vals)} = {total} points (done {len(done)})")
    count = len(done)
    for ts in ts_vals:
        added = False
        for es in es_vals:
            if (float(ts), float(es)) in done:
                continue
            r = evaluate_pooled(records, trajectories, ts, es)
            results.append(r)
            added = True
            count += 1
        if added:
            save_checkpoint(coarse_path, results)
            print(f"  [{count}/{total}] ...", flush=True)

    best = pick_best(results, METRIC)
    print(f"Coarse best: tau_softmax={best['tau_softmax']:.4f} "
          f"epsilon_switch={best['epsilon_switch']:.4f} {METRIC}={best[METRIC]:.4f}")

    refined_path = str(CHECKPOINT_DIR / "refined_results.json")
    refined = load_checkpoint(refined_path)
    done_r = get_completed_keys(refined, ["tau_softmax", "epsilon_switch"])
    ts_step = (20.0 - 0.1) / 19
    es_step = 1.0 / 20
    fine_ts = np.linspace(max(0.1, best["tau_softmax"] - ts_step),
                           min(20.0, best["tau_softmax"] + ts_step), 11)
    fine_es = np.linspace(max(0.0, best["epsilon_switch"] - es_step),
                           min(1.0, best["epsilon_switch"] + es_step), 11)
    print(f"\nRefined: {len(fine_ts)} x {len(fine_es)} = {len(fine_ts)*len(fine_es)} points")
    for ts in fine_ts:
        added = False
        for es in fine_es:
            if (float(ts), float(es)) in done_r:
                continue
            r = evaluate_pooled(records, trajectories, ts, es)
            refined.append(r)
            added = True
        if added:
            save_checkpoint(refined_path, refined)

    all_results = results + refined
    best = pick_best(all_results, METRIC)

    def obj(params):
        return -evaluate_pooled(records, trajectories, params[0], params[1])[METRIC]

    print(f"\nPolish from ts={best['tau_softmax']:.4f} es={best['epsilon_switch']:.4f}")
    opt = minimize(obj, [best["tau_softmax"], best["epsilon_switch"]],
                   method="L-BFGS-B", bounds=[(0.01, 50.0), (0.0, 1.0)],
                   options={"maxiter": 50, "ftol": 1e-6})
    r = evaluate_pooled(records, trajectories, float(opt.x[0]), float(opt.x[1]))
    all_results.append(r)
    best = pick_best(all_results, METRIC)

    eval_blocks = compute_disaggregated_metrics(
        records, trajectories,
        make_factory(best["tau_softmax"], best["epsilon_switch"]))

    output = {
        "metric_optimized": METRIC,
        "stage1_params": {
            "tau_prior": s1["tau_prior"], "epsilon": s1["epsilon"],
            "memory_strategy": strat.name,
        },
        "tuned_params": {
            "tau_softmax": float(best["tau_softmax"]),
            "epsilon_switch": float(best["epsilon_switch"]),
        },
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
