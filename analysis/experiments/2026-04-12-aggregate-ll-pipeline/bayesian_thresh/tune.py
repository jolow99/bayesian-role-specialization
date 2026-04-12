"""Stage 2: Bayesian Threshold — tunes tau_softmax, delta on pooled agg_ll."""

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


def expected_values_per_role(agent_i, intent, thp, ehp, prior, values):
    other = [a for a in range(3) if a != agent_i]
    other_probs = np.sum(prior, axis=agent_i)
    total = other_probs.sum()
    other_probs = other_probs / total if total > 0 else np.ones((3, 3)) / 9.0
    ev = np.zeros(3)
    for r_i in range(3):
        for r_j in range(3):
            for r_k in range(3):
                roles = [0, 0, 0]
                roles[agent_i] = r_i
                roles[other[0]] = r_j
                roles[other[1]] = r_k
                flat_idx = roles[0] * 9 + roles[1] * 3 + roles[2]
                ev[r_i] += other_probs[r_j, r_k] * float(values[flat_idx, intent, thp, ehp])
    return ev


def threshold_role_dist(agent_i, intent, thp, ehp, prior, values,
                        current_role, delta, tau):
    ev = expected_values_per_role(agent_i, intent, thp, ehp, prior, values)
    current_val = ev[current_role]
    candidates = [r for r in range(3) if r != current_role and (ev[r] - current_val) > delta]
    if not candidates:
        d = np.zeros(3); d[current_role] = 1.0; return d
    cv = np.array([ev[r] for r in candidates])
    cv = cv / tau
    cv -= cv.max()
    p = np.exp(cv); p /= p.sum()
    d = np.zeros(3)
    for i, r in enumerate(candidates):
        d[r] = p[i]
    return d


def predict_from_trajectory_thresh(trajectory, values, tau_softmax, delta):
    results = []
    for stage in trajectory:
        prior = stage["prior"]
        intent, thp, ehp = stage["intent"], stage["thp"], stage["ehp"]
        prev_roles = stage["prev_roles"]
        per_agent = []
        for i in range(3):
            if prev_roles is None:
                per_agent.append(softmax_role_dist(i, intent, thp, ehp, prior, values, tau_softmax))
            else:
                per_agent.append(threshold_role_dist(
                    i, intent, thp, ehp, prior, values,
                    current_role=prev_roles[i], delta=delta, tau=tau_softmax))
        results.append({
            "predicted_dist": build_joint_dist(per_agent),
            "human_combo": stage["human_combo"],
            "model_marginal": np.mean(per_agent, axis=0),
        })
    return results


def make_factory(tau_softmax, delta):
    def factory(records_subset, traj_subset):
        idx_map = {id(r): i for i, r in enumerate(records_subset)}
        def predict_fn(record):
            i = idx_map[id(record)]
            return predict_from_trajectory_thresh(
                traj_subset[i], record["env_config"]["values"], tau_softmax, delta)
        return predict_fn
    return factory


def evaluate_pooled(records, trajectories, tau_softmax, delta):
    factory = make_factory(tau_softmax, delta)
    val = compute_pooled_metric(records, trajectories, factory, metric=METRIC)
    return {
        "tau_softmax": float(tau_softmax),
        "delta": float(delta),
        METRIC: val,
    }


def main():
    print("=" * 66)
    print("Stage 2: Bayesian Threshold")
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
    done = get_completed_keys(results, ["tau_softmax", "delta"])

    ts_vals = np.linspace(0.1, 20.0, 20)
    d_vals = np.linspace(0.0, 0.5, 20)
    total = len(ts_vals) * len(d_vals)
    print(f"\nCoarse: {total} points (done {len(done)})")
    count = len(done)
    for ts in ts_vals:
        added = False
        for d in d_vals:
            if (float(ts), float(d)) in done:
                continue
            r = evaluate_pooled(records, trajectories, ts, d)
            results.append(r)
            added = True
            count += 1
        if added:
            save_checkpoint(coarse_path, results)
            print(f"  [{count}/{total}] ...", flush=True)

    best = pick_best(results, METRIC)
    print(f"Coarse best: tau_softmax={best['tau_softmax']:.4f} "
          f"delta={best['delta']:.4f} {METRIC}={best[METRIC]:.4f}")

    refined_path = str(CHECKPOINT_DIR / "refined_results.json")
    refined = load_checkpoint(refined_path)
    done_r = get_completed_keys(refined, ["tau_softmax", "delta"])
    ts_step = (20.0 - 0.1) / 19
    d_step = 0.5 / 19
    fine_ts = np.linspace(max(0.1, best["tau_softmax"] - ts_step),
                           min(20.0, best["tau_softmax"] + ts_step), 11)
    fine_d = np.linspace(max(0.0, best["delta"] - d_step),
                          min(0.5, best["delta"] + d_step), 11)
    print(f"\nRefined: {len(fine_ts) * len(fine_d)} points")
    for ts in fine_ts:
        added = False
        for d in fine_d:
            if (float(ts), float(d)) in done_r:
                continue
            r = evaluate_pooled(records, trajectories, ts, d)
            refined.append(r)
            added = True
        if added:
            save_checkpoint(refined_path, refined)

    all_results = results + refined
    best = pick_best(all_results, METRIC)

    def obj(params):
        return -evaluate_pooled(records, trajectories, params[0], params[1])[METRIC]

    print(f"\nPolish from ts={best['tau_softmax']:.4f} d={best['delta']:.4f}")
    opt = minimize(obj, [best["tau_softmax"], best["delta"]],
                   method="L-BFGS-B", bounds=[(0.01, 50.0), (0.0, 2.0)],
                   options={"maxiter": 50, "ftol": 1e-6})
    r = evaluate_pooled(records, trajectories, float(opt.x[0]), float(opt.x[1]))
    all_results.append(r)
    best = pick_best(all_results, METRIC)

    eval_blocks = compute_disaggregated_metrics(
        records, trajectories,
        make_factory(best["tau_softmax"], best["delta"]))

    output = {
        "metric_optimized": METRIC,
        "stage1_params": {
            "tau_prior": s1["tau_prior"], "epsilon": s1["epsilon"],
            "memory_strategy": strat.name,
        },
        "tuned_params": {
            "tau_softmax": float(best["tau_softmax"]),
            "delta": float(best["delta"]),
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
