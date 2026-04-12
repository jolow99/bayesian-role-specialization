"""Stage 2: Bayesian Threshold PS — tunes epsilon_switch, delta on pooled agg_ll."""

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


def thresh_ps_dist(prior, agent_i, prev_role, delta):
    marg = posterior_marginal(prior, agent_i)
    if prev_role is None:
        return marg
    current = marg[prev_role]
    candidates = [r for r in range(3) if r != prev_role and (marg[r] - current) > delta]
    if not candidates:
        d = np.zeros(3); d[prev_role] = 1.0; return d
    cp = np.array([marg[r] for r in candidates])
    cp = cp / cp.sum()
    d = np.zeros(3)
    for i, r in enumerate(candidates):
        d[r] = cp[i]
    return d


def predict_from_trajectory(trajectory, epsilon_switch, delta):
    results = []
    for stage in trajectory:
        prior = stage["prior"]
        prev_roles = stage["prev_roles"]
        per_agent = []
        for i in range(3):
            prev_r = prev_roles[i] if prev_roles is not None else None
            marg = posterior_marginal(prior, i)
            if prev_r is None:
                per_agent.append(marg)
                continue
            switch = thresh_ps_dist(prior, i, prev_r, delta)
            stick = np.zeros(3); stick[prev_r] = 1.0
            per_agent.append((1.0 - epsilon_switch) * stick + epsilon_switch * switch)
        results.append({
            "predicted_dist": build_joint_dist(per_agent),
            "human_combo": stage["human_combo"],
            "model_marginal": np.mean(per_agent, axis=0),
        })
    return results


def make_factory(epsilon_switch, delta):
    def factory(records_subset, traj_subset):
        idx_map = {id(r): i for i, r in enumerate(records_subset)}
        def predict_fn(record):
            i = idx_map[id(record)]
            return predict_from_trajectory(traj_subset[i], epsilon_switch, delta)
        return predict_fn
    return factory


def evaluate_pooled(records, trajectories, epsilon_switch, delta):
    factory = make_factory(epsilon_switch, delta)
    val = compute_pooled_metric(records, trajectories, factory, metric=METRIC)
    return {
        "epsilon_switch": float(epsilon_switch),
        "delta": float(delta),
        METRIC: val,
    }


def main():
    print("=" * 66)
    print("Stage 2: Bayesian Threshold PS")
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
    done = get_completed_keys(results, ["epsilon_switch", "delta"])
    es_vals = np.linspace(0.0, 1.0, 21)
    d_vals = np.linspace(0.0, 0.5, 11)
    total = len(es_vals) * len(d_vals)
    print(f"\nCoarse: {total} points (done {len(done)})")
    for es in es_vals:
        added = False
        for d in d_vals:
            if (float(es), float(d)) in done:
                continue
            r = evaluate_pooled(records, trajectories, es, d)
            results.append(r)
            added = True
        if added:
            save_checkpoint(coarse_path, results)

    best = pick_best(results, METRIC)
    print(f"Coarse best: eps_switch={best['epsilon_switch']:.4f} "
          f"delta={best['delta']:.4f} {METRIC}={best[METRIC]:.4f}")

    refined_path = str(CHECKPOINT_DIR / "refined_results.json")
    refined = load_checkpoint(refined_path)
    done_r = get_completed_keys(refined, ["epsilon_switch", "delta"])
    es_step = 1.0 / 20
    d_step = 0.5 / 10
    fine_es = np.linspace(max(0.0, best["epsilon_switch"] - es_step),
                           min(1.0, best["epsilon_switch"] + es_step), 11)
    fine_d = np.linspace(max(0.0, best["delta"] - d_step),
                          min(0.5, best["delta"] + d_step), 11)
    print(f"\nRefined: {len(fine_es) * len(fine_d)} points")
    for es in fine_es:
        added = False
        for d in fine_d:
            if (float(es), float(d)) in done_r:
                continue
            r = evaluate_pooled(records, trajectories, es, d)
            refined.append(r)
            added = True
        if added:
            save_checkpoint(refined_path, refined)

    all_results = results + refined
    best = pick_best(all_results, METRIC)

    def obj(params):
        return -evaluate_pooled(records, trajectories, params[0], params[1])[METRIC]

    print(f"\nPolish from es={best['epsilon_switch']:.4f} d={best['delta']:.4f}")
    opt = minimize(obj, [best["epsilon_switch"], best["delta"]],
                   method="L-BFGS-B", bounds=[(0.0, 1.0), (0.0, 1.0)],
                   options={"maxiter": 50, "ftol": 1e-6})
    r = evaluate_pooled(records, trajectories, float(opt.x[0]), float(opt.x[1]))
    all_results.append(r)
    best = pick_best(all_results, METRIC)

    eval_blocks = compute_disaggregated_metrics(
        records, trajectories,
        make_factory(best["epsilon_switch"], best["delta"]))

    output = {
        "metric_optimized": METRIC,
        "stage1_params": {
            "tau_prior": s1["tau_prior"], "epsilon": s1["epsilon"],
            "memory_strategy": strat.name,
        },
        "tuned_params": {
            "epsilon_switch": float(best["epsilon_switch"]),
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
