"""Stage 2: Mixture PS — tunes w only.

walk_ps (epsilon_switch) and thresh_ps (epsilon_switch, delta) are frozen at
their standalone fits in the sibling directories. This is a fair-comparison
tradeoff: we're not doing a joint fit.
"""

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


def walk_ps_dist(prior, agent_i, prev_role, epsilon_switch):
    marg = posterior_marginal(prior, agent_i)
    if prev_role is None:
        return marg
    stick = np.zeros(3)
    stick[prev_role] = 1.0
    return (1.0 - epsilon_switch) * stick + epsilon_switch * marg


def thresh_ps_switch(prior, agent_i, prev_role, delta):
    marg = posterior_marginal(prior, agent_i)
    if prev_role is None:
        return marg
    cur = marg[prev_role]
    candidates = [r for r in range(3) if r != prev_role and (marg[r] - cur) > delta]
    if not candidates:
        d = np.zeros(3); d[prev_role] = 1.0; return d
    cp = np.array([marg[r] for r in candidates])
    cp = cp / cp.sum()
    d = np.zeros(3)
    for i, r in enumerate(candidates):
        d[r] = cp[i]
    return d


def thresh_ps_full(prior, agent_i, prev_role, epsilon_switch, delta):
    """Thresh_ps behaves the same as walk_ps but picks the ``switch`` action
    from a thresholded marginal. This matches the structure of the standalone
    thresh_ps model so the mixture is over comparable distributions."""
    if prev_role is None:
        return posterior_marginal(prior, agent_i)
    stick = np.zeros(3); stick[prev_role] = 1.0
    switch = thresh_ps_switch(prior, agent_i, prev_role, delta)
    return (1.0 - epsilon_switch) * stick + epsilon_switch * switch


def predict_from_trajectory(trajectory, walk_eps, thresh_eps, thresh_delta, w):
    results = []
    for stage in trajectory:
        prior = stage["prior"]
        prev_roles = stage["prev_roles"]
        per_agent = []
        for i in range(3):
            pr = prev_roles[i] if prev_roles is not None else None
            d_walk = walk_ps_dist(prior, i, pr, walk_eps)
            d_thresh = thresh_ps_full(prior, i, pr, thresh_eps, thresh_delta)
            per_agent.append(w * d_walk + (1.0 - w) * d_thresh)
        results.append({
            "predicted_dist": build_joint_dist(per_agent),
            "human_combo": stage["human_combo"],
            "model_marginal": np.mean(per_agent, axis=0),
        })
    return results


def make_factory(walk_eps, thresh_eps, thresh_delta, w):
    def factory(records_subset, traj_subset):
        idx_map = {id(r): i for i, r in enumerate(records_subset)}
        def predict_fn(record):
            i = idx_map[id(record)]
            return predict_from_trajectory(
                traj_subset[i], walk_eps, thresh_eps, thresh_delta, w)
        return predict_fn
    return factory


def evaluate_pooled(records, trajectories, walk_eps, thresh_eps, thresh_delta, w):
    factory = make_factory(walk_eps, thresh_eps, thresh_delta, w)
    val = compute_pooled_metric(records, trajectories, factory, metric=METRIC)
    return {
        "walk_eps": float(walk_eps),
        "thresh_eps": float(thresh_eps),
        "thresh_delta": float(thresh_delta),
        "w": float(w),
        METRIC: val,
    }


def load_frozen_ps_params():
    """Load frozen params from the two standalone PS fits."""
    walk_path = EXPERIMENT_DIR / "bayesian_walk_ps" / "params.json"
    thresh_path = EXPERIMENT_DIR / "bayesian_thresh_ps" / "params.json"
    if not walk_path.exists() or not thresh_path.exists():
        raise FileNotFoundError(
            "mixture_ps depends on standalone fits in bayesian_walk_ps/ and "
            "bayesian_thresh_ps/. Run those tunes first."
        )
    with open(walk_path) as f:
        walk = json.load(f)
    with open(thresh_path) as f:
        thresh = json.load(f)
    return {
        "walk_eps": float(walk["tuned_params"]["epsilon_switch"]),
        "thresh_eps": float(thresh["tuned_params"]["epsilon_switch"]),
        "thresh_delta": float(thresh["tuned_params"]["delta"]),
    }


def main():
    print("=" * 66)
    print("Stage 2: Mixture PS (w-only fit, frozen walk/thresh)")
    print("=" * 66)

    s1 = load_stage1_params(EXPERIMENT_DIR)
    strat = strategy_from_params(
        s1.get("memory_strategy", "full"), s1.get("window"),
        s1.get("drift_delta", 0.0))
    print(f"Stage 1: tau_prior={s1['tau_prior']:.4f} epsilon={s1['epsilon']:.6f} "
          f"strategy={strat.name}")
    frozen = load_frozen_ps_params()
    print(f"Frozen: walk_eps={frozen['walk_eps']:.4f} "
          f"thresh_eps={frozen['thresh_eps']:.4f} "
          f"thresh_delta={frozen['thresh_delta']:.4f}")

    records = load_records(include_bot_rounds=True)
    print("Precomputing trajectories...")
    trajectories = precompute_trajectories(
        records, s1["tau_prior"], s1["epsilon"], memory_strategy=strat)

    coarse_path = str(CHECKPOINT_DIR / "coarse_results.json")
    results = load_checkpoint(coarse_path)
    done = get_completed_keys(results, ["w"])
    w_vals = np.linspace(0.0, 1.0, 21)
    print(f"\nCoarse (w): {len(w_vals)} points (done {len(done)})")
    for w in w_vals:
        if (float(w),) in done:
            continue
        r = evaluate_pooled(records, trajectories,
                            frozen["walk_eps"], frozen["thresh_eps"],
                            frozen["thresh_delta"], w)
        results.append(r)
        save_checkpoint(coarse_path, results)

    best = pick_best(results, METRIC)
    print(f"Coarse best: w={best['w']:.4f} {METRIC}={best[METRIC]:.4f}")

    def obj(params):
        w = float(params[0])
        return -evaluate_pooled(records, trajectories,
                                frozen["walk_eps"], frozen["thresh_eps"],
                                frozen["thresh_delta"], w)[METRIC]

    print(f"\nPolish from w={best['w']:.4f}")
    opt = minimize(obj, [best["w"]], method="L-BFGS-B",
                   bounds=[(0.0, 1.0)],
                   options={"maxiter": 50, "ftol": 1e-6})
    r = evaluate_pooled(records, trajectories,
                        frozen["walk_eps"], frozen["thresh_eps"],
                        frozen["thresh_delta"], float(opt.x[0]))
    results.append(r)
    best = pick_best(results, METRIC)

    eval_blocks = compute_disaggregated_metrics(
        records, trajectories,
        make_factory(frozen["walk_eps"], frozen["thresh_eps"],
                     frozen["thresh_delta"], best["w"]))

    output = {
        "metric_optimized": METRIC,
        "stage1_params": {
            "tau_prior": s1["tau_prior"], "epsilon": s1["epsilon"],
            "memory_strategy": strat.name,
        },
        "tuned_params": {
            "w": float(best["w"]),
            "walk_eps_frozen": frozen["walk_eps"],
            "thresh_eps_frozen": frozen["thresh_eps"],
            "thresh_delta_frozen": frozen["thresh_delta"],
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
