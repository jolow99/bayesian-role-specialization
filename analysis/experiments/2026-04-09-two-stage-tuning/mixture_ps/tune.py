"""Stage 2: Mixture PS — tunes epsilon_switch, delta, w.

Mixture of Walk PS and Threshold PS: w * walk_ps + (1-w) * thresh_ps.
Choice params: epsilon_switch, delta, w
"""

import sys
import json
import numpy as np
from pathlib import Path
from itertools import product
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from shared_utils import load_checkpoint, save_checkpoint, get_completed_keys, pick_best
from stage2_common import (
    load_stage1_params, load_records, precompute_trajectories,
    compute_all_metrics, build_joint_dist, posterior_marginal,
)

CHECKPOINT_DIR = SCRIPT_DIR / 'checkpoints'
CHECKPOINT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = SCRIPT_DIR / 'params.json'
METRIC = 'combo_r'


def walk_ps_dist(prior, agent_i, prev_role, epsilon_switch):
    marg = posterior_marginal(prior, agent_i)
    if prev_role is None:
        return marg
    stick = np.zeros(3)
    stick[prev_role] = 1.0
    return (1.0 - epsilon_switch) * stick + epsilon_switch * marg


def thresh_ps_dist(prior, agent_i, prev_role, delta):
    marg = posterior_marginal(prior, agent_i)
    if prev_role is None:
        return marg
    current_prob = marg[prev_role]
    candidates = [r for r in range(3) if r != prev_role and (marg[r] - current_prob) > delta]
    if not candidates:
        dist = np.zeros(3)
        dist[prev_role] = 1.0
        return dist
    candidate_probs = np.array([marg[r] for r in candidates])
    candidate_probs = candidate_probs / candidate_probs.sum()
    dist = np.zeros(3)
    for i, r in enumerate(candidates):
        dist[r] = candidate_probs[i]
    return dist


def predict_from_trajectory_mixture(trajectory, epsilon_switch, delta, w):
    results = []
    for stage in trajectory:
        prior = stage['prior']
        prev_roles = stage['prev_roles']

        per_agent = []
        for i in range(3):
            prev_r = prev_roles[i] if prev_roles is not None else None
            d_walk = walk_ps_dist(prior, i, prev_r, epsilon_switch)
            d_thresh = thresh_ps_dist(prior, i, prev_r, delta)
            per_agent.append(w * d_walk + (1.0 - w) * d_thresh)

        results.append({
            'predicted_dist': build_joint_dist(per_agent),
            'human_combo': stage['human_combo'],
            'model_marginal': np.mean(per_agent, axis=0),
        })
    return results


def evaluate(records, trajectories, epsilon_switch, delta, w):
    for i, rec in enumerate(records):
        rec['_traj_idx'] = i

    def predict_fn(record):
        idx = record['_traj_idx']
        return predict_from_trajectory_mixture(trajectories[idx], epsilon_switch, delta, w)

    metrics = compute_all_metrics(records, predict_fn)
    metrics['epsilon_switch'] = float(epsilon_switch)
    metrics['delta'] = float(delta)
    metrics['w'] = float(w)
    return metrics


def main():
    print("=" * 60)
    print("Stage 2: Mixture PS")
    print("=" * 60)

    s1 = load_stage1_params(EXPERIMENT_DIR)
    print(f"Stage 1 params: tau_prior={s1['tau_prior']:.4f} epsilon={s1['epsilon']:.6f} "
          f"strategy={s1['memory_strategy']}")

    records = load_records()
    print("Precomputing trajectories...")
    trajectories = precompute_trajectories(
        records, s1['tau_prior'], s1['epsilon'],
        window=s1['window'], drift_delta=s1['drift_delta'])

    # Phase 1: Coarse grid
    coarse_path = str(CHECKPOINT_DIR / 'coarse_results.json')
    results = load_checkpoint(coarse_path)
    completed = get_completed_keys(results, ['epsilon_switch', 'delta', 'w'])

    es_vals = np.linspace(0.0, 1.0, 11)
    d_vals = np.linspace(0.0, 0.5, 11)
    w_vals = np.linspace(0.0, 1.0, 11)
    total = len(es_vals) * len(d_vals) * len(w_vals)
    print(f"\nCoarse grid: {len(es_vals)} x {len(d_vals)} x {len(w_vals)} = {total} points")
    print(f"  Already completed: {len(completed)}")

    count = len(completed)
    for es in es_vals:
        batch_added = False
        for d in d_vals:
            for w in w_vals:
                if (float(es), float(d), float(w)) in completed:
                    continue
                res = evaluate(records, trajectories, es, d, w)
                results.append(res)
                batch_added = True
                count += 1
        if batch_added:
            save_checkpoint(coarse_path, results)
            print(f"  [{count}/{total}] ...", flush=True)

    best = pick_best(results, METRIC)
    print(f"Coarse best: eps_switch={best['epsilon_switch']:.4f} delta={best['delta']:.4f} "
          f"w={best['w']:.4f} combo_r={best['combo_r']:.4f}")

    # Phase 2: Refined
    refined_path = str(CHECKPOINT_DIR / 'refined_results.json')
    refined = load_checkpoint(refined_path)
    completed_r = get_completed_keys(refined, ['epsilon_switch', 'delta', 'w'])

    es_step = 1.0 / 10
    d_step = 0.5 / 10
    w_step = 1.0 / 10
    fine_es = np.linspace(max(0.0, best['epsilon_switch'] - es_step),
                          min(1.0, best['epsilon_switch'] + es_step), 11)
    fine_d = np.linspace(max(0.0, best['delta'] - d_step),
                         min(0.5, best['delta'] + d_step), 11)
    fine_w = np.linspace(max(0.0, best['w'] - w_step),
                         min(1.0, best['w'] + w_step), 11)

    refined_total = len(fine_es) * len(fine_d) * len(fine_w)
    print(f"\nRefined grid: {refined_total} points")
    count = len(completed_r)
    for es in fine_es:
        batch_added = False
        for d in fine_d:
            for w in fine_w:
                if (float(es), float(d), float(w)) in completed_r:
                    continue
                res = evaluate(records, trajectories, es, d, w)
                refined.append(res)
                batch_added = True
                count += 1
        if batch_added:
            save_checkpoint(refined_path, refined)
            print(f"  [{count}/{refined_total}] ...", flush=True)

    all_results = results + refined
    best = pick_best(all_results, METRIC)

    # Phase 3: Scipy polish
    def objective(params):
        return -evaluate(records, trajectories, params[0], params[1], params[2])[METRIC]

    print(f"\nScipy polish from es={best['epsilon_switch']:.4f} d={best['delta']:.4f} w={best['w']:.4f}")
    opt = minimize(objective, [best['epsilon_switch'], best['delta'], best['w']],
                   method='L-BFGS-B',
                   bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
                   options={'maxiter': 50, 'ftol': 1e-6})
    opt_res = evaluate(records, trajectories, opt.x[0], opt.x[1], opt.x[2])
    all_results.append(opt_res)
    best = pick_best(all_results, METRIC)

    output = {
        'metric_optimized': METRIC,
        'stage1_params': {
            'tau_prior': s1['tau_prior'], 'epsilon': s1['epsilon'],
            'window': s1['window'], 'drift_delta': s1['drift_delta'],
        },
        'aggregate_tuned': {
            'epsilon_switch': best['epsilon_switch'],
            'delta': best['delta'],
            'w': best['w'],
            'combo_r': best['combo_r'], 'marg_r': best['marg_r'], 'mean_ll': best['mean_ll'],
            'switch_combo_r': best.get('switch_combo_r', float('nan')),
            'switch_marg_r': best.get('switch_marg_r', float('nan')),
            'switch_mean_ll': best.get('switch_mean_ll', float('nan')),
        },
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
