"""Stage 2: Bayesian Threshold PS — tunes delta.

Switch if posterior marginal gap > delta.
Choice params: delta
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
    compute_all_metrics, build_joint_dist, posterior_marginal,
)

CHECKPOINT_DIR = SCRIPT_DIR / 'checkpoints'
CHECKPOINT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = SCRIPT_DIR / 'params.json'
METRIC = 'combo_r'


def threshold_role_dist_ps(agent_i, prior, current_role, delta):
    """Switch if marginal gap > delta, distribute among candidates."""
    marg = posterior_marginal(prior, agent_i)
    current_prob = marg[current_role]
    candidates = [r for r in range(3) if r != current_role and (marg[r] - current_prob) > delta]

    if not candidates:
        dist = np.zeros(3)
        dist[current_role] = 1.0
        return dist

    candidate_probs = np.array([marg[r] for r in candidates])
    candidate_probs = candidate_probs / candidate_probs.sum()

    dist = np.zeros(3)
    for i, r in enumerate(candidates):
        dist[r] = candidate_probs[i]
    return dist


def predict_from_trajectory_thresh_ps(trajectory, delta):
    results = []
    for stage in trajectory:
        prior = stage['prior']
        prev_roles = stage['prev_roles']

        per_agent = []
        for i in range(3):
            if prev_roles is None:
                per_agent.append(posterior_marginal(prior, i))
            else:
                per_agent.append(threshold_role_dist_ps(
                    i, prior, current_role=prev_roles[i], delta=delta))

        results.append({
            'predicted_dist': build_joint_dist(per_agent),
            'human_combo': stage['human_combo'],
            'model_marginal': np.mean(per_agent, axis=0),
        })
    return results


def evaluate(records, trajectories, delta):
    for i, rec in enumerate(records):
        rec['_traj_idx'] = i

    def predict_fn(record):
        idx = record['_traj_idx']
        return predict_from_trajectory_thresh_ps(trajectories[idx], delta)

    metrics = compute_all_metrics(records, predict_fn)
    metrics['delta'] = float(delta)
    return metrics


def main():
    print("=" * 60)
    print("Stage 2: Bayesian Threshold PS")
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
    completed = get_completed_keys(results, ['delta'])

    delta_vals = np.linspace(0.0, 0.5, 21)
    print(f"\nCoarse grid: {len(delta_vals)} points")
    print(f"  Already completed: {len(completed)}")

    for d in delta_vals:
        if (float(d),) in completed:
            continue
        res = evaluate(records, trajectories, d)
        results.append(res)
        save_checkpoint(coarse_path, results)

    best = pick_best(results, METRIC)
    print(f"Coarse best: delta={best['delta']:.4f} combo_r={best['combo_r']:.4f}")

    # Phase 2: Refined
    refined_path = str(CHECKPOINT_DIR / 'refined_results.json')
    refined = load_checkpoint(refined_path)
    completed_r = get_completed_keys(refined, ['delta'])

    d_step = 0.5 / 20
    fine_d = np.linspace(max(0.0, best['delta'] - d_step),
                         min(0.5, best['delta'] + d_step), 11)

    print(f"\nRefined grid: {len(fine_d)} points")
    for d in fine_d:
        if (float(d),) in completed_r:
            continue
        res = evaluate(records, trajectories, d)
        refined.append(res)
        save_checkpoint(refined_path, refined)

    all_results = results + refined
    best = pick_best(all_results, METRIC)

    # Phase 3: Scipy polish
    def objective(params):
        return -evaluate(records, trajectories, params[0])[METRIC]

    print(f"\nScipy polish from delta={best['delta']:.4f}")
    opt = minimize(objective, [best['delta']], method='L-BFGS-B',
                   bounds=[(0.0, 1.0)], options={'maxiter': 50, 'ftol': 1e-6})
    opt_res = evaluate(records, trajectories, opt.x[0])
    all_results.append(opt_res)
    best = pick_best(all_results, METRIC)

    output = {
        'metric_optimized': METRIC,
        'stage1_params': {
            'tau_prior': s1['tau_prior'], 'epsilon': s1['epsilon'],
            'window': s1['window'], 'drift_delta': s1['drift_delta'],
        },
        'aggregate_tuned': {
            'delta': best['delta'],
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
