"""Stage 2: Bayesian Threshold — tunes tau_softmax, delta.

Stick unless value gap > delta, then softmax among candidates.
Choice params: tau_softmax, delta
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
    compute_all_metrics, build_joint_dist,
    ROLE_SHORT, ROLE_CHAR_TO_IDX,
)
from shared.inference import softmax_role_dist

CHECKPOINT_DIR = SCRIPT_DIR / 'checkpoints'
CHECKPOINT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = SCRIPT_DIR / 'params.json'
METRIC = 'combo_r'


def expected_values_per_role(agent_i, intent, team_hp, enemy_hp, prior, values):
    """Compute expected value for each role of agent_i, marginalizing over others."""
    other_agents = [a for a in range(3) if a != agent_i]
    other_probs = np.sum(prior, axis=agent_i)
    total = other_probs.sum()
    other_probs = other_probs / total if total > 0 else np.ones((3, 3)) / 9.0

    ev = np.zeros(3)
    for r_i in range(3):
        for r_j in range(3):
            for r_k in range(3):
                roles = [0, 0, 0]
                roles[agent_i] = r_i
                roles[other_agents[0]] = r_j
                roles[other_agents[1]] = r_k
                flat_idx = roles[0] * 9 + roles[1] * 3 + roles[2]
                ev[r_i] += other_probs[r_j, r_k] * float(values[flat_idx, intent, team_hp, enemy_hp])
    return ev


def threshold_role_dist(agent_i, intent, team_hp, enemy_hp, prior, values,
                        current_role, delta, tau):
    """Stick unless value gap > delta, then softmax among candidates."""
    ev = expected_values_per_role(agent_i, intent, team_hp, enemy_hp, prior, values)
    current_val = ev[current_role]
    candidates = [r for r in range(3) if r != current_role and (ev[r] - current_val) > delta]

    if not candidates:
        dist = np.zeros(3)
        dist[current_role] = 1.0
        return dist

    candidate_vals = np.array([ev[r] for r in candidates])
    scaled = candidate_vals / tau
    scaled -= scaled.max()
    exp_vals = np.exp(scaled)
    probs = exp_vals / exp_vals.sum()

    dist = np.zeros(3)
    for i, r in enumerate(candidates):
        dist[r] = probs[i]
    return dist


def predict_from_trajectory_thresh(record, trajectory, values, tau_softmax, delta):
    results = []
    for stage in trajectory:
        prior = stage['prior']
        intent, thp, ehp = stage['intent'], stage['thp'], stage['ehp']
        prev_roles = stage['prev_roles']

        per_agent = []
        for i in range(3):
            if prev_roles is None:
                per_agent.append(softmax_role_dist(i, intent, thp, ehp, prior, values, tau_softmax))
            else:
                per_agent.append(threshold_role_dist(
                    i, intent, thp, ehp, prior, values,
                    current_role=prev_roles[i], delta=delta, tau=tau_softmax))

        results.append({
            'predicted_dist': build_joint_dist(per_agent),
            'human_combo': stage['human_combo'],
            'model_marginal': np.mean(per_agent, axis=0),
        })
    return results


def evaluate(records, trajectories, tau_softmax, delta):
    for i, rec in enumerate(records):
        rec['_traj_idx'] = i

    def predict_fn(record):
        idx = record['_traj_idx']
        return predict_from_trajectory_thresh(
            record, trajectories[idx], record['env_config']['values'],
            tau_softmax, delta)

    metrics = compute_all_metrics(records, predict_fn)
    metrics['tau_softmax'] = float(tau_softmax)
    metrics['delta'] = float(delta)
    return metrics


def main():
    print("=" * 60)
    print("Stage 2: Bayesian Threshold")
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
    completed = get_completed_keys(results, ['tau_softmax', 'delta'])

    ts_vals = np.linspace(0.1, 20.0, 20)
    delta_vals = np.linspace(0.0, 0.5, 20)
    total = len(ts_vals) * len(delta_vals)
    print(f"\nCoarse grid: {len(ts_vals)} x {len(delta_vals)} = {total} points")
    print(f"  Already completed: {len(completed)}")

    count = len(completed)
    for ts in ts_vals:
        batch_added = False
        for d in delta_vals:
            if (float(ts), float(d)) in completed:
                continue
            res = evaluate(records, trajectories, ts, d)
            results.append(res)
            batch_added = True
            count += 1
        if batch_added:
            save_checkpoint(coarse_path, results)
            print(f"  [{count}/{total}] ...", flush=True)

    best = pick_best(results, METRIC)
    print(f"Coarse best: tau_softmax={best['tau_softmax']:.4f} delta={best['delta']:.4f} "
          f"combo_r={best['combo_r']:.4f}")

    # Phase 2: Refined
    refined_path = str(CHECKPOINT_DIR / 'refined_results.json')
    refined = load_checkpoint(refined_path)
    completed_r = get_completed_keys(refined, ['tau_softmax', 'delta'])

    ts_step = (20.0 - 0.1) / 19
    d_step = 0.5 / 19
    fine_ts = np.linspace(max(0.1, best['tau_softmax'] - ts_step),
                          min(20.0, best['tau_softmax'] + ts_step), 11)
    fine_d = np.linspace(max(0.0, best['delta'] - d_step),
                         min(0.5, best['delta'] + d_step), 11)

    print(f"\nRefined grid: {len(fine_ts)} x {len(fine_d)} = {len(fine_ts)*len(fine_d)} points")
    for ts in fine_ts:
        batch_added = False
        for d in fine_d:
            if (float(ts), float(d)) in completed_r:
                continue
            res = evaluate(records, trajectories, ts, d)
            refined.append(res)
            batch_added = True
        if batch_added:
            save_checkpoint(refined_path, refined)

    all_results = results + refined
    best = pick_best(all_results, METRIC)

    # Phase 3: Scipy polish
    def objective(params):
        return -evaluate(records, trajectories, params[0], params[1])[METRIC]

    print(f"\nScipy polish from ts={best['tau_softmax']:.4f} delta={best['delta']:.4f}")
    opt = minimize(objective, [best['tau_softmax'], best['delta']],
                   method='L-BFGS-B',
                   bounds=[(0.01, 50.0), (0.0, 2.0)],
                   options={'maxiter': 50, 'ftol': 1e-6})
    opt_res = evaluate(records, trajectories, opt.x[0], opt.x[1])
    all_results.append(opt_res)
    best = pick_best(all_results, METRIC)

    output = {
        'metric_optimized': METRIC,
        'stage1_params': {
            'tau_prior': s1['tau_prior'], 'epsilon': s1['epsilon'],
            'window': s1['window'], 'drift_delta': s1['drift_delta'],
        },
        'aggregate_tuned': {
            'tau_softmax': best['tau_softmax'],
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
