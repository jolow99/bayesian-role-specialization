"""Stage 2: Bayesian Walk — tunes tau_softmax, epsilon_switch.

Stick-or-switch with softmax best response on switch.
Choice params: tau_softmax, epsilon_switch
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


def predict_from_trajectory_walk(record, trajectory, values, tau_softmax, epsilon_switch):
    results = []
    for stage in trajectory:
        prior = stage['prior']
        intent, thp, ehp = stage['intent'], stage['thp'], stage['ehp']
        prev_roles = stage['prev_roles']

        switch_dist = [softmax_role_dist(i, intent, thp, ehp, prior, values, tau_softmax)
                       for i in range(3)]

        per_agent = []
        for i in range(3):
            if prev_roles is None:
                per_agent.append(switch_dist[i])
            else:
                stick = np.zeros(3)
                stick[prev_roles[i]] = 1.0
                per_agent.append((1.0 - epsilon_switch) * stick + epsilon_switch * switch_dist[i])

        results.append({
            'predicted_dist': build_joint_dist(per_agent),
            'human_combo': stage['human_combo'],
            'model_marginal': np.mean(per_agent, axis=0),
        })
    return results


def evaluate(records, trajectories, tau_softmax, epsilon_switch):
    for i, rec in enumerate(records):
        rec['_traj_idx'] = i

    def predict_fn(record):
        idx = record['_traj_idx']
        return predict_from_trajectory_walk(
            record, trajectories[idx], record['env_config']['values'],
            tau_softmax, epsilon_switch)

    metrics = compute_all_metrics(records, predict_fn)
    metrics['tau_softmax'] = float(tau_softmax)
    metrics['epsilon_switch'] = float(epsilon_switch)
    return metrics


def main():
    print("=" * 60)
    print("Stage 2: Bayesian Walk")
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
    completed = get_completed_keys(results, ['tau_softmax', 'epsilon_switch'])

    ts_vals = np.linspace(0.1, 20.0, 20)
    es_vals = np.linspace(0.0, 1.0, 21)
    total = len(ts_vals) * len(es_vals)
    print(f"\nCoarse grid: {len(ts_vals)} x {len(es_vals)} = {total} points")
    print(f"  Already completed: {len(completed)}")

    count = len(completed)
    for ts in ts_vals:
        batch_added = False
        for es in es_vals:
            if (float(ts), float(es)) in completed:
                continue
            res = evaluate(records, trajectories, ts, es)
            results.append(res)
            batch_added = True
            count += 1
        if batch_added:
            save_checkpoint(coarse_path, results)
            print(f"  [{count}/{total}] ...", flush=True)

    best = pick_best(results, METRIC)
    print(f"Coarse best: tau_softmax={best['tau_softmax']:.4f} eps_switch={best['epsilon_switch']:.4f} "
          f"combo_r={best['combo_r']:.4f}")

    # Phase 2: Refined
    refined_path = str(CHECKPOINT_DIR / 'refined_results.json')
    refined = load_checkpoint(refined_path)
    completed_r = get_completed_keys(refined, ['tau_softmax', 'epsilon_switch'])

    ts_step = (20.0 - 0.1) / 19
    es_step = 1.0 / 20
    fine_ts = np.linspace(max(0.1, best['tau_softmax'] - ts_step),
                          min(20.0, best['tau_softmax'] + ts_step), 11)
    fine_es = np.linspace(max(0.0, best['epsilon_switch'] - es_step),
                          min(1.0, best['epsilon_switch'] + es_step), 11)

    print(f"\nRefined grid: {len(fine_ts)} x {len(fine_es)} = {len(fine_ts)*len(fine_es)} points")
    for ts in fine_ts:
        batch_added = False
        for es in fine_es:
            if (float(ts), float(es)) in completed_r:
                continue
            res = evaluate(records, trajectories, ts, es)
            refined.append(res)
            batch_added = True
        if batch_added:
            save_checkpoint(refined_path, refined)

    all_results = results + refined
    best = pick_best(all_results, METRIC)

    # Phase 3: Scipy polish
    def objective(params):
        return -evaluate(records, trajectories, params[0], params[1])[METRIC]

    print(f"\nScipy polish from ts={best['tau_softmax']:.4f} es={best['epsilon_switch']:.4f}")
    opt = minimize(objective, [best['tau_softmax'], best['epsilon_switch']],
                   method='L-BFGS-B',
                   bounds=[(0.01, 50.0), (0.0, 1.0)],
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
            'epsilon_switch': best['epsilon_switch'],
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
