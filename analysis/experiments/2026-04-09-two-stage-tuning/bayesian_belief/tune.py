"""Stage 2: Bayesian Belief — no choice params, just evaluates with stage 1 params.

Marginalizes posterior directly for role predictions.
Choice params: (none)
"""

import sys
import json
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from shared_utils import pick_best
from stage2_common import (
    load_stage1_params, load_records, precompute_trajectories,
    compute_all_metrics, build_joint_dist, posterior_marginal,
)

OUTPUT_PATH = SCRIPT_DIR / 'params.json'
METRIC = 'combo_r'


def predict_from_trajectory_belief(trajectory):
    results = []
    for stage in trajectory:
        prior = stage['prior']
        per_agent = [posterior_marginal(prior, i) for i in range(3)]

        results.append({
            'predicted_dist': build_joint_dist(per_agent),
            'human_combo': stage['human_combo'],
            'model_marginal': np.mean(per_agent, axis=0),
        })
    return results


def main():
    print("=" * 60)
    print("Stage 2: Bayesian Belief")
    print("=" * 60)

    s1 = load_stage1_params(EXPERIMENT_DIR)
    print(f"Stage 1 params: tau_prior={s1['tau_prior']:.4f} epsilon={s1['epsilon']:.6f} "
          f"strategy={s1['memory_strategy']}")

    records = load_records()
    print("Precomputing trajectories...")
    trajectories = precompute_trajectories(
        records, s1['tau_prior'], s1['epsilon'],
        window=s1['window'], drift_delta=s1['drift_delta'])

    for i, rec in enumerate(records):
        rec['_traj_idx'] = i

    def predict_fn(record):
        idx = record['_traj_idx']
        return predict_from_trajectory_belief(trajectories[idx])

    metrics = compute_all_metrics(records, predict_fn)

    output = {
        'metric_optimized': METRIC,
        'stage1_params': {
            'tau_prior': s1['tau_prior'], 'epsilon': s1['epsilon'],
            'window': s1['window'], 'drift_delta': s1['drift_delta'],
        },
        'aggregate_tuned': {
            'combo_r': metrics['combo_r'], 'marg_r': metrics['marg_r'],
            'mean_ll': metrics['mean_ll'],
            'switch_combo_r': metrics.get('switch_combo_r', float('nan')),
            'switch_marg_r': metrics.get('switch_marg_r', float('nan')),
            'switch_mean_ll': metrics.get('switch_mean_ll', float('nan')),
        },
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
