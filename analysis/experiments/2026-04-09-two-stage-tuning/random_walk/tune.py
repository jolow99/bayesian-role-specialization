"""Stage 2: Random Walk — tunes eps (no Bayesian component).

(1-eps) stick + (eps/2) per other role.
Choice params: eps
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
    load_records, compute_all_metrics,
    ROLE_CHAR_TO_IDX, ALL_ROLE_COMBOS,
)
from shared.evaluation import run_predictions, compute_pearson, compute_log_likelihood

CHECKPOINT_DIR = SCRIPT_DIR / 'checkpoints'
CHECKPOINT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = SCRIPT_DIR / 'params.json'
METRIC = 'combo_r'


def evaluate(records, eps):
    for i, rec in enumerate(records):
        rec['_traj_idx'] = i

    def predict_fn(record):
        preds = []
        for s, human_combo in enumerate(record['stage_roles']):
            prev = record['stage_roles'][s - 1] if s > 0 else None
            if prev is None:
                dist = {c: 1.0 / 27 for c in ALL_ROLE_COMBOS}
            else:
                dist = {}
                for combo in ALL_ROLE_COMBOS:
                    p = 1.0
                    for c, prev_c in zip(combo, prev):
                        p *= (1.0 - eps) if c == prev_c else (eps / 2.0)
                    dist[combo] = p
                total = sum(dist.values())
                dist = {c: p / total for c, p in dist.items()}
            marg = np.zeros(3)
            for combo, prob in dist.items():
                for c in combo:
                    marg[ROLE_CHAR_TO_IDX[c]] += prob
            marg /= 3.0
            preds.append({
                'predicted_dist': dist,
                'human_combo': human_combo,
                'model_marginal': marg,
            })
        return preds

    metrics = compute_all_metrics(records, predict_fn)
    metrics['eps'] = float(eps)
    return metrics


def main():
    print("=" * 60)
    print("Random Walk (no Bayesian component)")
    print("=" * 60)

    records = load_records()

    # Phase 1: Coarse grid
    coarse_path = str(CHECKPOINT_DIR / 'coarse_results.json')
    results = load_checkpoint(coarse_path)
    completed = get_completed_keys(results, ['eps'])

    eps_vals = np.linspace(0.0, 0.5, 21)
    print(f"\nCoarse grid: {len(eps_vals)} points")
    print(f"  Already completed: {len(completed)}")

    for eps in eps_vals:
        if (float(eps),) in completed:
            continue
        res = evaluate(records, eps)
        results.append(res)
        save_checkpoint(coarse_path, results)

    best = pick_best(results, METRIC)
    print(f"Coarse best: eps={best['eps']:.4f} combo_r={best['combo_r']:.4f}")

    # Phase 2: Refined
    refined_path = str(CHECKPOINT_DIR / 'refined_results.json')
    refined = load_checkpoint(refined_path)
    completed_r = get_completed_keys(refined, ['eps'])

    e_step = 0.5 / 20
    fine_eps = np.linspace(max(0.0, best['eps'] - e_step),
                           min(0.5, best['eps'] + e_step), 11)

    print(f"\nRefined grid: {len(fine_eps)} points")
    for eps in fine_eps:
        if (float(eps),) in completed_r:
            continue
        res = evaluate(records, eps)
        refined.append(res)
        save_checkpoint(refined_path, refined)

    all_results = results + refined
    best = pick_best(all_results, METRIC)

    # Phase 3: Scipy polish
    def objective(params):
        return -evaluate(records, params[0])[METRIC]

    print(f"\nScipy polish from eps={best['eps']:.4f}")
    opt = minimize(objective, [best['eps']], method='L-BFGS-B',
                   bounds=[(0.001, 0.999)], options={'maxiter': 50, 'ftol': 1e-6})
    opt_res = evaluate(records, opt.x[0])
    all_results.append(opt_res)
    best = pick_best(all_results, METRIC)

    output = {
        'metric_optimized': METRIC,
        'stage1_params': None,
        'aggregate_tuned': {
            'eps': best['eps'],
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
