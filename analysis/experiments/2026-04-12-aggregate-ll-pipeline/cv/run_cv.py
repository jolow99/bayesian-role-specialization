"""Game-level leave-one-out cross-validation across all 7 Stage-2 models.

For each fold:
    1. Hold out one game (or paired singleton games for stratification).
    2. Refit each model's tunable params on the training fold using
       ``agg_ll_pooled`` as the objective.
    3. Evaluate the fitted params on the held-out fold, reporting
       disaggregated metrics for human / bot / pooled subsets.

Stage 1 params are loaded from the full-data fit; we do not refit them per
fold (deferred to a future experiment).

Outputs: ``cv/cv_results.json`` containing per-fold metrics and param
stability summaries for every model.
"""

from __future__ import annotations

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from stage2_common import (
    load_stage1_params, load_records, precompute_trajectories,
    compute_pooled_metric, compute_disaggregated_metrics,
)
from memory_strategies import strategy_from_params
from cv_utils import stratified_game_folds

# Import each model's prediction factory. Use relative imports from sibling
# directories via sys.path manipulation.
for sub in ("bayesian_belief", "bayesian_value", "bayesian_walk",
            "bayesian_thresh", "bayesian_walk_ps", "bayesian_thresh_ps",
            "mixture_ps"):
    sys.path.insert(0, str(EXPERIMENT_DIR / sub))

from bayesian_belief.tune import make_predict_fn as belief_factory
from bayesian_value.tune import make_factory as value_factory
from bayesian_walk.tune import make_factory as walk_factory
from bayesian_thresh.tune import make_factory as thresh_factory
from bayesian_walk_ps.tune import make_factory as walk_ps_factory
from bayesian_thresh_ps.tune import make_factory as thresh_ps_factory
from mixture_ps.tune import make_factory as mixture_factory

OUTPUT_PATH = SCRIPT_DIR / "cv_results.json"


# ──────────────────────────────────────────────────────────────────────
# Per-model refit helpers
# ──────────────────────────────────────────────────────────────────────

def _fit_one(records, trajectories, param_names, bounds, x0, factory_fn):
    """Scipy L-BFGS-B fit maximising pooled ``agg_ll``.

    ``factory_fn(params) -> predict_fn_factory`` constructs the factory for a
    given params tuple.
    """
    def obj(p):
        factory = factory_fn(*p)
        return -compute_pooled_metric(records, trajectories, factory,
                                       metric="agg_ll")

    opt = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 40, "ftol": 1e-5})
    fitted = {n: float(opt.x[i]) for i, n in enumerate(param_names)}
    return fitted


def fit_belief(records, trajectories):
    return {}


def fit_value(records, trajectories, x0=None):
    def factory_fn(tau):
        return value_factory(float(tau))
    return _fit_one(records, trajectories, ["tau_softmax"],
                    [(0.01, 50.0)], x0 or [3.0], factory_fn)


def fit_walk(records, trajectories, x0=None):
    def factory_fn(tau, eps):
        return walk_factory(float(tau), float(eps))
    return _fit_one(records, trajectories,
                    ["tau_softmax", "epsilon_switch"],
                    [(0.01, 50.0), (0.0, 1.0)],
                    x0 or [3.0, 0.3], factory_fn)


def fit_thresh(records, trajectories, x0=None):
    def factory_fn(tau, delta):
        return thresh_factory(float(tau), float(delta))
    return _fit_one(records, trajectories,
                    ["tau_softmax", "delta"],
                    [(0.01, 50.0), (0.0, 2.0)],
                    x0 or [3.0, 0.1], factory_fn)


def fit_walk_ps(records, trajectories, x0=None):
    def factory_fn(eps):
        return walk_ps_factory(float(eps))
    return _fit_one(records, trajectories, ["epsilon_switch"],
                    [(0.0, 1.0)], x0 or [0.3], factory_fn)


def fit_thresh_ps(records, trajectories, x0=None):
    def factory_fn(eps, delta):
        return thresh_ps_factory(float(eps), float(delta))
    return _fit_one(records, trajectories,
                    ["epsilon_switch", "delta"],
                    [(0.0, 1.0), (0.0, 1.0)],
                    x0 or [0.3, 0.1], factory_fn)


def fit_mixture(records, trajectories, frozen, x0=None):
    def factory_fn(w):
        return mixture_factory(
            frozen["walk_eps"], frozen["thresh_eps"],
            frozen["thresh_delta"], float(w))
    return _fit_one(records, trajectories, ["w"],
                    [(0.0, 1.0)], x0 or [0.5], factory_fn)


# ──────────────────────────────────────────────────────────────────────
# Per-model evaluation helpers (build factory from fitted params)
# ──────────────────────────────────────────────────────────────────────

def factory_for(model, params, frozen=None):
    if model == "bayesian_belief":
        return belief_factory
    if model == "bayesian_value":
        return value_factory(params["tau_softmax"])
    if model == "bayesian_walk":
        return walk_factory(params["tau_softmax"], params["epsilon_switch"])
    if model == "bayesian_thresh":
        return thresh_factory(params["tau_softmax"], params["delta"])
    if model == "bayesian_walk_ps":
        return walk_ps_factory(params["epsilon_switch"])
    if model == "bayesian_thresh_ps":
        return thresh_ps_factory(params["epsilon_switch"], params["delta"])
    if model == "mixture_ps":
        return mixture_factory(
            frozen["walk_eps"], frozen["thresh_eps"],
            frozen["thresh_delta"], params["w"])
    raise ValueError(f"unknown model {model!r}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 66)
    print("Game-level LOO cross-validation — all 7 Stage-2 models")
    print("=" * 66)

    s1 = load_stage1_params(EXPERIMENT_DIR)
    strat = strategy_from_params(
        s1.get("memory_strategy", "full"), s1.get("window"),
        s1.get("drift_delta", 0.0))
    print(f"Stage 1: tau_prior={s1['tau_prior']:.4f} epsilon={s1['epsilon']:.6f} "
          f"strategy={strat.name}")

    records = load_records(include_bot_rounds=True)
    print(f"\nPreflight counts:")
    print(f"  records: {len(records)}")
    print(f"  env_ids: {len(set(r['env_id'] for r in records))}")
    print(f"  game_ids: {len(set(r['game_id'] for r in records))}")

    print("\nPrecomputing trajectories (once for all folds)...")
    all_trajs = precompute_trajectories(
        records, s1["tau_prior"], s1["epsilon"], memory_strategy=strat)
    idx_map = {id(r): i for i, r in enumerate(records)}

    def split_trajs(record_subset):
        return [all_trajs[idx_map[id(r)]] for r in record_subset]

    folds = stratified_game_folds(records)
    print(f"Stratified folds: {len(folds)}")
    for fi, fold in enumerate(folds):
        n_h = sum(1 for r in fold["heldout"] if r["round_type"] == "human")
        n_b = sum(1 for r in fold["heldout"] if r["round_type"] == "bot")
        print(f"  fold {fi}: heldout {fold['heldout_game_ids']} ({n_h}H, {n_b}B)")

    # Load mixture's frozen params (from standalone fits).
    mix_frozen_path = EXPERIMENT_DIR / "mixture_ps" / "params.json"
    mix_frozen = None
    if mix_frozen_path.exists():
        with open(mix_frozen_path) as f:
            p = json.load(f)["tuned_params"]
        mix_frozen = {
            "walk_eps": p["walk_eps_frozen"],
            "thresh_eps": p["thresh_eps_frozen"],
            "thresh_delta": p["thresh_delta_frozen"],
        }

    models = [
        ("bayesian_belief", fit_belief, None),
        ("bayesian_value", fit_value, None),
        ("bayesian_walk", fit_walk, None),
        ("bayesian_thresh", fit_thresh, None),
        ("bayesian_walk_ps", fit_walk_ps, None),
        ("bayesian_thresh_ps", fit_thresh_ps, None),
        ("mixture_ps",
         lambda recs, trajs: fit_mixture(recs, trajs, mix_frozen),
         mix_frozen),
    ]

    cv_results = {}
    for model_name, fit_fn, frozen in models:
        if model_name == "mixture_ps" and mix_frozen is None:
            print(f"\n[skip] {model_name}: standalone ps fits missing")
            continue
        print(f"\n--- {model_name} ---")
        per_fold = []
        for fi, fold in enumerate(folds):
            train = fold["train"]
            heldout = fold["heldout"]
            train_trajs = split_trajs(train)
            heldout_trajs = split_trajs(heldout)

            fitted = fit_fn(train, train_trajs)
            factory = factory_for(model_name, fitted, frozen)
            heldout_eval = compute_disaggregated_metrics(
                heldout, heldout_trajs, factory)
            entry = {
                "fold": fi,
                "heldout_game_ids": fold["heldout_game_ids"],
                "fitted": fitted,
                "heldout": heldout_eval,
            }
            per_fold.append(entry)
            print(
                f"  fold {fi}: fitted={fitted} "
                f"heldout_pooled_agg_ll={heldout_eval['pooled']['agg_ll']:.4f}"
            )

        # Aggregate stability stats.
        pooled_agg = [f["heldout"]["pooled"]["agg_ll"] for f in per_fold]
        human_agg = [f["heldout"]["human"]["agg_ll"] for f in per_fold]
        bot_agg = [f["heldout"]["bot"]["agg_ll"] for f in per_fold]
        spreads = [h - b for h, b in zip(human_agg, bot_agg)
                   if not (np.isnan(h) or np.isnan(b))]

        param_stats = {}
        if per_fold[0]["fitted"]:
            for k in per_fold[0]["fitted"]:
                vals = [f["fitted"][k] for f in per_fold]
                mean = float(np.mean(vals))
                std = float(np.std(vals))
                param_stats[k] = {
                    "mean": mean, "std": std,
                    "cov": float(std / abs(mean)) if mean != 0 else float("nan"),
                }

        cv_results[model_name] = {
            "folds": per_fold,
            "heldout_pooled_agg_ll": {
                "mean": float(np.nanmean(pooled_agg)),
                "std": float(np.nanstd(pooled_agg)),
                "min": float(np.nanmin(pooled_agg)),
                "max": float(np.nanmax(pooled_agg)),
            },
            "heldout_human_agg_ll": {
                "mean": float(np.nanmean(human_agg)),
                "std": float(np.nanstd(human_agg)),
            },
            "heldout_bot_agg_ll": {
                "mean": float(np.nanmean(bot_agg)),
                "std": float(np.nanstd(bot_agg)),
            },
            "human_bot_spread": {
                "mean": float(np.mean(spreads)) if spreads else float("nan"),
                "std": float(np.std(spreads)) if spreads else float("nan"),
                "n": len(spreads),
            },
            "param_stability": param_stats,
        }
        print(
            f"  -> heldout pooled agg_ll: "
            f"{cv_results[model_name]['heldout_pooled_agg_ll']['mean']:.4f} "
            f"± {cv_results[model_name]['heldout_pooled_agg_ll']['std']:.4f}"
        )

    with open(OUTPUT_PATH, "w") as f:
        json.dump(cv_results, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
