"""Triple-objective × 7-model comparison on the 04-23 export (human-only).

For each of 3 objectives (combo_r, agg_ll, mean_ll) × 7 models, fits the
model's params on the clean human team-rounds from the 04-23 export using
that objective, then evaluates ALL metrics at the fitted point.

Self-contained: uses only pipeline.py and models.py from this folder.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from pipeline import (
    compute_pooled_metric, eval_subset, load_human_team_records,
    precompute_trajectories, strategy_from_params, _json_default,
)
from models import (
    belief_factory, value_factory, walk_factory, thresh_factory,
    walk_ps_factory, thresh_ps_factory, mixture_ps_factory,
)

OBJECTIVES = ["combo_r", "agg_ll", "mean_ll"]

MODELS = {
    "bayesian_belief": {"params": [], "bounds": []},
    "bayesian_value": {
        "params": ["tau_softmax"],
        "bounds": [(0.01, 50.0)],
        "coarse": [np.linspace(0.1, 20.0, 20)],
    },
    "bayesian_walk": {
        "params": ["tau_softmax", "epsilon_switch"],
        "bounds": [(0.01, 50.0), (0.0, 1.0)],
        "coarse": [np.linspace(0.1, 20.0, 15), np.linspace(0.0, 1.0, 15)],
    },
    "bayesian_thresh": {
        "params": ["tau_softmax", "delta"],
        "bounds": [(0.01, 50.0), (0.0, 2.0)],
        "coarse": [np.linspace(0.1, 20.0, 15), np.linspace(0.0, 0.5, 15)],
    },
    "bayesian_walk_ps": {
        "params": ["epsilon_switch"],
        "bounds": [(0.0, 1.0)],
        "coarse": [np.linspace(0.0, 1.0, 21)],
    },
    "bayesian_thresh_ps": {
        "params": ["epsilon_switch", "delta"],
        "bounds": [(0.0, 1.0), (0.0, 1.0)],
        "coarse": [np.linspace(0.0, 1.0, 15), np.linspace(0.0, 0.5, 11)],
    },
    "mixture_ps": {"params": ["w"], "bounds": [(0.0, 1.0)],
                   "coarse": [np.linspace(0.0, 1.0, 21)]},
}

FACTORY_MAP = {
    "bayesian_value": lambda tau_softmax: value_factory(tau_softmax),
    "bayesian_walk": lambda tau_softmax, epsilon_switch:
        walk_factory(tau_softmax, epsilon_switch),
    "bayesian_thresh": lambda tau_softmax, delta:
        thresh_factory(tau_softmax, delta),
    "bayesian_walk_ps": lambda epsilon_switch:
        walk_ps_factory(epsilon_switch),
    "bayesian_thresh_ps": lambda epsilon_switch, delta:
        thresh_ps_factory(epsilon_switch, delta),
}

OUTPUT_PATH = SCRIPT_DIR / "results.json"
TABLE_PATH = SCRIPT_DIR / "comparison_table.md"
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# Fitting helpers
# ──────────────────────────────────────────────────────────────────────

def fit_model(model_name, records, trajectories, objective, frozen_ps=None):
    spec = MODELS[model_name]

    if model_name == "bayesian_belief":
        factory = belief_factory()
        metrics = eval_subset(records, factory(records, trajectories))
        return {"fitted_params": {}, "eval": _clean_metrics(metrics)}

    if model_name == "mixture_ps":
        return _fit_mixture(records, trajectories, objective, frozen_ps)

    param_names = spec["params"]
    bounds = spec["bounds"]
    grids = spec["coarse"]
    make_factory_fn = FACTORY_MAP[model_name]

    def objective_fn(param_vals):
        factory = make_factory_fn(*param_vals)
        return compute_pooled_metric(records, trajectories, factory,
                                     metric=objective)

    best_val = -np.inf
    best_params = None
    if len(grids) == 1:
        for p0 in grids[0]:
            v = objective_fn([p0])
            if not np.isnan(v) and v > best_val:
                best_val = v
                best_params = [float(p0)]
    elif len(grids) == 2:
        for p0 in grids[0]:
            for p1 in grids[1]:
                v = objective_fn([p0, p1])
                if not np.isnan(v) and v > best_val:
                    best_val = v
                    best_params = [float(p0), float(p1)]

    if best_params is None:
        best_params = [float((b[0] + b[1]) / 2) for b in bounds]

    def neg_obj(params):
        v = objective_fn(list(params))
        return -v if not np.isnan(v) else 1e10

    opt = minimize(neg_obj, best_params, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 50, "ftol": 1e-6})
    final_params = [float(x) for x in opt.x]

    final_val = objective_fn(final_params)
    if np.isnan(final_val) or final_val < best_val:
        final_params = best_params

    factory = make_factory_fn(*final_params)
    predict_fn = factory(records, trajectories)
    metrics = eval_subset(records, predict_fn)

    fitted = dict(zip(param_names, final_params))
    return {"fitted_params": fitted, "eval": _clean_metrics(metrics)}


def _fit_mixture(records, trajectories, objective, frozen_ps):
    walk_eps = frozen_ps["walk_eps"]
    thresh_eps = frozen_ps["thresh_eps"]
    thresh_delta = frozen_ps["thresh_delta"]

    def objective_fn(w):
        factory = mixture_ps_factory(walk_eps, thresh_eps, thresh_delta, w)
        return compute_pooled_metric(records, trajectories, factory,
                                     metric=objective)

    best_val = -np.inf
    best_w = 0.5
    for w in np.linspace(0.0, 1.0, 21):
        v = objective_fn(float(w))
        if not np.isnan(v) and v > best_val:
            best_val = v
            best_w = float(w)

    def neg_obj(params):
        v = objective_fn(float(params[0]))
        return -v if not np.isnan(v) else 1e10

    opt = minimize(neg_obj, [best_w], method="L-BFGS-B", bounds=[(0.0, 1.0)],
                   options={"maxiter": 50, "ftol": 1e-6})
    final_w = float(opt.x[0])
    final_val = objective_fn(final_w)
    if np.isnan(final_val) or final_val < best_val:
        final_w = best_w

    factory = mixture_ps_factory(walk_eps, thresh_eps, thresh_delta, final_w)
    predict_fn = factory(records, trajectories)
    metrics = eval_subset(records, predict_fn)

    return {
        "fitted_params": {
            "w": final_w,
            "walk_eps_frozen": walk_eps,
            "thresh_eps_frozen": thresh_eps,
            "thresh_delta_frozen": thresh_delta,
        },
        "eval": _clean_metrics(metrics),
    }


def _clean_metrics(m):
    return {k: float(v) if isinstance(v, (int, float, np.floating, np.integer))
            else v for k, v in m.items()}


# ──────────────────────────────────────────────────────────────────────
# Output: table + figures
# ──────────────────────────────────────────────────────────────────────

MODEL_ORDER = [
    "bayesian_walk_ps", "bayesian_walk", "mixture_ps",
    "bayesian_thresh_ps", "bayesian_thresh",
    "bayesian_belief", "bayesian_value",
]


def build_table(cells, n_records):
    lines = [
        "# Metric Comparison (04-23 export, human-only, clean teams)",
        "",
        f"Fitted on {n_records} clean human team-rounds from the 04-23 "
        "export (see `pipeline.load_human_team_records` — every team-round "
        "must have a matching value matrix in `human-envs-big-pilot-matrices/"
        "<stat_profile>__<role_combo>`; missing matrices raise rather than "
        "silently dropping records). Each cell shows the metric value when "
        "the model was fit under that objective.",
        "",
    ]

    eval_metrics = ["combo_r", "marg_r", "agg_ll", "mean_ll"]
    for em in eval_metrics:
        lines.append(f"## Eval metric: `{em}`")
        lines.append("")
        header = "| Model | " + " | ".join(f"Fit: {obj}" for obj in OBJECTIVES) + " |"
        sep = "|-------|" + "|".join("--------:" for _ in OBJECTIVES) + "|"
        lines.append(header)
        lines.append(sep)
        for model in MODEL_ORDER:
            if model not in cells:
                continue
            row = f"| {model} |"
            for obj in OBJECTIVES:
                if obj in cells[model]:
                    v = cells[model][obj]["eval"].get(em, float("nan"))
                    row += f" {v:.4f} |"
                else:
                    row += " — |"
            lines.append(row)
        lines.append("")

    lines.append("## Fitted Parameters")
    lines.append("")
    lines.append("| Model | Objective | Params |")
    lines.append("|-------|-----------|--------|")
    for model in MODEL_ORDER:
        if model not in cells:
            continue
        for obj in OBJECTIVES:
            if obj not in cells[model]:
                continue
            fp = cells[model][obj]["fitted_params"]
            ps = ", ".join(f"{k}={v:.4f}" for k, v in fp.items()
                           if not k.endswith("_frozen"))
            lines.append(f"| {model} | {obj} | {ps or '(none)'} |")
    lines.append("")

    return "\n".join(lines)


def make_figures(cells):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available, skipping figures")
        return

    model_order = [m for m in MODEL_ORDER if m in cells]
    eval_metrics = ["combo_r", "marg_r", "agg_ll", "mean_ll"]
    metric_labels = {"combo_r": "Combo r", "marg_r": "Marg r",
                     "agg_ll": "Agg LL", "mean_ll": "Mean LL"}

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    for ax, em in zip(axes, eval_metrics):
        matrix = np.full((len(model_order), len(OBJECTIVES)), np.nan)
        for i, model in enumerate(model_order):
            for j, obj in enumerate(OBJECTIVES):
                if obj in cells.get(model, {}):
                    matrix[i, j] = cells[model][obj]["eval"].get(em, np.nan)

        rank_matrix = np.full_like(matrix, np.nan)
        for j in range(len(OBJECTIVES)):
            col = matrix[:, j]
            valid = ~np.isnan(col)
            if valid.any():
                order = np.argsort(-col[valid])
                ranks = np.empty_like(order)
                ranks[order] = np.arange(valid.sum())
                rank_matrix[valid, j] = ranks

        ax.imshow(rank_matrix, cmap="RdYlGn_r", aspect="auto",
                  vmin=0, vmax=len(model_order) - 1)
        ax.set_xticks(range(len(OBJECTIVES)))
        ax.set_xticklabels(OBJECTIVES, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(model_order)))
        ax.set_yticklabels(model_order if ax == axes[0] else [], fontsize=9)
        ax.set_title(metric_labels.get(em, em), fontsize=11)

        for i in range(len(model_order)):
            for j in range(len(OBJECTIVES)):
                v = matrix[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=7, color="black")

    fig.suptitle("Model Ranking Stability Across Objectives (04-23)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ranking_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {FIGURES_DIR / 'ranking_heatmap.png'}")

    param_models = {m: MODELS[m]["params"] for m in model_order
                    if MODELS[m]["params"] and m != "mixture_ps"}
    if "mixture_ps" in cells:
        param_models["mixture_ps"] = ["w"]

    n_panels = sum(len(ps) for ps in param_models.values())
    if n_panels > 0:
        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4),
                                  squeeze=False)
        ax_idx = 0
        for model, pnames in param_models.items():
            for pname in pnames:
                if pname.endswith("_frozen"):
                    continue
                ax = axes[0, ax_idx]
                vals = []
                for obj in OBJECTIVES:
                    if obj in cells.get(model, {}):
                        fp = cells[model][obj]["fitted_params"]
                        vals.append(fp.get(pname, np.nan))
                    else:
                        vals.append(np.nan)
                ax.bar(range(len(OBJECTIVES)), vals,
                       color=["#4e79a7", "#f28e2b", "#59a14f"])
                ax.set_xticks(range(len(OBJECTIVES)))
                ax.set_xticklabels(OBJECTIVES, fontsize=9)
                ax.set_title(f"{model}\n{pname}", fontsize=10)
                ax.set_ylabel(pname, fontsize=9)
                ax_idx += 1
        fig.suptitle("Fitted Parameters by Objective (04-23)", fontsize=13)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "param_sensitivity.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {FIGURES_DIR / 'param_sensitivity.png'}")

    all_points = {em: [] for em in eval_metrics}
    labels = []
    for model in model_order:
        for obj in OBJECTIVES:
            if obj in cells.get(model, {}):
                ev = cells[model][obj]["eval"]
                for em in eval_metrics:
                    all_points[em].append(ev.get(em, np.nan))
                labels.append(f"{model[:8]}|{obj[:5]}")

    n_em = len(eval_metrics)
    fig, axes = plt.subplots(n_em, n_em, figsize=(12, 12))
    for i, em_i in enumerate(eval_metrics):
        for j, em_j in enumerate(eval_metrics):
            ax = axes[i, j]
            if i == j:
                ax.hist([v for v in all_points[em_i] if not np.isnan(v)],
                        bins=10, color="#4e79a7", alpha=0.7)
                ax.set_xlabel(metric_labels.get(em_i, em_i), fontsize=9)
            else:
                x = np.array(all_points[em_j])
                y = np.array(all_points[em_i])
                valid = ~(np.isnan(x) | np.isnan(y))
                ax.scatter(x[valid], y[valid], s=20, alpha=0.7, c="#4e79a7")
                if valid.sum() > 2:
                    r = np.corrcoef(x[valid], y[valid])[0, 1]
                    ax.set_title(f"r={r:.3f}", fontsize=8)
            if i == n_em - 1:
                ax.set_xlabel(metric_labels.get(em_j, em_j), fontsize=9)
            if j == 0:
                ax.set_ylabel(metric_labels.get(em_i, em_i), fontsize=9)
    fig.suptitle("Metric Correlations Across All Fits (04-23)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "metric_correlation.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {FIGURES_DIR / 'metric_correlation.png'}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 66)
    print("04-23 metric comparison: 3 objectives x 7 models (human-only)")
    print("=" * 66)

    s1_path = SCRIPT_DIR / "stage1_inference" / "best_inference_params.json"
    if not s1_path.exists():
        print(f"ERROR: Stage 1 params not found at {s1_path}")
        print("Run stage1_inference/tune.py first.")
        sys.exit(1)
    with open(s1_path) as f:
        s1 = json.load(f)
    strat = strategy_from_params(
        s1.get("memory_strategy", "full"),
        s1.get("window"),
        s1.get("drift_delta", 0.0),
    )
    print(f"Stage 1: tau_prior={s1['tau_prior']:.4f} "
          f"epsilon={s1['epsilon']:.6f} strategy={strat.name}")

    records = load_human_team_records()

    print("Precomputing trajectories...")
    trajectories = precompute_trajectories(
        records, s1["tau_prior"], s1["epsilon"], memory_strategy=strat)

    cells = {}
    fit_order = [
        "bayesian_belief", "bayesian_value",
        "bayesian_walk", "bayesian_thresh",
        "bayesian_walk_ps", "bayesian_thresh_ps",
        "mixture_ps",
    ]

    for model in fit_order:
        if model == "mixture_ps":
            continue
        cells[model] = {}
        print(f"\n{'─' * 50}")
        print(f"Model: {model}")
        for obj in OBJECTIVES:
            t0 = time.time()
            result = fit_model(model, records, trajectories, obj)
            dt = time.time() - t0
            cells[model][obj] = result
            fp = result["fitted_params"]
            ev = result["eval"]
            ps = ", ".join(f"{k}={v:.4f}" for k, v in fp.items())
            print(f"  {obj:8s}: {ps or '(none)':30s} → "
                  f"combo_r={ev.get('combo_r', float('nan')):.4f} "
                  f"agg_ll={ev.get('agg_ll', float('nan')):.4f} "
                  f"mean_ll={ev.get('mean_ll', float('nan')):.4f} "
                  f"({dt:.1f}s)")

    cells["mixture_ps"] = {}
    print(f"\n{'─' * 50}")
    print("Model: mixture_ps")
    for obj in OBJECTIVES:
        walk_fp = cells["bayesian_walk_ps"][obj]["fitted_params"]
        thresh_fp = cells["bayesian_thresh_ps"][obj]["fitted_params"]
        frozen_ps = {
            "walk_eps": walk_fp["epsilon_switch"],
            "thresh_eps": thresh_fp["epsilon_switch"],
            "thresh_delta": thresh_fp["delta"],
        }
        t0 = time.time()
        result = fit_model("mixture_ps", records, trajectories, obj,
                           frozen_ps=frozen_ps)
        dt = time.time() - t0
        cells["mixture_ps"][obj] = result
        ev = result["eval"]
        fp = result["fitted_params"]
        print(f"  {obj:8s}: w={fp.get('w', float('nan')):.4f} "
              f"(frozen walk_eps={frozen_ps['walk_eps']:.4f} "
              f"thresh_eps={frozen_ps['thresh_eps']:.4f} "
              f"thresh_delta={frozen_ps['thresh_delta']:.4f}) → "
              f"combo_r={ev.get('combo_r', float('nan')):.4f} "
              f"agg_ll={ev.get('agg_ll', float('nan')):.4f} "
              f"({dt:.1f}s)")

    output = {
        "export": "bayesian-role-specialization-2026-04-23-09-12-55",
        "stage1_params": {
            "tau_prior": s1["tau_prior"],
            "epsilon": s1["epsilon"],
            "memory_strategy": strat.name,
        },
        "n_records": len(records),
        "objectives": OBJECTIVES,
        "cells": cells,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=_json_default)
    print(f"\nSaved results to {OUTPUT_PATH}")

    table = build_table(cells, len(records))
    with open(TABLE_PATH, "w") as f:
        f.write(table)
    print(f"Saved table to {TABLE_PATH}")

    print("\nGenerating figures...")
    make_figures(cells)

    print("\n" + "=" * 66)
    print("SUMMARY: Model ranking by combo_r under each objective")
    print("=" * 66)
    for obj in OBJECTIVES:
        ranked = sorted(cells.items(),
                        key=lambda kv: kv[1].get(obj, {}).get("eval", {}).get("combo_r", -999),
                        reverse=True)
        ranking = [f"{m}({kv[obj]['eval'].get('combo_r', float('nan')):.3f})"
                   for m, kv in ranked if obj in kv]
        print(f"  {obj:8s}: {' > '.join(ranking)}")


if __name__ == "__main__":
    main()
