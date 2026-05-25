"""Stage 2: fit each of 7 models under each of 3 objectives, then build the
3-objective × 4-metric × 7-model ranking heatmap (plus a Random baseline row).

For each of:
    OBJECTIVES = [combo_r, agg_ll, mean_ll]
    MODELS = [belief, value, walk, thresh, walk_ps, thresh_ps, mixture_ps]
we fit the model's params by maximising the objective, then evaluate all
four core metrics (combo_r, marg_r, agg_ll, mean_ll) at the fitted point.

The ranking heatmap rows are models, columns are evaluation metrics, and
each (model, metric) cell shows three sub-bars — one per fit objective —
coloured by within-(metric, objective) rank.

Outputs:
    results.json           — 21 cells of fitted params + eval, + Random
    comparison_table.md    — 4 metric tables, each with 7 model rows × 3 fit-obj cols
    figures/ranking_heatmap.png  — the headline figure
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from shared.constants import ALL_ROLE_COMBOS

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
    "mixture_ps": {
        "params": ["w"],
        "bounds": [(0.0, 1.0)],
        "coarse": [np.linspace(0.0, 1.0, 21)],
    },
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

# Display order from best-expected to worst-expected (walk family first).
MODEL_ORDER = [
    "bayesian_walk", "bayesian_walk_ps", "mixture_ps",
    "bayesian_belief", "bayesian_value",
    "bayesian_thresh", "bayesian_thresh_ps",
]
MODEL_LABELS = {
    "bayesian_walk_ps": "Bayesian Walk-PS",
    "bayesian_walk": "Bayesian Walk",
    "mixture_ps": "Mixture-PS",
    "bayesian_thresh_ps": "Bayesian Thresh-PS",
    "bayesian_thresh": "Bayesian Threshold",
    "bayesian_belief": "Bayesian-Belief",
    "bayesian_value": "Bayesian-Value",
}

# Core metrics used by the ranking heatmap (paper Fig 4).
CORE_METRICS = [
    ("combo_r", "combo_r\nPearson, per-combo × stage × env"),
    ("marg_r", "marg_r\nPearson, role marginals"),
    ("agg_ll", "agg_ll\naggregate cross-entropy (↑ better)"),
    ("mean_ll", "mean_ll\nper-sample mean log-lik (↑ better)"),
]

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
# Random baseline (objective-free reference row)
# ──────────────────────────────────────────────────────────────────────

def random_predict_fn_factory():
    uniform = {c: 1.0 / 27 for c in ALL_ROLE_COMBOS}
    uniform_marg = np.ones(3) / 3.0
    def predict_fn(record):
        return [
            {"predicted_dist": dict(uniform),
             "human_combo": hc,
             "model_marginal": uniform_marg.copy()}
            for hc in record["stage_roles"]
        ]
    return predict_fn


def eval_random(records):
    return _clean_metrics(eval_subset(records, random_predict_fn_factory()))


# ──────────────────────────────────────────────────────────────────────
# Table: 4 sub-tables, one per metric, each 7 models × 3 fit objectives
# ──────────────────────────────────────────────────────────────────────

def build_table(cells, random_eval, n_records, s1):
    lines = [
        "# Stage-2 model comparison (5 exports, human-only, clean teams)",
        "",
        f"Fitted on **{n_records}** clean human team-rounds using Stage-1 "
        f"params (τ_prior = {s1['tau_prior']:.4f}, ε = {s1['epsilon']:.6f}, "
        f"memory_strategy = `{s1['memory_strategy']}`). "
        "Each of the 7 models was fit under each of 3 objectives "
        "(`combo_r`, `agg_ll`, `mean_ll`); all four core metrics are then "
        "reported at the fitted point.",
        "",
        "The **Random** row is a 1/27-uniform reference (no parameters, no fit).",
        "",
    ]

    for key, title in CORE_METRICS:
        lines.append(f"## Eval metric: `{key}` ({title.split(chr(10))[1] if chr(10) in title else ''})")
        lines.append("")
        header = "| Model | " + " | ".join(f"Fit on `{o}`" for o in OBJECTIVES) + " |"
        sep = "|-------|" + "|".join("--------:" for _ in OBJECTIVES) + "|"
        lines.append(header)
        lines.append(sep)
        for model in MODEL_ORDER:
            if model not in cells:
                continue
            row = f"| {MODEL_LABELS[model]} |"
            for obj in OBJECTIVES:
                cell = cells[model].get(obj)
                if cell is None:
                    row += " — |"
                else:
                    v = cell["eval"].get(key, float("nan"))
                    row += f" {v:.4f} |"
            lines.append(row)
        # Random row
        v = random_eval.get(key, float("nan"))
        lines.append(f"| _Random (1/27)_ | {v:.4f} | {v:.4f} | {v:.4f} |")
        lines.append("")

    lines.append("## Fitted parameters")
    lines.append("")
    lines.append("| Model | Fit objective | Params |")
    lines.append("|-------|---------------|--------|")
    for model in MODEL_ORDER:
        if model not in cells:
            continue
        for obj in OBJECTIVES:
            cell = cells[model].get(obj)
            if cell is None:
                continue
            fp = cell["fitted_params"]
            ps = ", ".join(f"{k}={v:.4f}" for k, v in fp.items()
                           if not k.endswith("_frozen"))
            lines.append(f"| {MODEL_LABELS[model]} | {obj} | {ps or '(none)'} |")
    lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Ranking heatmap
# ──────────────────────────────────────────────────────────────────────

def ranking_heatmap(cells, random_eval, n_records, s1):
    """4-panel heatmap. Each panel = one metric.

    Rows are models; columns are fit-objectives. Cell colour encodes rank
    within the column (0 = best). Numeric value drawn on each cell. A
    Random reference row at the bottom shows the chance floor; it's not
    used in the rank coloring.
    """
    model_order = [m for m in MODEL_ORDER if m in cells]
    n_models = len(model_order)
    n_obj = len(OBJECTIVES)

    fig, axes = plt.subplots(
        1, len(CORE_METRICS),
        figsize=(3.5 * len(CORE_METRICS), 0.55 * (n_models + 1) + 2.5)
    )

    for ax, (key, title) in zip(axes, CORE_METRICS):
        matrix = np.full((n_models, n_obj), np.nan)
        for i, model in enumerate(model_order):
            for j, obj in enumerate(OBJECTIVES):
                cell = cells[model].get(obj)
                if cell is not None:
                    matrix[i, j] = cell["eval"].get(key, np.nan)

        rank_matrix = np.full_like(matrix, np.nan)
        for j in range(n_obj):
            col = matrix[:, j]
            valid = ~np.isnan(col)
            if valid.any():
                order = np.argsort(-col[valid])
                ranks = np.empty(int(valid.sum()), dtype=int)
                ranks[order] = np.arange(int(valid.sum()))
                rank_matrix[valid, j] = ranks

        # Append Random reference row (NaN rank → grey).
        full_matrix = np.vstack([matrix, [random_eval.get(key, np.nan)] * n_obj])
        full_rank = np.vstack([rank_matrix, [np.nan] * n_obj])

        cmap = plt.get_cmap("RdYlGn_r")
        cmap.set_bad("#d0d0d0")
        im = ax.imshow(np.ma.masked_invalid(full_rank), cmap=cmap,
                        aspect="auto", vmin=0, vmax=n_models - 1)
        ax.set_xticks(range(n_obj))
        ax.set_xticklabels([f"fit\n{o}" for o in OBJECTIVES], fontsize=8)
        ax.set_yticks(range(n_models + 1))
        if ax is axes[0]:
            ax.set_yticklabels(
                [MODEL_LABELS[m] for m in model_order] + ["Random (1/27)"],
                fontsize=9,
            )
        else:
            ax.set_yticklabels([])
        ax.set_title(title, fontsize=10)

        for i in range(full_matrix.shape[0]):
            for j in range(full_matrix.shape[1]):
                v = full_matrix[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=8, color="black")

        # Faint separator line between models and Random
        ax.axhline(n_models - 0.5, color="black", linewidth=0.6, alpha=0.5)

    fig.suptitle(
        "Model ranking stability across fit objectives "
        f"(n={n_records} clean human team-rounds; "
        f"Stage 1: {s1['memory_strategy']}, τ={s1['tau_prior']:.2f}, ε={s1['epsilon']:.3f})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    out = FIGURES_DIR / "ranking_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 66)
    print("2026-05-25 full pipeline — Stage 2 (7 models × 3 objectives)")
    print("=" * 66)

    s1_path = SCRIPT_DIR / "stage1_inference" / "best_inference_params.json"
    if not s1_path.exists():
        print(f"ERROR: Stage 1 params not found at {s1_path}")
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

    cells = {m: {} for m in MODEL_ORDER}

    # Order: fit non-mixture models first per objective, then mixture (which
    # needs the matching walk_ps/thresh_ps fits already done).
    base_order = [m for m in MODEL_ORDER if m != "mixture_ps"]

    for obj in OBJECTIVES:
        print(f"\n{'─' * 50}")
        print(f"Objective: {obj}")
        for model in base_order:
            t0 = time.time()
            result = fit_model(model, records, trajectories, obj)
            dt = time.time() - t0
            cells[model][obj] = result
            fp = result["fitted_params"]
            ev = result["eval"]
            ps = ", ".join(f"{k}={v:.4f}" for k, v in fp.items())
            print(
                f"  {model:22s} fit={obj:8s}: {ps or '(none)':30s} → "
                f"combo_r={ev.get('combo_r', float('nan')):.4f} "
                f"agg_ll={ev.get('agg_ll', float('nan')):.4f} "
                f"mean_ll={ev.get('mean_ll', float('nan')):.4f} "
                f"({dt:.1f}s)",
                flush=True,
            )

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
        fp = result["fitted_params"]
        ev = result["eval"]
        print(
            f"  mixture_ps             fit={obj:8s}: w={fp['w']:.4f} → "
            f"combo_r={ev.get('combo_r', float('nan')):.4f} "
            f"agg_ll={ev.get('agg_ll', float('nan')):.4f} "
            f"mean_ll={ev.get('mean_ll', float('nan')):.4f} "
            f"({dt:.1f}s)",
            flush=True,
        )

    print("\nRandom baseline (1/27 uniform, objective-free reference)...")
    random_eval = eval_random(records)
    print(
        f"  Random                                : "
        f"combo_r={random_eval.get('combo_r', float('nan')):.4f} "
        f"agg_ll={random_eval.get('agg_ll', float('nan')):.4f} "
        f"mean_ll={random_eval.get('mean_ll', float('nan')):.4f}"
    )

    output = {
        "exports": [d.name for d in __import__("pipeline").EXPORT_DIRS],
        "stage1_params": {
            "tau_prior": s1["tau_prior"],
            "epsilon": s1["epsilon"],
            "memory_strategy": strat.name,
        },
        "n_records": len(records),
        "objectives": OBJECTIVES,
        "cells": cells,
        "random_baseline": random_eval,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=_json_default)
    print(f"\nSaved results to {OUTPUT_PATH}")

    table = build_table(cells, random_eval, len(records), output["stage1_params"])
    with open(TABLE_PATH, "w") as f:
        f.write(table)
    print(f"Saved table to {TABLE_PATH}")

    print("\nGenerating ranking heatmap...")
    ranking_heatmap(cells, random_eval, len(records), output["stage1_params"])


if __name__ == "__main__":
    main()
