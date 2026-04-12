"""Generate all summary visualizations for 2026-04-12-aggregate-ll-pipeline.

Produces:
    summary/01_model_ranking.png          — bar chart across 4 metrics
    summary/02_human_bot_spread.png       — disaggregated agg_ll comparison
    summary/03_cv_boxplots.png            — CV fold-level held-out agg_ll
    summary/04_agg_ll_vs_combo_r.png      — scatter (small grid, key models)
    summary/05_per_env_*.png              — per-env stage-over-stage model vs human
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from stage2_common import (
    load_records, precompute_trajectories, build_joint_dist,
    posterior_marginal, load_stage1_params,
)
from memory_strategies import strategy_from_params
from shared.evaluation import run_predictions, compute_pearson
from shared.inference import softmax_role_dist, combo_marginal
from shared.constants import ROLE_SHORT, ROLE_CHAR_TO_IDX, ROLE_COLORS

MODELS = [
    "bayesian_belief",
    "bayesian_walk_ps",
    "bayesian_walk",
    "mixture_ps",
    "bayesian_value",
    "bayesian_thresh_ps",
    "bayesian_thresh",
]
SHORT = {
    "bayesian_belief": "Belief",
    "bayesian_value": "Value",
    "bayesian_walk": "Walk",
    "bayesian_thresh": "Thresh",
    "bayesian_walk_ps": "Walk-PS",
    "bayesian_thresh_ps": "Thresh-PS",
    "mixture_ps": "Mix-PS",
}


def load_all_params():
    out = {}
    for m in MODELS:
        p = EXPERIMENT_DIR / m / "params.json"
        if p.exists():
            out[m] = json.load(open(p))
    return out


# ──────────────────────────────────────────────────────────────────────
# 1. Model ranking bar chart
# ──────────────────────────────────────────────────────────────────────

def plot_model_ranking(all_params):
    metrics = ["agg_ll", "combo_r", "marg_r", "mean_ll"]
    titles = ["Aggregate LL (↑ better)", "Combo Pearson r (↑ better)",
              "Marginal Pearson r (↑ better)", "Mean per-team LL (↑ better)"]
    subsets = ["human", "bot", "pooled"]
    colors = {"human": "#3498db", "bot": "#e74c3c", "pooled": "#2c3e50"}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    models_present = [m for m in MODELS if m in all_params]
    labels = [SHORT[m] for m in models_present]
    x = np.arange(len(models_present))
    width = 0.25

    for ax, metric, title in zip(axes, metrics, titles):
        for si, subset in enumerate(subsets):
            vals = []
            for m in models_present:
                ev = all_params[m].get("eval", {}).get(subset, {})
                v = ev.get(metric, float("nan"))
                vals.append(v if not np.isnan(v) else 0)
            offset = (si - 1) * width
            bars = ax.bar(x + offset, vals, width, label=subset,
                          color=colors[subset], alpha=0.85)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(0, color="gray", lw=0.5)

    # Clip thresh on agg_ll so the chart is readable
    axes[0].set_ylim(bottom=max(axes[0].get_ylim()[0], -6))
    axes[3].set_ylim(bottom=max(axes[3].get_ylim()[0], -6))

    fig.suptitle("Stage 2 model comparison — disaggregated metrics",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(SCRIPT_DIR / "01_model_ranking.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 01_model_ranking.png")


# ──────────────────────────────────────────────────────────────────────
# 2. Human / bot spread comparison
# ──────────────────────────────────────────────────────────────────────

def plot_spread(all_params):
    fig, ax = plt.subplots(figsize=(12, 6))
    models_present = [m for m in MODELS if m in all_params]
    labels = [SHORT[m] for m in models_present]
    x = np.arange(len(models_present))
    width = 0.3

    human_vals, bot_vals = [], []
    for m in models_present:
        h = all_params[m]["eval"].get("human", {}).get("agg_ll", float("nan"))
        b = all_params[m]["eval"].get("bot", {}).get("agg_ll", float("nan"))
        human_vals.append(h if not np.isnan(h) else 0)
        bot_vals.append(b if not np.isnan(b) else 0)

    ax.bar(x - width / 2, human_vals, width, label="Human rounds",
           color="#3498db", alpha=0.85)
    ax.bar(x + width / 2, bot_vals, width, label="Bot rounds",
           color="#e74c3c", alpha=0.85)

    # Spread annotations
    for i, (h, b) in enumerate(zip(human_vals, bot_vals)):
        spread = h - b
        y = max(h, b) + 0.15
        if abs(spread) < 10:
            ax.annotate(f"Δ={spread:+.2f}", (i, min(y, -0.5)),
                        ha="center", fontsize=7, color="#7f8c8d")

    ax.set_ylabel("agg_ll (↑ better)")
    ax.set_title("Human vs Bot round performance by model", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(bottom=max(min(human_vals + bot_vals) - 0.5, -6))
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / "02_human_bot_spread.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 02_human_bot_spread.png")


# ──────────────────────────────────────────────────────────────────────
# 3. CV boxplots
# ──────────────────────────────────────────────────────────────────────

def plot_cv_boxplots():
    cv_path = EXPERIMENT_DIR / "cv" / "cv_results.json"
    if not cv_path.exists():
        print("Skipping CV boxplots (cv_results.json not found)")
        return
    cv = json.load(open(cv_path))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    subset_titles = [
        ("pooled", "Held-out pooled agg_ll"),
        ("human", "Held-out human agg_ll"),
        ("bot", "Held-out bot agg_ll"),
    ]

    models_present = [m for m in MODELS if m in cv]
    labels = [SHORT[m] for m in models_present]

    for ax, (subset, title) in zip(axes, subset_titles):
        data = []
        for m in models_present:
            fold_vals = []
            for fold in cv[m]["folds"]:
                v = fold["heldout"].get(subset, {}).get("agg_ll", float("nan"))
                if not np.isnan(v):
                    fold_vals.append(v)
            data.append(fold_vals if fold_vals else [0])

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
        for patch in bp["boxes"]:
            patch.set_facecolor("#3498db")
            patch.set_alpha(0.6)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("agg_ll")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3, axis="y")
        # Clip for readability
        all_vals = [v for d in data for v in d]
        lo = np.percentile(all_vals, 5) if all_vals else -5
        ax.set_ylim(bottom=max(lo - 1, -8))

    fig.suptitle("Cross-validation stability (16 game-level LOO folds)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(SCRIPT_DIR / "03_cv_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 03_cv_boxplots.png")


# ──────────────────────────────────────────────────────────────────────
# 4. agg_ll vs combo_r scatter (evaluate small grid for key models)
# ──────────────────────────────────────────────────────────────────────

def plot_scatter(records, trajectories):
    """Evaluate agg_ll AND combo_r at a handful of param settings for
    walk_ps, belief, value. Plot the objective frontier."""
    from stage2_common import eval_subset

    # walk_ps: sweep epsilon_switch
    walk_ps_pts = []
    for eps in np.linspace(0.0, 1.0, 21):
        from bayesian_walk_ps.tune import make_factory as wf
        factory = wf(float(eps))
        predict_fn = factory(records, trajectories)
        m = eval_subset(records, predict_fn)
        walk_ps_pts.append((m["agg_ll"], m["combo_r"], eps))

    # value: sweep tau_softmax
    value_pts = []
    for tau in [0.5, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50]:
        from bayesian_value.tune import make_factory as vf
        factory = vf(float(tau))
        predict_fn = factory(records, trajectories)
        m = eval_subset(records, predict_fn)
        value_pts.append((m["agg_ll"], m["combo_r"], tau))

    # belief: single point
    from bayesian_belief.tune import make_predict_fn as bf
    predict_fn = bf(records, trajectories)
    m = eval_subset(records, predict_fn)
    belief_pt = (m["agg_ll"], m["combo_r"])

    fig, ax = plt.subplots(figsize=(9, 7))
    wp = np.array(walk_ps_pts)
    ax.scatter(wp[:, 0], wp[:, 1], c="#2ecc71", s=40, alpha=0.7,
               label="Walk-PS (ε sweep)", zorder=3)
    vp = np.array(value_pts)
    ax.scatter(vp[:, 0], vp[:, 1], c="#e74c3c", s=40, alpha=0.7,
               label="Value (τ sweep)", zorder=3)
    ax.scatter(*belief_pt, c="#3498db", s=100, marker="*", zorder=4,
               label="Belief (no params)")

    # Annotate extremes
    best_wp = wp[np.argmax(wp[:, 0])]
    ax.annotate(f"ε={best_wp[2]:.2f}", (best_wp[0], best_wp[1]),
                fontsize=8, textcoords="offset points", xytext=(5, 5))
    best_vp = vp[np.argmax(vp[:, 0])]
    ax.annotate(f"τ={best_vp[2]:.0f}", (best_vp[0], best_vp[1]),
                fontsize=8, textcoords="offset points", xytext=(5, -10))

    ax.set_xlabel("Pooled agg_ll (↑ better)")
    ax.set_ylabel("Pooled combo_r (↑ better)")
    ax.set_title("Objective frontier: agg_ll vs combo_r", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / "04_agg_ll_vs_combo_r.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved 04_agg_ll_vs_combo_r.png")


# ──────────────────────────────────────────────────────────────────────
# 5. Per-env stage-over-stage: top model vs human
# ──────────────────────────────────────────────────────────────────────

def plot_per_env(records, trajectories):
    """For each env, 3-panel figure: combo distributions, marginals, model
    comparison with belief + walk_ps."""
    from bayesian_belief.tune import make_predict_fn as bf
    from bayesian_walk_ps.tune import make_factory as wf

    # Load fitted walk_ps params
    wp_params = json.load(
        open(EXPERIMENT_DIR / "bayesian_walk_ps" / "params.json"))
    eps = wp_params["tuned_params"]["epsilon_switch"]

    belief_fn = bf(records, trajectories)
    walk_fn = wf(eps)(records, trajectories)

    belief_results = run_predictions(records, belief_fn)
    walk_results = run_predictions(records, walk_fn)

    ROLE_CLR = {"F": "#e74c3c", "T": "#3498db", "M": "#2ecc71"}

    for env_id in sorted(belief_results.keys()):
        bdata = belief_results[env_id]
        wdata = walk_results[env_id]
        canon = bdata["canonical_combos"]
        stat_profile = bdata["stat_profile"]
        optimal_canon = bdata["canonical_optimal"]
        max_stages = bdata["max_stages"]
        stages = list(range(1, max_stages + 1))

        # Build per-combo time series
        human_p = {cc: [] for cc in canon}
        belief_p = {cc: [] for cc in canon}
        walk_p = {cc: [] for cc in canon}
        for s in range(max_stages):
            n = bdata["stage_counts"].get(s, 0)
            hc = bdata["stage_human"].get(s, {})
            bp = bdata["stage_predicted"].get(s, {})
            wp = wdata["stage_predicted"].get(s, {})
            for cc in canon:
                human_p[cc].append(hc.get(cc, 0) / n if n > 0 else 0)
                belief_p[cc].append(bp.get(cc, 0.0))
                walk_p[cc].append(wp.get(cc, 0.0))

        played = [cc for cc in canon
                  if cc == optimal_canon
                  or any(p > 0 for p in human_p[cc])
                  or any(p > 0.02 for p in walk_p[cc])]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: Human combo frequencies
        for cc in played:
            is_opt = cc == optimal_canon
            kw = dict(linewidth=2.5 if is_opt else 1.2,
                      markersize=8 if is_opt else 4,
                      label=f"{cc}*" if is_opt else cc)
            if is_opt:
                kw["color"] = "red"
            axes[0].plot(stages, human_p[cc], "o-", **kw)
        axes[0].set_title(f"Human ({bdata['n_teams']} teams)", fontweight="bold")

        # Panel 2: Walk-PS model
        for cc in played:
            is_opt = cc == optimal_canon
            kw = dict(linewidth=2.5 if is_opt else 1.2,
                      markersize=8 if is_opt else 4,
                      label=f"{cc}*" if is_opt else cc)
            if is_opt:
                kw["color"] = "red"
            axes[1].plot(stages, walk_p[cc], "o-", **kw)
        axes[1].set_title(f"Walk-PS (ε={eps:.3f})", fontweight="bold")

        # Panel 3: Marginals (human vs belief vs walk)
        for role_idx, role_name in enumerate(["F", "T", "M"]):
            color = ROLE_CLR[role_name]
            h_m = [bdata["stage_human_marginal"].get(s, np.zeros(3))[role_idx]
                   for s in range(max_stages)]
            b_m = [bdata["stage_model_marginal"].get(s, np.zeros(3))[role_idx]
                   for s in range(max_stages)]
            w_m = [wdata["stage_model_marginal"].get(s, np.zeros(3))[role_idx]
                   for s in range(max_stages)]
            axes[2].plot(stages, h_m, "o-", color=color, lw=2,
                         label=f"{role_name} human")
            axes[2].plot(stages, b_m, "s--", color=color, lw=1.5, alpha=0.5,
                         label=f"{role_name} belief")
            axes[2].plot(stages, w_m, "^:", color=color, lw=1.5, alpha=0.7,
                         label=f"{role_name} walk")

        for ax in axes[:2]:
            ax.set_xlabel("Stage")
            ax.set_ylabel("P(combo)")
            ax.set_xticks(stages)
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        axes[2].set_xlabel("Stage")
        axes[2].set_ylabel("P(role)")
        axes[2].set_title("Marginals")
        axes[2].set_xticks(stages)
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].legend(fontsize=6, ncol=3)
        axes[2].grid(True, alpha=0.3)

        # Determine round type(s) for this env
        env_recs = [r for r in records if r["env_id"] == env_id]
        round_types = set(r["round_type"] for r in env_recs)
        rt_label = "/".join(sorted(round_types))

        fig.suptitle(
            f"Env {env_id} | {stat_profile} | Optimal: {optimal_canon} | "
            f"N={bdata['n_teams']} | {rt_label}",
            fontsize=11, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        safe_id = str(env_id).replace("/", "_")
        fig.savefig(SCRIPT_DIR / f"05_per_env_{safe_id}.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {len(belief_results)} per-env plots (05_per_env_*.png)")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 66)
    print("Generating visualizations")
    print("=" * 66)

    all_params = load_all_params()
    print(f"Loaded params for {len(all_params)} models")

    # 1 + 2: static from params.json
    plot_model_ranking(all_params)
    plot_spread(all_params)

    # 3: from CV results
    plot_cv_boxplots()

    # 4 + 5: need records + trajectories
    s1 = load_stage1_params(EXPERIMENT_DIR)
    strat = strategy_from_params(
        s1.get("memory_strategy", "full"), s1.get("window"),
        s1.get("drift_delta", 0.0))
    records = load_records(include_bot_rounds=True)
    print("Precomputing trajectories...")
    trajectories = precompute_trajectories(
        records, s1["tau_prior"], s1["epsilon"], memory_strategy=strat)

    plot_scatter(records, trajectories)
    plot_per_env(records, trajectories)

    print("\nAll visualizations saved to summary/")


if __name__ == "__main__":
    main()
