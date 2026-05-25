"""Paper Fig 5 — per-position per-stage role choice, humans vs Walk vs Belief.

Decomposed by player position (which has a fixed stat profile in each env)
rather than pooled across positions. For each top-N env, three rows
(humans / Walk / Belief) × three columns (P1, P2, P3) × five stages of
F/T/M stacked bars. This exposes whether the model gets the
stat-specific deviation pattern right, not just the overall F/T/M mix.

Walk params are read from the full-pipeline ``results.json`` under the
``combo_r`` objective (matching the paper text).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent / "2026-05-25-full-pipeline"
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(PIPELINE_DIR))

from pipeline import (
    load_human_team_records, precompute_trajectories,
    strategy_from_params,
)
from models import belief_factory, walk_factory
from shared.constants import ROLE_CHAR_TO_IDX

ROLE_COLORS = {0: "#e74c3c", 1: "#3498db", 2: "#2ecc71"}
ROLE_NAMES = ["F", "T", "M"]

# Number of envs to display.
TOP_K = 3


def _ftm_from_counts(counts):
    total = counts.sum()
    return counts / total if total > 0 else np.zeros(3)


def main():
    with open(PIPELINE_DIR / "results.json") as f:
        results = json.load(f)
    s1 = results["stage1_params"]
    strat = strategy_from_params(s1["memory_strategy"], None, 0.0)
    walk_fp = results["cells"]["bayesian_walk"]["combo_r"]["fitted_params"]
    print(f"Stage 1: {s1}")
    print(f"Walk params (combo_r fit): {walk_fp}")

    records = load_human_team_records()
    trajectories = precompute_trajectories(
        records, s1["tau_prior"], s1["epsilon"], memory_strategy=strat)

    # Choose top-K envs by sample size.
    env_records = defaultdict(list)
    for rec, traj in zip(records, trajectories):
        env_records[(rec["stat_profile"], rec["optimal_roles"])].append((rec, traj))
    top = sorted(env_records.items(), key=lambda kv: -len(kv[1]))[:TOP_K]
    chosen = [(k, v) for k, v in top]
    print(f"Top-{TOP_K} envs:")
    for (sp, oc), v in chosen:
        print(f"  {sp}__{oc}: n={len(v)}")

    walk_factory_fn = walk_factory(walk_fp["tau_softmax"], walk_fp["epsilon_switch"])
    walk_predict_all = walk_factory_fn(records, trajectories)
    belief_predict_all = belief_factory()(records, trajectories)

    # Build per-env per-position per-stage stats.
    # human[(env, position, stage)] -> 3-vec of role probabilities
    # walk[(env, position, stage)] -> 3-vec mean of model's per-agent role marg
    # belief[(env, position, stage)] -> 3-vec mean of model's per-agent role marg
    env_data = {}
    for (stat_profile, optimal_roles), pairs in chosen:
        env_key = f"{stat_profile}__{optimal_roles}"
        max_stages = max(len(rec["stage_roles"]) for rec, _ in pairs)
        n_teams = len(pairs)

        human = np.zeros((3, max_stages, 3))  # pos, stage, role
        human_counts = np.zeros((3, max_stages))
        walk = np.zeros((3, max_stages, 3))
        walk_counts = np.zeros((3, max_stages))
        belief = np.zeros((3, max_stages, 3))
        belief_counts = np.zeros((3, max_stages))

        # We need per-position role probability predictions, not the pooled
        # `model_marginal`. The model factories don't expose this directly,
        # so recompute by inspecting each stage's per-agent distribution.
        # The `predicted_dist` over 27 combos lets us reconstruct per-position
        # marginals.
        for rec, traj in pairs:
            stage_roles = rec["stage_roles"]
            walk_preds = walk_predict_all(rec)
            belief_preds = belief_predict_all(rec)

            for s in range(len(stage_roles)):
                combo = stage_roles[s]
                for pos in range(3):
                    role = ROLE_CHAR_TO_IDX[combo[pos]]
                    human[pos, s, role] += 1
                    human_counts[pos, s] += 1

                # Reconstruct per-position role distribution from the joint
                # predicted distribution.
                for pred_list, dest, count in [
                    (walk_preds, walk, walk_counts),
                    (belief_preds, belief, belief_counts),
                ]:
                    if s >= len(pred_list):
                        continue
                    pd = pred_list[s]["predicted_dist"]
                    for pos in range(3):
                        for r in range(3):
                            # sum probability of all combos where position pos has role r
                            mass = 0.0
                            for combo_str, p in pd.items():
                                if ROLE_CHAR_TO_IDX[combo_str[pos]] == r:
                                    mass += p
                            dest[pos, s, r] += mass
                        count[pos, s] += 1

        # Normalise to probabilities
        for pos in range(3):
            for s in range(max_stages):
                if human_counts[pos, s] > 0:
                    human[pos, s] /= human_counts[pos, s]
                if walk_counts[pos, s] > 0:
                    walk[pos, s] /= walk_counts[pos, s]
                if belief_counts[pos, s] > 0:
                    belief[pos, s] /= belief_counts[pos, s]
        env_data[env_key] = {
            "stat_profile": stat_profile,
            "optimal_roles": optimal_roles,
            "max_stages": max_stages,
            "n_teams": n_teams,
            "human": human, "walk": walk, "belief": belief,
        }

    # ── Render: rows = envs, sub-rows = (humans / Walk / Belief),
    # cols = 3 player positions.
    n_envs = len(env_data)
    fig, axes = plt.subplots(
        n_envs * 3, 3,
        figsize=(13, 3.2 * n_envs),
        sharey=True, sharex=True,
    )
    if n_envs * 3 == 1:
        axes = np.array([axes])
    elif axes.ndim == 1:
        axes = axes.reshape(-1, 3)

    row_labels = ["Humans", "Bayesian-Walk", "Bayesian-Belief"]
    bar_w = 0.8 / 3.0

    for env_idx, (env_key, d) in enumerate(env_data.items()):
        stat_profile_parts = d["stat_profile"].split("_")
        for sub_row, (label, mat) in enumerate([
            ("Humans", d["human"]),
            ("Bayesian-Walk", d["walk"]),
            ("Bayesian-Belief", d["belief"]),
        ]):
            for pos in range(3):
                ax = axes[env_idx * 3 + sub_row, pos]
                stages = np.arange(d["max_stages"])
                for r in range(3):
                    ax.bar(stages + (r - 1) * bar_w,
                            mat[pos, :, r], width=bar_w,
                            color=ROLE_COLORS[r], edgecolor="white",
                            linewidth=0.4,
                            label=ROLE_NAMES[r] if (env_idx == 0 and sub_row == 0 and pos == 0) else None)
                ax.set_ylim(0, 1)
                ax.set_xticks(stages)
                ax.set_xticklabels([f"S{s+1}" for s in stages], fontsize=8)
                ax.axhline(1 / 3, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)

                if sub_row == 0 and env_idx == 0:
                    profile = stat_profile_parts[pos] if pos < len(stat_profile_parts) else "??"
                    ax.set_title(f"Position {pos+1} (stats {profile})",
                                 fontsize=10, fontweight="bold")
                if pos == 0:
                    if sub_row == 0:
                        ax.set_ylabel(
                            f"{env_key}\nn={d['n_teams']}\n\n{label}",
                            fontsize=9, fontweight="bold",
                        )
                    else:
                        ax.set_ylabel(label, fontsize=9, fontweight="bold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", title="Role",
               ncol=3, frameon=True, fontsize=10,
               bbox_to_anchor=(0.99, 1.0))
    fig.suptitle(
        f"Per-position per-stage role choice, humans vs. Bayesian-Walk vs. Bayesian-Belief\n"
        f"Top-{TOP_K} envs; Walk fit on combo_r "
        f"(τ_softmax={walk_fp['tau_softmax']:.2f}, ε_switch={walk_fp['epsilon_switch']:.2f}); "
        f"each panel shows one player position's role distribution at each of 5 stages",
        fontsize=12, fontweight="bold", y=1.0,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    out = FIGURES_DIR / "fig_qualitative_by_env.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


if __name__ == "__main__":
    main()
