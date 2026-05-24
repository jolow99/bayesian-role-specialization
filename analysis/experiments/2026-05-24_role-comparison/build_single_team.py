"""Role-comparison figure for a single team-round.

Small multiples, one panel per player. Each panel plots role on the
y-axis (F / T / M) against stage on the x-axis, with four step lines:

  - the player's actual choice (thick black)
  - each model's marginal top pick for that player (one line per model)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
METRIC_DIR = EXP_ROOT / "2026-05-12-current-export-metric-comparison"
sys.path.insert(0, str(METRIC_DIR))

from pipeline import (  # noqa: E402
    load_human_team_records, precompute_trajectories, strategy_from_params,
    posterior_marginal, build_joint_dist,
)
from shared.constants import ROLE_CHAR_TO_IDX, ROLE_COLORS, ROLE_SHORT  # noqa: E402
from shared.inference import softmax_role_dist  # noqa: E402

# ──────────────────────────────────────────────────────────────────────
# Params (match 2026-05-12 metric numbers)
# ──────────────────────────────────────────────────────────────────────
TAU_PRIOR = 4.638476144217848
EPSILON = 0.06241645791201582
MEMORY_STRATEGY = "drift_prior_0.500"
TAU_VALUE = 13.71598290227467
TAU_WALK = 7.20651148477258
EPSILON_SWITCH = 0.5589855617201609

MODEL_NAMES = ("Belief", "Value", "Walk")

# Role colours (from shared.constants, role index → hex)
ROLE_C = {0: ROLE_COLORS["F"], 1: ROLE_COLORS["T"], 2: ROLE_COLORS["M"]}

# Model colours — chosen to be distinct from the F/T/M red/blue/green palette
MODEL_C = {
    "Belief": "#d97706",  # amber
    "Value": "#7c3aed",   # violet
    "Walk":  "#0f766e",   # teal
}
MODEL_MARKER = {"Belief": "o", "Value": "s", "Walk": "D"}

CHANCE = 1.0 / 27.0  # uniform over the 27 role combos


# ──────────────────────────────────────────────────────────────────────
# Per-player marginals per stage per model
# ──────────────────────────────────────────────────────────────────────

def compute_per_player_marginals(trajectory, env_config):
    """marginals[model_name] = (n_stages, 3 players, 3 roles)."""
    n_stages = len(trajectory)
    out = {name: np.zeros((n_stages, 3, 3)) for name in MODEL_NAMES}
    values = env_config["values"]
    for s, stage in enumerate(trajectory):
        prior = stage["prior"]
        intent = stage["intent"]
        thp = stage["thp"]
        ehp = stage["ehp"]
        prev_roles = stage["prev_roles"]
        for i in range(3):
            out["Belief"][s, i] = posterior_marginal(prior, i)
            out["Value"][s, i] = softmax_role_dist(
                i, intent, thp, ehp, prior, values, TAU_VALUE)
            if prev_roles is None:
                walk = softmax_role_dist(
                    i, intent, thp, ehp, prior, values, TAU_WALK)
            else:
                switch = softmax_role_dist(
                    i, intent, thp, ehp, prior, values, TAU_WALK)
                stick = np.zeros(3)
                stick[prev_roles[i]] = 1.0
                walk = (1.0 - EPSILON_SWITCH) * stick + EPSILON_SWITCH * switch
            out["Walk"][s, i] = walk
    return out


def joint_combo_dists(per_player):
    """Convert per-player marginals to joint P(combo) per (stage, model).

    Returns dict[model_name] = list of dict[combo_str -> probability], one per stage.
    """
    out = {}
    for name, arr in per_player.items():
        n_stages = arr.shape[0]
        per_stage = []
        for s in range(n_stages):
            per_agent = [arr[s, i] for i in range(3)]
            per_stage.append(build_joint_dist(per_agent))
        out[name] = per_stage
    return out


def map_combo_per_stage(joint_dists):
    """For each model & stage, return (MAP combo string, P(MAP))."""
    out = {}
    for name, per_stage in joint_dists.items():
        out[name] = []
        for combos in per_stage:
            top_combo = max(combos, key=combos.get)
            out[name].append((top_combo, combos[top_combo]))
    return out


# ──────────────────────────────────────────────────────────────────────
# Drawing
# ──────────────────────────────────────────────────────────────────────

def draw_combo_tile(ax, x, y, w, h, combo_str, alpha=1.0, border="#222"):
    """Draw a 3-tile combo glyph: P1 | P2 | P3, each colored by their role.

    combo_str: 3-character string like 'TMF'.
    """
    tile_w = w / 3.0
    for pid, ch in enumerate(combo_str):
        r = ROLE_CHAR_TO_IDX[ch]
        ax.add_patch(mpatches.Rectangle(
            (x + pid * tile_w, y), tile_w, h,
            facecolor=ROLE_C[r], edgecolor="none", alpha=alpha,
        ))
    # Outer border + internal dividers
    ax.add_patch(mpatches.Rectangle(
        (x, y), w, h, facecolor="none", edgecolor=border, linewidth=0.7,
    ))
    for pid in range(1, 3):
        ax.plot([x + pid * tile_w, x + pid * tile_w], [y, y + h],
                color="white", linewidth=0.6, zorder=3)


def render(record, per_player, stat_profile, fig_path):
    """Small multiples: one panel per player. y = role, x = stage.

    Each panel shows 4 step lines: actual choice + each model's marginal
    top pick (argmax over the model's per-player role distribution).
    """
    stage_roles = record["stage_roles"]            # list of "TMF"-style strs
    n_stages = len(stage_roles)
    stages = np.arange(1, n_stages + 1)

    # Actual role index per (stage, player)
    actual = np.array([
        [ROLE_CHAR_TO_IDX[combo[i]] for i in range(3)]
        for combo in stage_roles
    ])  # shape (n_stages, 3)

    # Model top-pick per (stage, player) — argmax over marginals.
    # per_player[name] is (n_stages, 3 players, 3 roles).
    model_pick = {
        name: per_player[name].argmax(axis=2)  # (n_stages, 3 players)
        for name in MODEL_NAMES
    }

    # Tiny vertical offsets so overlapping lines stay readable. The actual
    # line stays on-grid; models shift by small amounts.
    JITTER = {"actual": 0.0, "Belief": +0.12, "Value": -0.12, "Walk": +0.06}

    # ── Figure: 3 stacked panels (one per player) ──────────────────────
    fig_w = max(6.5, 1.6 + n_stages * 1.0)
    fig_h = 5.4
    fig, axes = plt.subplots(
        3, 1, figsize=(fig_w, fig_h), facecolor="white",
        sharex=True, gridspec_kw={"hspace": 0.18},
    )

    title = (f"Env {record['env_id']}  ·  stats {stat_profile}  ·  "
             f"game {record['game_id'][-6:]} r{record['round_number']}")
    fig.text(0.04, 0.965, title, fontsize=11, ha="left", va="center")

    role_labels = ["Fighter", "Tank", "Medic"]  # index 0/1/2

    for i, ax in enumerate(axes):
        # Actual choice — thick dark step line, on the grid.
        ax.plot(
            stages, actual[:, i] + JITTER["actual"],
            color="#111", linewidth=2.2, marker="o", markersize=6,
            drawstyle="steps-mid", zorder=4, label="Actual",
        )

        # Model lines — thinner, colored, slightly offset so they don't
        # disappear when they agree.
        for name in MODEL_NAMES:
            ax.plot(
                stages, model_pick[name][:, i] + JITTER[name],
                color=MODEL_C[name], linewidth=1.4,
                marker=MODEL_MARKER[name], markersize=5.5,
                drawstyle="steps-mid", zorder=3, alpha=0.95,
                label=name,
            )

        # Y axis: role names, tick label coloured by role.
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(role_labels, fontsize=9)
        for tick_lbl, idx in zip(ax.get_yticklabels(), [0, 1, 2]):
            tick_lbl.set_color(ROLE_C[idx])
        ax.set_ylim(-0.55, 2.55)
        # Faint horizontal guides at each role level
        for r in (0, 1, 2):
            ax.axhline(r, color="#eee", linewidth=0.6, zorder=0)

        ax.set_xlim(0.5, n_stages + 0.5)
        ax.set_xticks(stages)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Per-player label on the right (P1/P2/P3 + stat profile slice)
        try:
            stat_parts = stat_profile.split("_")
            stat_slice = stat_parts[i] if i < len(stat_parts) else ""
        except Exception:
            stat_slice = ""
        right_lbl = f"P{i+1}"
        if stat_slice:
            right_lbl += f"\n{stat_slice}"
        ax.text(1.01, 0.5, right_lbl, transform=ax.transAxes,
                ha="left", va="center", fontsize=9, color="#444")

    # Only the bottom panel shows stage labels.
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False, length=0)
    axes[-1].set_xticklabels([f"Stage {s}" for s in stages], fontsize=9)
    axes[-1].tick_params(axis="x", length=0, pad=2)

    # Legend (figure-level, above all panels)
    legend_handles = [
        plt.Line2D([0], [0], color="#111", linewidth=2.2, marker="o",
                   markersize=6, label="Actual"),
    ] + [
        plt.Line2D([0], [0], color=MODEL_C[n], linewidth=1.4,
                   marker=MODEL_MARKER[n], markersize=5.5, label=n)
        for n in MODEL_NAMES
    ]
    fig.legend(
        handles=legend_handles, loc="upper right",
        bbox_to_anchor=(0.98, 0.965),
        ncol=4, frameon=False, fontsize=9, handlelength=2.4,
        columnspacing=1.4,
    )

    fig.savefig(fig_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[render] wrote {fig_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def pick_round(records):
    """First clean human round with >=4 stages and some role variation."""
    for record in records:
        sr = record["stage_roles"]
        if len(sr) < 4:
            continue
        per_player_roles = [[combo[i] for combo in sr] for i in range(3)]
        if not any(len(set(rs)) > 1 for rs in per_player_roles):
            continue
        return record
    return records[0]


def main():
    records = load_human_team_records(verbose=True)
    record = pick_round(records)
    print(f"[main] picked game {record['game_id']} r{record['round_number']} "
          f"env {record['env_id']} stages {record['stage_roles']}")

    strategy = strategy_from_params(MEMORY_STRATEGY, None, None)
    trajectories = precompute_trajectories([record], TAU_PRIOR, EPSILON,
                                            strategy)
    per_player = compute_per_player_marginals(trajectories[0],
                                               record["env_config"])

    out_dir = HERE / "figures"
    out_dir.mkdir(exist_ok=True)
    fig_path = out_dir / (
        f"{record['env_id']}__{record['game_id'][-6:]}_"
        f"r{record['round_number']}.png"
    )
    render(record, per_player, record["stat_profile"], fig_path)


if __name__ == "__main__":
    main()
