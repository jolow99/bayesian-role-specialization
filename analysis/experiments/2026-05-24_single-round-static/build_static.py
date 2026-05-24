"""Single-round static Tufte-style figure.

Prototype for a *static* (non-animated) single-round visualization that
condenses what the existing GIF frames spread across stages. Goal: fit
everything a reader needs to evaluate one round on one page, without
hiding turn vs stage structure or relegating model predictions to a
footer.

Panels (sharing x = turn, stage boundaries marked):
  1. Roles + actions + model-belief overlay
     - 3 step-lines per player at y in {F, T, M}
     - Per-turn markers shaped by action (▲ ATTACK, ■ BLOCK, ♥ HEAL)
     - At each stage boundary, 3 disks per player per role (one per
       model) sized by that model's marginal P(role) for that player
  2. Inference grid (2 cols × 3 rows) for the 6 observer→target pairs
  3a. Turn-level HP (team + enemy areas) with enemy-attack bands
  3b. Per-stage P(team's actual combo) for each model + P(optimal) ref
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
METRIC_DIR = EXP_ROOT / "2026-05-12-current-export-metric-comparison"
sys.path.insert(0, str(METRIC_DIR))

from pipeline import (  # noqa: E402
    load_human_team_records, precompute_trajectories, strategy_from_params,
    posterior_marginal,
)
from shared import EXPORTS_DIR  # noqa: E402
from shared.constants import (  # noqa: E402
    ROLE_COLORS, ROLE_CHAR_TO_IDX, ROLE_SHORT, TURNS_PER_STAGE,
)
from shared.data_loading import load_all_exports  # noqa: E402
from shared.inference import (  # noqa: E402
    preferred_action, game_step, softmax_role_dist,
)

# ──────────────────────────────────────────────────────────────────────
# Tunable params (match 2026-05-23 trajectory GIFs / published numbers)
# ──────────────────────────────────────────────────────────────────────
TAU_PRIOR = 4.638476144217848
EPSILON = 0.06241645791201582
MEMORY_STRATEGY = "drift_prior_0.500"
TAU_VALUE = 13.71598290227467
TAU_WALK = 7.20651148477258
EPSILON_SWITCH = 0.5589855617201609

# ──────────────────────────────────────────────────────────────────────
# Visual constants
# ──────────────────────────────────────────────────────────────────────
PLAYER_COLORS = ["#9b59b6", "#e67e22", "#16a085"]  # P1, P2, P3
MODEL_COLORS = {
    "Belief": "#1f3a93",
    "Value":  "#c0392b",
    "Walk":   "#16a085",
}
ACTION_MARKERS = {0: "^", 1: "s", 2: "v"}  # ATTACK, BLOCK, HEAL
ACTION_NAMES = {0: "ATTACK", 1: "BLOCK", 2: "HEAL"}
ROLE_Y = {"M": 2, "T": 1, "F": 0}
ROLE_LABEL = {0: "F", 1: "T", 2: "M"}


# ──────────────────────────────────────────────────────────────────────
# Data assembly
# ──────────────────────────────────────────────────────────────────────

def assemble_round(record, pr_lookup):
    """Combine team record + corresponding PlayerRounds into one bundle."""
    env = record["env_config"]
    player_stats = env["player_stats"]
    team_max_hp = env["team_max_hp"]
    enemy_max_hp = env["enemy_max_hp"]
    boss_damage = env["boss_damage"]
    lds = record["lds"]
    stage_roles = record["stage_roles"]
    n_stages = len(stage_roles)

    # Turn-by-turn replay (HP, actions, intent)
    team_hp = float(team_max_hp)
    enemy_hp = float(enemy_max_hp)
    turn_hp = [(team_hp, enemy_hp)]
    turn_intent = []
    turn_actions = []
    turn_idx = 0
    for s in range(n_stages):
        roles = [ROLE_CHAR_TO_IDX[c] for c in stage_roles[s]]
        for _ in range(TURNS_PER_STAGE):
            if turn_idx >= len(lds):
                break
            intent = int(lds[turn_idx])
            actions = [preferred_action(roles[i], intent, team_hp, team_max_hp)
                       for i in range(3)]
            new_thp, new_ehp = game_step(intent, team_hp, enemy_hp, actions,
                                          player_stats, boss_damage, team_max_hp)
            turn_intent.append(intent)
            turn_actions.append(actions)
            turn_hp.append((new_thp, new_ehp))
            team_hp, enemy_hp = new_thp, new_ehp
            turn_idx += 1

    # Inferences (per observer per stage)
    inferences = {}  # (observer, target) -> list of (stage, inferred_role)
    for pid in range(3):
        pr = pr_lookup[pid]
        for s, stage in enumerate(pr.round.stages):
            if s == 0 or not stage.inferred_roles:
                continue
            for target_pos, inferred_role in stage.inferred_roles.items():
                key = (pid, target_pos)
                inferences.setdefault(key, []).append((s, inferred_role))

    outcome = pr_lookup[0].round.outcome

    return {
        "n_stages": n_stages,
        "stage_roles": stage_roles,
        "turn_hp": turn_hp,
        "turn_intent": turn_intent,
        "turn_actions": turn_actions,
        "inferences": inferences,
        "player_stats": player_stats,
        "team_max_hp": team_max_hp,
        "enemy_max_hp": enemy_max_hp,
        "optimal": record["optimal_roles"],
        "stat_profile": record["stat_profile"],
        "env_id": record["env_id"],
        "game_id": record["game_id"],
        "round_number": record["round_number"],
        "outcome": outcome,
    }


# ──────────────────────────────────────────────────────────────────────
# Model predictions (per-player marginals per stage)
# ──────────────────────────────────────────────────────────────────────

def per_player_marginals(trajectory, env_config):
    """Return dict[model_name] = (n_stages, 3, 3) array — [stage, player, role]."""
    n_stages = len(trajectory)
    out = {name: np.zeros((n_stages, 3, 3)) for name in MODEL_COLORS}

    values = env_config["values"]

    for s, stage in enumerate(trajectory):
        prior = stage["prior"]
        intent = stage["intent"]
        thp = stage["thp"]
        ehp = stage["ehp"]
        prev_roles = stage["prev_roles"]

        for i in range(3):
            belief_marg = posterior_marginal(prior, i)
            out["Belief"][s, i] = belief_marg

            value_marg = softmax_role_dist(i, intent, thp, ehp, prior,
                                            values, TAU_VALUE)
            out["Value"][s, i] = value_marg

            if prev_roles is None:
                walk_marg = softmax_role_dist(i, intent, thp, ehp, prior,
                                               values, TAU_WALK)
            else:
                switch = softmax_role_dist(i, intent, thp, ehp, prior,
                                            values, TAU_WALK)
                stick = np.zeros(3)
                stick[prev_roles[i]] = 1.0
                walk_marg = (1.0 - EPSILON_SWITCH) * stick + EPSILON_SWITCH * switch
            out["Walk"][s, i] = walk_marg

    return out


def per_stage_combo_probs(per_player, combo_str):
    """Multiply per-player marginals for a given combo across stages.

    per_player[model] is (n_stages, 3, 3). combo_str is 'TFM'. Returns
    dict[model_name] = (n_stages,) array of P(combo).
    """
    target = [ROLE_CHAR_TO_IDX[c] for c in combo_str]
    out = {}
    for name, arr in per_player.items():
        n_stages = arr.shape[0]
        probs = np.zeros(n_stages)
        for s in range(n_stages):
            probs[s] = arr[s, 0, target[0]] * arr[s, 1, target[1]] * arr[s, 2, target[2]]
        out[name] = probs
    return out


# ──────────────────────────────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────────────────────────────

def render(bundle, per_player, fig_path):
    n_stages = bundle["n_stages"]
    n_turns = len(bundle["turn_actions"])
    stage_roles = bundle["stage_roles"]
    turn_hp = bundle["turn_hp"]
    turn_intent = bundle["turn_intent"]
    turn_actions = bundle["turn_actions"]
    inferences = bundle["inferences"]
    team_max_hp = bundle["team_max_hp"]
    enemy_max_hp = bundle["enemy_max_hp"]
    optimal = bundle["optimal"]
    stat_profile = bundle["stat_profile"]
    player_stats = bundle["player_stats"]

    fig = plt.figure(figsize=(12, 9), facecolor="white")
    gs = GridSpec(
        nrows=4, ncols=1,
        height_ratios=[2.6, 1.1, 1.0, 0.9],
        hspace=0.35, left=0.10, right=0.96, top=0.89, bottom=0.07,
    )
    ax_top = fig.add_subplot(gs[0])
    ax_inf = fig.add_subplot(gs[1])
    ax_hp = fig.add_subplot(gs[2])
    ax_mod = fig.add_subplot(gs[3])

    # Shared x-extent: turn 0.5 .. n_turns + 0.5
    x_lo, x_hi = 0.5, n_turns + 0.5

    # ── Title ────────────────────────────────────────────────────────
    outcome_color = {"WIN": "#27ae60", "LOSE": "#c0392b",
                      "TIMEOUT": "#e67e22"}.get(bundle["outcome"], "#666")
    title = (f"Env {bundle['env_id']}  ·  stats {stat_profile}  ·  "
             f"optimal {optimal}  ·  game {bundle['game_id'][-6:]} "
             f"r{bundle['round_number']}")
    fig.text(0.10, 0.975, title, fontsize=11, ha="left", va="center")
    fig.text(0.96, 0.975, bundle["outcome"], fontsize=11, ha="right",
              va="center", color=outcome_color, fontweight="bold")

    # ──────────────────────────────────────────────────────────────────
    # Panel 1 — Roles + actions + model-belief overlay
    # ──────────────────────────────────────────────────────────────────
    ax_top.set_xlim(x_lo, x_hi)
    ax_top.set_ylim(-0.4, 2.4)
    # Convention: Fighter at top, Tank middle, Medic bottom
    ax_top.invert_yaxis()
    ax_top.set_yticks([0, 1, 2])
    ax_top.set_yticklabels(["Fighter", "Tank", "Medic"], fontsize=10)
    ax_top.set_ylabel("Role", fontsize=10)
    ax_top.tick_params(axis="x", labelbottom=False, length=0)

    # Stage backgrounds: alternating very-light gray for stage demarcation
    for s in range(n_stages):
        if s % 2 == 1:
            ax_top.axvspan(s * TURNS_PER_STAGE + 0.5,
                            (s + 1) * TURNS_PER_STAGE + 0.5,
                            color="#000", alpha=0.03, zorder=0)

    # Stage boundary verticals
    for s in range(1, n_stages):
        x = s * TURNS_PER_STAGE + 0.5
        ax_top.axvline(x, color="#888", linewidth=0.5, alpha=0.4, zorder=0.5)

    # Stage labels at the (visual) top of the panel (above Fighter row)
    for s in range(n_stages):
        x_mid = s * TURNS_PER_STAGE + 1.5
        ax_top.text(x_mid, -0.32, f"S{s+1}", ha="center", va="center",
                     fontsize=8, color="#888")

    # Enemy-attack markers: small red ▼ above each attack turn (one per turn)
    for ti, intent in enumerate(turn_intent):
        if intent == 1:
            ax_top.scatter([ti + 1], [-0.18], marker="v", s=28,
                            color="#e74c3c", edgecolor="none", zorder=5)

    # Reference horizontal grid lines (faint)
    for y in (0, 1, 2):
        ax_top.axhline(y, color="#ddd", linewidth=0.5, zorder=0.5)

    # Role step-lines (one per player) + action markers
    for pid in range(3):
        pcolor = PLAYER_COLORS[pid]
        # Step line through stages
        xs, ys = [], []
        for s in range(n_stages):
            role_char = stage_roles[s][pid]
            y = ROLE_Y[role_char]
            xs.extend([s * TURNS_PER_STAGE + 0.5,
                       (s + 1) * TURNS_PER_STAGE + 0.5])
            ys.extend([y, y])
        ax_top.plot(xs, ys, color=pcolor, linewidth=1.6, alpha=0.85,
                     zorder=3, solid_joinstyle="miter")

        # Per-turn action markers AT role y-position
        for ti, actions in enumerate(turn_actions):
            s = ti // TURNS_PER_STAGE
            role_char = stage_roles[s][pid]
            y = ROLE_Y[role_char]
            marker = ACTION_MARKERS[actions[pid]]
            ax_top.scatter([ti + 1], [y], marker=marker, s=42,
                            color=pcolor, edgecolor="white", linewidth=0.6,
                            zorder=4)

    # No per-player model overlay on the role panel. Per-stage joint
    # P(combo) predictions live in the bottom panel where they belong.

    # Player legend (top right)
    player_legend = [
        mpatches.Patch(color=PLAYER_COLORS[i],
                        label=f"P{i+1}  {stat_profile.split('_')[i]}")
        for i in range(3)
    ]
    ax_top.legend(handles=player_legend, loc="upper right",
                   fontsize=8, frameon=False, ncol=3,
                   bbox_to_anchor=(1.0, 1.09))

    for spine in ("top", "right"):
        ax_top.spines[spine].set_visible(False)
    ax_top.spines["left"].set_color("#888")
    ax_top.spines["bottom"].set_color("#888")

    # ──────────────────────────────────────────────────────────────────
    # Panel 2 — Inference grid
    # ──────────────────────────────────────────────────────────────────
    # 6 (observer→target) pairs, arranged in 2 columns of 3 rows
    pairs = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    row_for_pair = {p: i % 3 for i, p in enumerate(pairs)}
    col_for_pair = {p: i // 3 for i, p in enumerate(pairs)}

    ax_inf.set_xlim(x_lo, x_hi)
    ax_inf.set_ylim(-0.5, 2.5)
    ax_inf.tick_params(axis="both", labelbottom=False, labelleft=False, length=0)
    for spine in ("top", "right", "left", "bottom"):
        ax_inf.spines[spine].set_visible(False)

    # Two-column layout: split x axis into halves visually but inferences
    # still index by stage. We draw both columns at stage-boundary x.
    # Column 0 (left half): pairs index 0,1,2  Column 1 (right half): 3,4,5
    n_half_turns = n_turns / 2.0

    def x_for(col, s):
        # within column, stages 0..n_stages-1 spread across half axis
        col_left = x_lo if col == 0 else x_lo + n_half_turns
        col_right = x_lo + n_half_turns if col == 0 else x_hi
        col_pad = 0.5
        stage_w = (col_right - col_left - 2 * col_pad) / max(n_stages, 1)
        return col_left + col_pad + (s + 0.5) * stage_w

    # Faint column divider
    ax_inf.axvline(x_lo + n_half_turns, color="#ccc", linewidth=0.5)

    # Row labels at left of each column
    for pair in pairs:
        r = row_for_pair[pair]
        c = col_for_pair[pair]
        observer, target = pair
        label = f"P{observer+1}→P{target+1}"
        x_label = x_lo + 0.05 if c == 0 else x_lo + n_half_turns + 0.05
        ax_inf.text(x_label, 2 - r, label, fontsize=8,
                     color=PLAYER_COLORS[observer], va="center", ha="left")

    # Inference letters at each stage (only s >= 1)
    for pair, entries in inferences.items():
        if pair not in row_for_pair:
            continue
        r = row_for_pair[pair]
        c = col_for_pair[pair]
        observer, target = pair
        for s, inferred_role in entries:
            # Truth: target's role in stage s-1
            if s - 1 >= n_stages:
                continue
            true_char = stage_roles[s - 1][target]
            inferred_char = ROLE_LABEL[inferred_role]
            correct = (inferred_char == true_char)
            box_color = "#27ae60" if correct else "#e74c3c"
            x = x_for(c, s)
            y = 2 - r
            ax_inf.add_patch(mpatches.FancyBboxPatch(
                (x - 0.20, y - 0.30), 0.40, 0.60,
                boxstyle="round,pad=0.02",
                facecolor=box_color, edgecolor="none", alpha=0.85, zorder=2,
            ))
            ax_inf.text(x, y, inferred_char, ha="center", va="center",
                         color="white", fontsize=9, fontweight="bold", zorder=3)

    # Stage markers at top of inference panel
    for s in range(n_stages):
        for c in (0, 1):
            x = x_for(c, s)
            ax_inf.text(x, 2.45, f"S{s+1}", ha="center", va="center",
                         fontsize=7, color="#bbb")

    ax_inf.set_ylabel("Inferences", fontsize=10, color="#666")

    # ──────────────────────────────────────────────────────────────────
    # Panel 3a — HP
    # ──────────────────────────────────────────────────────────────────
    turns_x = np.arange(0, n_turns + 1)  # turn 0 = pre-game state
    team_hps = np.array([h[0] for h in turn_hp])
    enemy_hps = np.array([h[1] for h in turn_hp])

    # Normalize by max so both share a 0..1 axis
    ax_hp.fill_between(turns_x + 0.5, 0, team_hps / team_max_hp,
                        color="#3498db", alpha=0.35, step="post", zorder=2,
                        label="Team HP")
    ax_hp.fill_between(turns_x + 0.5, 0, enemy_hps / enemy_max_hp,
                        color="#e74c3c", alpha=0.20, step="post", zorder=1,
                        label="Enemy HP")
    ax_hp.plot(turns_x + 0.5, team_hps / team_max_hp,
                color="#2980b9", linewidth=1.4, drawstyle="steps-post", zorder=3)
    ax_hp.plot(turns_x + 0.5, enemy_hps / enemy_max_hp,
                color="#c0392b", linewidth=1.4, drawstyle="steps-post", zorder=3)

    # Stage alternating background to match top panel
    for s in range(n_stages):
        if s % 2 == 1:
            ax_hp.axvspan(s * TURNS_PER_STAGE + 0.5,
                           (s + 1) * TURNS_PER_STAGE + 0.5,
                           color="#000", alpha=0.03, zorder=0)

    # Stage boundary verticals
    for s in range(1, n_stages):
        x = s * TURNS_PER_STAGE + 0.5
        ax_hp.axvline(x, color="#888", linewidth=0.5, alpha=0.4)

    # Enemy-attack markers above HP plot (aligned with top panel)
    for ti, intent in enumerate(turn_intent):
        if intent == 1:
            ax_hp.scatter([ti + 1], [1.04], marker="v", s=22,
                           color="#e74c3c", edgecolor="none",
                           zorder=5, clip_on=False)

    ax_hp.set_xlim(x_lo, x_hi)
    ax_hp.set_ylim(0, 1.02)
    ax_hp.set_yticks([0, 0.5, 1.0])
    ax_hp.set_yticklabels(["0", "½", "max"], fontsize=8)
    ax_hp.set_ylabel("HP (fraction\nof max)", fontsize=9)
    ax_hp.tick_params(axis="x", labelbottom=False, length=0)
    for spine in ("top", "right"):
        ax_hp.spines[spine].set_visible(False)
    ax_hp.spines["left"].set_color("#888")
    ax_hp.spines["bottom"].set_color("#888")

    # Annotate endpoint HP values
    ax_hp.text(n_turns + 0.5, team_hps[-1] / team_max_hp,
                f"  {int(team_hps[-1])}", color="#2980b9", fontsize=8,
                va="center")
    ax_hp.text(n_turns + 0.5, enemy_hps[-1] / enemy_max_hp,
                f"  {int(enemy_hps[-1])}", color="#c0392b", fontsize=8,
                va="center")

    # ──────────────────────────────────────────────────────────────────
    # Panel 3b — Model probability strip
    # ──────────────────────────────────────────────────────────────────
    # Per-stage P(actual combo) line per model. Stage-step plot (constant
    # within stage), x in turn units.
    p_actual = {name: np.zeros(n_stages) for name in MODEL_COLORS}
    p_optimal = {name: np.zeros(n_stages) for name in MODEL_COLORS}
    for s in range(n_stages):
        combo_actual = stage_roles[s]
        combo_probs = per_stage_combo_probs(per_player, combo_actual)
        opt_probs = per_stage_combo_probs(per_player, optimal)
        for name in MODEL_COLORS:
            p_actual[name][s] = combo_probs[name][s]
            p_optimal[name][s] = opt_probs[name][s]

    def stage_step(x_axis, vals):
        """Convert per-stage values to step-friendly (xs, ys)."""
        xs, ys = [], []
        for s, v in enumerate(vals):
            xs.extend([s * TURNS_PER_STAGE + 0.5,
                       (s + 1) * TURNS_PER_STAGE + 0.5])
            ys.extend([v, v])
        return xs, ys

    for name in ("Belief", "Value", "Walk"):
        xs, ys = stage_step(None, p_actual[name])
        ax_mod.plot(xs, ys, color=MODEL_COLORS[name], linewidth=1.6,
                     alpha=0.9, label=f"{name} P(chosen)")
        xs, ys = stage_step(None, p_optimal[name])
        ax_mod.plot(xs, ys, color=MODEL_COLORS[name], linewidth=1.0,
                     alpha=0.55, linestyle="--", label=f"{name} P(optimal)")

    # Chance reference: 1/27
    ax_mod.axhline(1.0 / 27, color="#888", linewidth=0.6, linestyle=":")
    ax_mod.text(x_hi, 1.0 / 27, "  1/27", color="#888", fontsize=7,
                 va="center")

    # Stage boundaries
    for s in range(1, n_stages):
        x = s * TURNS_PER_STAGE + 0.5
        ax_mod.axvline(x, color="#888", linewidth=0.5, alpha=0.4)

    ax_mod.set_xlim(x_lo, x_hi)
    y_max = max(
        max(p_actual[name].max() for name in MODEL_COLORS),
        max(p_optimal[name].max() for name in MODEL_COLORS),
    )
    y_top = min(1.02, max(0.25, y_max * 1.25))
    ax_mod.set_ylim(0, y_top)
    if y_top <= 0.30:
        ticks = [0, 0.1, 0.2]
    elif y_top <= 0.55:
        ticks = [0, 0.25, 0.5]
    else:
        ticks = [0, 0.5, 1.0]
    ax_mod.set_yticks(ticks)
    ax_mod.set_yticklabels([f"{t:g}" for t in ticks], fontsize=8)
    ax_mod.set_ylabel("P(combo)\nper stage", fontsize=9)
    ax_mod.set_xticks(np.arange(1, n_turns + 1))
    ax_mod.set_xticklabels([str(i) for i in range(1, n_turns + 1)], fontsize=8)
    ax_mod.set_xlabel("turn", fontsize=9)
    for spine in ("top", "right"):
        ax_mod.spines[spine].set_visible(False)
    ax_mod.spines["left"].set_color("#888")
    ax_mod.spines["bottom"].set_color("#888")
    ax_mod.legend(loc="upper right", fontsize=7, frameon=False,
                   ncol=3, bbox_to_anchor=(1.0, 1.25))

    # Footer: action marker key
    fig.text(0.10, 0.015,
              "action markers:  ▲ ATTACK   ■ BLOCK   ▼ HEAL  "
              "·  red ▼ above panel = enemy attacks this turn  "
              "·  alternating gray bands = stage separators  "
              "·  dashed lines (bottom panel) = P(optimal)",
              fontsize=8, color="#666")

    fig.savefig(fig_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[render] wrote {fig_path}")


# ──────────────────────────────────────────────────────────────────────
# Main: pick a round and render
# ──────────────────────────────────────────────────────────────────────

def pick_round(records, all_prs):
    """Pick a clean human round with >=4 stages and some role variation."""
    # Lookup PlayerRounds by (export, game, round_number, player_id)
    pr_lookup_by_round = {}
    for pr in all_prs:
        if pr.round.round_type != "human":
            continue
        key = (pr.export_name, pr.game_id, pr.round.round_number)
        pr_lookup_by_round.setdefault(key, {})[pr.player_id] = pr

    candidates = []
    for record in records:
        key = (record["export_name"], record["game_id"], record["round_number"])
        prs = pr_lookup_by_round.get(key, {})
        if len(prs) != 3:
            continue
        sr = record["stage_roles"]
        if len(sr) < 4:
            continue
        # Has role variation? At least one player changes role across stages
        per_player_roles = [[combo[i] for combo in sr] for i in range(3)]
        if not any(len(set(rs)) > 1 for rs in per_player_roles):
            continue
        candidates.append((record, prs))

    if not candidates:
        # Fallback: just take any 4+stage record
        for record in records:
            key = (record["export_name"], record["game_id"], record["round_number"])
            prs = pr_lookup_by_round.get(key, {})
            if len(prs) == 3 and len(record["stage_roles"]) >= 4:
                candidates.append((record, prs))
                break

    if not candidates:
        raise RuntimeError("No suitable round found.")

    # Pick first
    return candidates[0]


def main():
    print("[main] loading human team records ...")
    records = load_human_team_records(verbose=True)

    print("[main] loading raw player-rounds for inferences ...")
    all_prs = load_all_exports()

    record, pr_lookup = pick_round(records, all_prs)
    print(f"[main] picked game {record['game_id']} r{record['round_number']} "
          f"env {record['env_id']} stages {record['stage_roles']}")

    strategy = strategy_from_params(MEMORY_STRATEGY, None, None)
    trajectories = precompute_trajectories([record], TAU_PRIOR, EPSILON, strategy)
    trajectory = trajectories[0]

    bundle = assemble_round(record, pr_lookup)
    per_player = per_player_marginals(trajectory, record["env_config"])

    out_dir = HERE / "figures"
    out_dir.mkdir(exist_ok=True)
    fig_path = out_dir / f"{record['env_id']}__{record['game_id'][-6:]}_r{record['round_number']}.png"
    render(bundle, per_player, fig_path)


if __name__ == "__main__":
    main()
