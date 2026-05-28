"""Shared 03-24-style qualitative renderer.

Three-panel single-round figure (roles + inferences + HP), styled after
``experiments/2026-03-24_combined_pilot_viz/analysis.ipynb`` —
P1=red / P2=blue / P3=green, action symbols above each turn, dashed
per-player optimal-role lines, red shading on enemy-attack turns,
6 pairwise inference rows (green=correct, red=incorrect), and a
bar-chart HP panel with a "Start" tick.

Usage:
    render_qualitative_human(record, all_prs, save_path,
                              extra_subtitle=None,
                              role_trajectory_caption=None)

`record` is a pipeline team-round dict; `all_prs` is the list returned
by ``shared.data_loading.load_all_exports``; we look up the round in
``all_prs`` to pull per-turn ``action`` strings + per-turn HP straight
from ``gameSummary`` (so the figure stays faithful to what was logged,
not just our replay).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shared.constants import (
    ACTION_SYMBOLS, ROLE_NAMES, ROLE_SHORT, TURNS_PER_STAGE,
)

# 03-24 style palette
PLAYER_COLORS = ["#e74c3c", "#3498db", "#2ecc71"]  # P1=red, P2=blue, P3=green
MARKERS = ["o", "s", "^"]
COLOR_CORRECT = "#2ecc71"
COLOR_INCORRECT = "#e74c3c"
OUTCOME_COLOR = {"WIN": "green", "LOSE": "red", "TIMEOUT": "orange"}


def _flatten_turns(pr):
    """Per-turn dicts (role, action, teamHealth, enemyHealth, stage, inferredRoles) for one PR."""
    out = []
    for stage in pr.round.stages:
        for turn in stage.turns:
            out.append({
                "role": stage.role_idx,
                "action": turn.get("action", "?"),
                "teamHealth": turn.get("teamHealth", 0),
                "enemyHealth": turn.get("enemyHealth", 0),
                "stage": stage.stage,
                "inferredRoles": stage.inferred_roles,
            })
    return out


def _find_team_prs(record, all_prs):
    """Look up the three PlayerRounds for this team-round, sorted by player_id."""
    matches = [
        pr for pr in all_prs
        if pr.round.round_type == "human"
        and pr.game_id == record["game_id"]
        and pr.round.round_number == record["round_number"]
    ]
    matches.sort(key=lambda pr: pr.player_id)
    return matches


def render_qualitative_human(record, all_prs, save_path,
                              extra_subtitle=None,
                              role_trajectory_caption=None):
    """Render a single human team-round, 03-24 style.

    `extra_subtitle`         — optional second-line subtitle.
    `role_trajectory_caption`— optional caption under the figure summarising
                                the team's role trajectory (used by the
                                Section-3 flip-flop figure).
    """
    team_prs = _find_team_prs(record, all_prs)
    if len(team_prs) != 3:
        print(f"  [qual] couldn't find 3 PlayerRounds for "
              f"{record['game_id']} r{record['round_number']}")
        return False

    rnd = team_prs[0].round
    config = rnd.config
    intent = rnd.enemy_intent_sequence
    optimal = rnd.optimal_roles or []

    player_turns = {pr.player_id: _flatten_turns(pr) for pr in team_prs}
    n = max((len(t) for t in player_turns.values()), default=0)
    if n == 0:
        return False
    turn_x = list(range(1, n + 1))
    n_stages = (n + TURNS_PER_STAGE - 1) // TURNS_PER_STAGE

    fig = plt.figure(figsize=(14, 9.5), facecolor="white")
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 2, 1.4], hspace=0.35)
    ax_r = fig.add_subplot(gs[0])
    ax_i = fig.add_subplot(gs[1])
    ax_h = fig.add_subplot(gs[2])

    # ── Roles panel ──
    for pid in sorted(player_turns):
        turns = player_turns[pid]
        roles = [t["role"] for t in turns]
        actions = [t["action"] for t in turns]
        if not roles or all(r == -1 for r in roles):
            continue
        ps = rnd.player_stats
        sl = (f"({ps.get('STR', '?')}/{ps.get('DEF', '?')}/{ps.get('SUP', '?')})"
              if ps else "")
        player_tn = list(range(1, len(roles) + 1))

        ax_r.plot(player_tn, roles, marker=MARKERS[pid],
                  color=PLAYER_COLORS[pid],
                  label=f"P{pid+1} {sl}",
                  markersize=8, linewidth=2, zorder=10)

        for x, r, a in zip(player_tn, roles, actions):
            if r == -1 or a == "?":
                continue
            sym = ACTION_SYMBOLS.get(a, "?")
            ax_r.text(x, r + 0.18 + pid * 0.12, sym,
                       fontsize=7, fontweight="bold",
                       color=PLAYER_COLORS[pid],
                       ha="center", va="bottom",
                       bbox=dict(boxstyle="round,pad=0.08",
                                  facecolor="white",
                                  edgecolor=PLAYER_COLORS[pid],
                                  alpha=0.85, linewidth=0.5))

    # Enemy-attack shading
    for i, ch in enumerate(intent[:n]):
        if ch == "1":
            ax_r.axvspan(i + 0.7, i + 1.3, alpha=0.10, color="red")

    # Per-player optimal role reference lines
    for p, o in enumerate(optimal[:3]):
        ax_r.axhline(y=o, color=PLAYER_COLORS[p], linestyle="--", alpha=0.3)

    # Stage boundaries + labels
    for s in range(1, n_stages):
        x = s * TURNS_PER_STAGE + 0.5
        ax_r.axvline(x, color="#999", linestyle=":", alpha=0.4)
        ax_i.axvline(x, color="#999", linestyle=":", alpha=0.4)
        ax_h.axvline(x - 0.5, color="#999", linestyle=":", alpha=0.4)

    ax_r.set_ylim(-0.5, 3.3)
    ax_r.set_yticks([0, 1, 2])
    ax_r.set_yticklabels(["Fighter", "Tank", "Medic"])
    ax_r.set_xlim(0.5, n + 0.5)
    ax_r.set_xticks(turn_x)
    ax_r.set_xticklabels([str(t) for t in turn_x])
    ax_r.grid(True, alpha=0.3)

    # Stage labels at the top of the role panel (inside the axes)
    for s in range(n_stages):
        x_mid = s * TURNS_PER_STAGE + (TURNS_PER_STAGE + 1) / 2
        if s * TURNS_PER_STAGE >= n:
            continue
        ax_r.text(x_mid, 3.12, f"S{s + 1}",
                   ha="center", va="center", fontsize=9,
                   color="#555", fontweight="bold")

    # Single combined legend: player markers + enemy-attack shading patch.
    # Pinned outside the axes on the right so it can't clip stage labels.
    import matplotlib.patches as mpatches
    player_handles, player_labels = ax_r.get_legend_handles_labels()
    shading_patch = mpatches.Patch(facecolor="red", alpha=0.10,
                                     label="enemy attacks this turn")
    ax_r.legend(player_handles + [shading_patch],
                 player_labels + ["enemy attacks this turn"],
                 loc="upper left", bbox_to_anchor=(1.005, 1.0),
                 fontsize=8, frameon=True, borderaxespad=0)

    outcome = rnd.outcome
    oc = OUTCOME_COLOR.get(outcome, "black")
    config_id = config.get("optimalRolesId", record.get("optimal_roles", "N/A"))
    stat_profile = rnd.stat_profile_id
    game_id = team_prs[0].game_id
    title = (f"Config: {config_id}  |  Stats: {stat_profile}  |  "
             f"Round {rnd.round_number}  |  {outcome}\n"
             f"Game: {game_id}")
    if extra_subtitle:
        title = f"{extra_subtitle}\n{title}"
    ax_r.set_title(title, color=oc, fontweight="bold", fontsize=10)

    # ── Inferences panel ──
    # Rows: P1→P2, P1→P3, P2→P1, P2→P3, P3→P1, P3→P2
    inf_pairs = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    inf_labels = [f"P{o+1}→P{t+1}" for o, t in inf_pairs]
    row_for_pair = {p: i for i, p in enumerate(inf_pairs)}

    for pid in sorted(player_turns):
        turns = player_turns[pid]
        prev_stage = 0
        for ti, t in enumerate(turns):
            stage = t["stage"]
            if stage > prev_stage and prev_stage > 0:
                inferred = t.get("inferredRoles") or {}
                for target_pos, ir in inferred.items():
                    pair = (pid, target_pos)
                    if pair not in row_for_pair:
                        continue
                    ri = row_for_pair[pair]
                    target_turns = player_turns.get(target_pos, [])
                    actual = (target_turns[ti - 1]["role"]
                              if ti > 0 and ti - 1 < len(target_turns)
                              else -1)
                    if actual != -1:
                        correct = (ir == actual)
                        c = COLOR_CORRECT if correct else COLOR_INCORRECT
                    else:
                        c = "#999"
                    ax_i.text(
                        ti + 1, ri, ROLE_SHORT.get(ir, "?"),
                        fontsize=9, fontweight="bold",
                        color=c, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.12",
                                   facecolor="white", edgecolor=c, alpha=0.9),
                    )
            prev_stage = stage

    ax_i.set_ylim(-0.5, 5.5)
    ax_i.set_yticks(range(6))
    ax_i.set_yticklabels(inf_labels, fontsize=8)
    ax_i.set_xlim(0.5, n + 0.5)
    ax_i.set_xticks(turn_x)
    ax_i.set_xticklabels([])
    ax_i.set_title("Inferences  (green = correct, red = incorrect)",
                    fontsize=9)
    ax_i.grid(True, alpha=0.2)

    # ── HP panel ──
    max_th = config.get("maxTeamHealth", record["env_config"].get("team_max_hp", 10))
    max_eh = config.get("maxEnemyHealth", record["env_config"].get("enemy_max_hp", 30))
    first_pid = sorted(player_turns)[0]
    first_turns = player_turns[first_pid]
    actual_n = len(first_turns)
    th = [max_th] + [t["teamHealth"] for t in first_turns]
    eh = [max_eh] + [t["enemyHealth"] for t in first_turns]
    health_x = np.arange(0, actual_n + 1)
    w = 0.35
    ax_h.bar(health_x - w / 2, th, w, label="Team HP", color="#3498db", alpha=0.7)
    ax_h.bar(health_x + w / 2, eh, w, label="Enemy HP", color="#e74c3c", alpha=0.7)
    ax_h.set_xlim(-0.5, actual_n + 0.5)
    ax_h.set_xticks(health_x)
    ax_h.set_xticklabels(["Start"] + [str(i) for i in range(1, actual_n + 1)],
                          fontsize=7)
    ax_h.set_xlabel("Turn")
    ax_h.set_ylabel("Health")
    ax_h.legend(loc="upper right", fontsize=8)
    ax_h.grid(True, alpha=0.3, axis="y")

    if role_trajectory_caption:
        fig.text(0.5, 0.005, role_trajectory_caption,
                  ha="center", va="bottom", fontsize=10,
                  color="#222", fontweight="bold")

    fig.tight_layout(rect=(0, 0.02 if role_trajectory_caption else 0, 1, 1))
    fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True
