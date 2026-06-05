"""Redesigned qualitative single-round renderer (post-meeting layout).

Layout (columns = stages, storyboard style):
  * Top strip: compact team-HP / boss-HP bars at the start of each stage.
  * One horizontal track per player (rows = players, NOT rows = roles):
    each stage cell is a block colored by the chosen role (F=red, T=blue,
    M=green) containing the role's emoji (⚔️ / 🛡️ / 💊).
  * Below each track, one mini-row per teammate: a 3-bar chart of the
    MODEL posterior over that teammate's previous-stage role at the moment
    the human reported their inference, with the human's actual guess
    outlined in black. (Stage 1 has no inferences.)
  * Stages where the team's joint combo equals the value-optimal combo get
    a highlight box around the column, labelled "optimal".
  * Optionally, stages where an identical-stat player pair occupies the
    same role get an orange dashed "mirror" box (symmetry-breaking
    failures).

Geometry is drawn in data coordinates on a single equal-aspect axes.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from common import (
    ROLE_COLORS_IDX, ROLE_EMOJI, compute_posteriors, emoji_array,
    load_stage1, prepare_team, stage_value_vector, target_marginal,
    combo_to_idx, idx_to_combo,
)
from shared.constants import ROLE_NAMES, ROLE_SHORT

# ── geometry constants (data units; axes aspect is equal) ──
COL_W = 1.0          # stage column width
CELL_PAD = 0.06      # padding inside a column
TRACK_H = 0.62       # player role-block height
INF_H = 0.46         # one inference mini-row height
HP_H = 0.42          # HP strip height
BLOCK_GAP = 0.22     # vertical gap between player blocks
LABEL_X = -0.12      # right edge of the row-label margin


def _pad_square(arr):
    """Pad an RGBA crop to square so imshow extents don't distort it."""
    h, w = arr.shape[:2]
    side = max(h, w)
    out = np.zeros((side, side, 4), dtype=arr.dtype)
    y0, x0 = (side - h) // 2, (side - w) // 2
    out[y0:y0 + h, x0:x0 + w] = arr
    return out


def _stage_start_states(team_prs):
    """Logged (team_hp, enemy_hp) at the start of each stage + maxima."""
    rnd = team_prs[0].round
    cfg = rnd.config
    max_thp = cfg.get("maxTeamHealth", 15)
    max_ehp = cfg.get("maxEnemyHealth", 30)
    states = [(float(max_thp), float(max_ehp))]
    for stage in rnd.stages:
        if stage.turns:
            t = stage.turns[-1]
            states.append((float(t["teamHealth"]), float(t["enemyHealth"])))
    return states, max_thp, max_ehp


def render_round_v2(team_prs, record, save_path, title=None,
                    mirror_pair=None, caption=None):
    """team_prs: 3 PlayerRounds sorted by player_id. record: pipeline
    team-round dict (for env values). mirror_pair: (pos_a, pos_b) of
    identical-stat players whose same-role stages get a 'mirror' box."""
    s1, strat = load_stage1()
    rnd = team_prs[0].round

    # Per-player per-stage roles (position order)
    roles_by_player = [[s.role_idx for s in pr.round.stages]
                       for pr in sorted(team_prs, key=lambda p: p.player_id)]
    n_stages = max(len(r) for r in roles_by_player)

    # Model posteriors at the start of each stage (Stage-1 fitted params)
    data = prepare_team(team_prs)
    posteriors = compute_posteriors(data, s1["tau_prior"], s1["epsilon"],
                                    strat)

    # Inference reports: guesses[obs][stage_idx][target] = guessed role idx
    guesses = {pr.player_id: {si: dict(stage.inferred_roles or {})
                              for si, stage in enumerate(pr.round.stages)}
               for pr in team_prs}

    # Stage-start HP (logged) + value-optimal combo per stage
    states, max_thp, max_ehp = _stage_start_states(team_prs)
    optimal_combos, played_combos = [], []
    for s in range(n_stages):
        thp, ehp = states[s] if s < len(states) else states[-1]
        vals = stage_value_vector(record, thp, ehp)
        optimal_combos.append(idx_to_combo(int(np.argmax(vals)))
                              if vals is not None else None)
        played_combos.append("".join(
            ROLE_SHORT[roles_by_player[p][s]] if s < len(roles_by_player[p])
            else "?" for p in range(3)))

    # ── vertical layout (top → bottom) ──
    y = 0.0
    hp_y = y - HP_H                           # HP strip
    y = hp_y - BLOCK_GAP - 0.18
    track_y, inf_y = [], []                   # per player
    teammates = [[1, 2], [0, 2], [0, 1]]
    for p in range(3):
        track_y.append(y - TRACK_H)
        y = track_y[-1]
        rows = []
        for _ in teammates[p]:
            rows.append(y - INF_H)
            y = rows[-1]
        inf_y.append(rows)
        y -= BLOCK_GAP
    y_bottom = y

    fig_w = 1.9 + n_stages * 1.25
    fig_h = (0.6 - y_bottom) * 1.25 + (0.6 if caption else 0.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_aspect("equal")

    def col_x(s):
        return s * COL_W

    # ── HP strip ──
    for s in range(n_stages):
        thp, ehp = states[s] if s < len(states) else states[-1]
        x0 = col_x(s) + CELL_PAD
        w_full = COL_W - 2 * CELL_PAD
        for k, (val, mx, color, lab) in enumerate(
                [(thp, max_thp, "#3498db", "team HP"),
                 (ehp, max_ehp, "#e74c3c", "boss HP")]):
            bar_y = hp_y + HP_H - (k + 1) * (HP_H / 2 - 0.01) - k * 0.02
            bar_h = HP_H / 2 - 0.04
            ax.add_patch(Rectangle((x0, bar_y), w_full, bar_h,
                                   facecolor="#eee", edgecolor="#bbb",
                                   linewidth=0.5))
            ax.add_patch(Rectangle((x0, bar_y), w_full * val / mx, bar_h,
                                   facecolor=color, edgecolor="none",
                                   alpha=0.85))
            ax.text(x0 + w_full - 0.025, bar_y + bar_h / 2,
                    f"{val:.0f}", ha="right", va="center", fontsize=5.5,
                    color="#333", zorder=6)
            if s == 0:
                ax.text(LABEL_X, bar_y + bar_h / 2, lab,
                        ha="right", va="center", fontsize=7, color=color,
                        fontweight="bold")

    # ── stage headers ──
    for s in range(n_stages):
        ax.text(col_x(s) + COL_W / 2, hp_y + HP_H + 0.16, f"Stage {s + 1}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color="#333")

    # ── player tracks + inference mini-rows ──
    emoji_cache = {r: _pad_square(emoji_array(ROLE_EMOJI[r]))
                   for r in range(3) if emoji_array(ROLE_EMOJI[r]) is not None}
    stats_parts = rnd.stat_profile_id.split("_")

    for p in range(3):
        ty = track_y[p]
        ax.text(LABEL_X, ty + TRACK_H / 2,
                f"P{p + 1}\n({'/'.join(stats_parts[p])})", ha="right",
                va="center", fontsize=9, fontweight="bold", color="#222")
        # dividing line above each player block
        ax.axhline(ty + TRACK_H + 0.10, xmin=0.02, xmax=0.98,
                   color="#ccc", linewidth=0.8)

        for s in range(min(n_stages, len(roles_by_player[p]))):
            r = roles_by_player[p][s]
            x0 = col_x(s) + CELL_PAD
            w = COL_W - 2 * CELL_PAD
            ax.add_patch(FancyBboxPatch(
                (x0, ty), w, TRACK_H,
                boxstyle="round,pad=0,rounding_size=0.06",
                facecolor=ROLE_COLORS_IDX[r], edgecolor="white",
                linewidth=1.2, alpha=0.92))
            if r in emoji_cache:
                e = 0.21
                cx = col_x(s) + COL_W / 2
                cy = ty + TRACK_H / 2
                ax.imshow(emoji_cache[r],
                          extent=(cx - e, cx + e, cy - e, cy + e),
                          zorder=5, interpolation="bilinear")
            ax.text(x0 + w - 0.045, ty + TRACK_H - 0.045, ROLE_SHORT[r],
                    ha="right", va="top", fontsize=6.5, color="white",
                    fontweight="bold", zorder=6)

        # inference mini-rows
        for row_i, t in enumerate(teammates[p]):
            ry = inf_y[p][row_i]
            ax.text(LABEL_X, ry + INF_H / 2, f"belief\nabout P{t + 1}",
                    ha="right", va="center", fontsize=6.5, color="#666")
            for s in range(1, n_stages):
                guess = guesses.get(p, {}).get(s, {}).get(t)
                if guess is None or s >= len(posteriors):
                    continue
                marg = target_marginal(posteriors[s], t)
                base_y = ry + 0.05
                max_h = INF_H - 0.13
                bw = 0.16
                for role in range(3):
                    bx = col_x(s) + COL_W / 2 + (role - 1) * (bw + 0.05) - bw / 2
                    h = max(float(marg[role]) * max_h, 0.012)
                    ax.add_patch(Rectangle(
                        (bx, base_y), bw, h,
                        facecolor=ROLE_COLORS_IDX[role],
                        edgecolor="black" if role == guess else "none",
                        linewidth=1.3, alpha=0.95 if role == guess else 0.75,
                        zorder=4))
                    if role == guess:
                        ax.text(bx + bw / 2, base_y - 0.045, "▲", ha="center",
                                va="top", fontsize=5.5, color="#222")

    # ── optimal / mirror column boxes ──
    box_top = hp_y + HP_H + 0.34
    box_bot = inf_y[2][-1] - 0.06
    for s in range(n_stages):
        if optimal_combos[s] is not None and \
                played_combos[s] == optimal_combos[s]:
            ax.add_patch(FancyBboxPatch(
                (col_x(s) + 0.012, box_bot), COL_W - 0.024, box_top - box_bot,
                boxstyle="round,pad=0,rounding_size=0.07",
                facecolor="none", edgecolor="#f1c40f", linewidth=2.4,
                zorder=8))
            ax.text(col_x(s) + COL_W / 2, box_top + 0.05, "optimal",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color="#b7950b", zorder=9)
        elif mirror_pair is not None:
            a, b = mirror_pair
            if (s < len(roles_by_player[a]) and s < len(roles_by_player[b])
                    and roles_by_player[a][s] == roles_by_player[b][s]):
                ax.add_patch(FancyBboxPatch(
                    (col_x(s) + 0.012, box_bot), COL_W - 0.024,
                    box_top - box_bot,
                    boxstyle="round,pad=0,rounding_size=0.07",
                    facecolor="none", edgecolor="#e67e22", linewidth=2.0,
                    linestyle=(0, (4, 2.5)), zorder=8))
                ax.text(col_x(s) + COL_W / 2, box_top + 0.05, "mirror",
                        ha="center", va="bottom", fontsize=9,
                        fontweight="bold", color="#ca6f1e", zorder=9)

    # ── legend ──
    leg_y = y_bottom - 0.18
    lx = 0.0
    for r in range(3):
        ax.add_patch(Rectangle((lx, leg_y - 0.09), 0.18, 0.18,
                               facecolor=ROLE_COLORS_IDX[r],
                               edgecolor="none"))
        ax.text(lx + 0.23, leg_y, ROLE_NAMES[r], ha="left", va="center",
                fontsize=7.5, color="#333")
        lx += 0.23 + 0.135 * len(ROLE_NAMES[r]) + 0.18
    ax.text(lx + 0.15, leg_y,
            "mini-bars: model posterior over teammate's previous-stage "
            "role;  ▲/outline = the human's report",
            ha="left", va="center", fontsize=7.5, color="#555")

    if title:
        ax.text(col_x(0), hp_y + HP_H + 0.62, title, ha="left", va="bottom",
                fontsize=11, fontweight="bold", color="#111")
    if caption:
        ax.text(col_x(0), leg_y - 0.42, caption, ha="left", va="top",
                fontsize=8.5, color="#333", wrap=True)

    ax.set_xlim(-1.55, n_stages * COL_W + 0.32)
    ax.set_ylim((leg_y - (0.9 if caption else 0.25)),
                hp_y + HP_H + (1.0 if title else 0.5))
    ax.axis("off")
    fig.savefig(save_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[qualitative_v2] wrote {save_path}")
    return optimal_combos, played_combos
