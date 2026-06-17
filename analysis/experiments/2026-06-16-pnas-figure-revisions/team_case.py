"""R3_team_case — human-team analogue of the bot-round adaptation
storyboard (R3_adaptation_case), for the rational + sticky-choice claim.

Pinned case (chosen 2026-06-16, see case_search.py / README): a fully
symmetric human team (222_222_222) that starts badly mis-coordinated
(TMT, value-rank 26/27), shows STICKINESS — at stage 2 two players lag
the Bayesian best-response (P2 stays Medic, P3 stays Tank while
best-response says Fighter for both) — then RELENTS (both switch to
Fighter at stage 3), reaching the value-optimal combo (FFF, rank 1) and
WINNING. P1 is the early best-responder (switches at stage 2). This is
the visual signature of role choice as a mixture of rational
value-seeking and sticky inertia (cf. the Bayesian Walk-PS fit in
R2_individual_fitting).

Panels (same structure as the bot-round version, generalized to 3
humans, top → bottom):
  1. stage headers + faint dividers + per-turn ticks
  2. HP strip — per-turn paired mini-bars (LOGGED team/boss HP)
  3-5. one role track per human — role card per stage (vector Twemoji
       icon + role letter), logged A/B/H per turn, best-response flag
       when the model would switch, stickiness highlight + relent arrows
  6-8. one belief row per TARGET player — Bayesian observer posterior
       marginal over that player's role per stage; the (up to 2)
       teammates' reports overlaid (caret under the named role, labeled
       by reporter; filled = matches the target's true previous role)
  9. enemy-intent row
 10. compact legend (role icons, not color-reliant; marker meanings)

Role icons are TRUE VECTOR Twemoji (svg_icons.draw_role_icon) — embedded
as real paths in the PDF, multicolor, no rasterization.

Run from analysis/:
    uv run python experiments/2026-06-16-pnas-figure-revisions/team_case.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from common_human import (  # noqa: E402
    OUT_DIR, best_response, human_posteriors, load_human_records,
    load_stage1_canonical, stage_value_rank, target_marginal,
)
from svg_icons import draw_role_icon  # noqa: E402
from shared.constants import ROLE_NAMES, ROLE_SHORT  # noqa: E402

# Pinned case.
CASE_GAME_ID = "01KRBT30X9RGSB5Q5P3PX48RYB"
CASE_ROUND = 4

ROLE_COLORS = {0: "#e74c3c", 1: "#3498db", 2: "#2ecc71"}   # F / T / M
TEAM_HP_COLOR = "#3498db"
ENEMY_HP_COLOR = "#e74c3c"
BR_COLOR = "#b8860b"        # best-response / stickiness accent (dark gold)
RELENT_COLOR = "#27ae60"    # the rational switch
GREY = "#999999"
ACTION_NAME_TO_LETTER = {"ATTACK": "A", "BLOCK": "B", "HEAL": "H"}

# ── geometry (data units; axes aspect is equal) ──
COL_W = 1.0
TURN_W = COL_W / 2
CELL_PAD = 0.05
TRACK_H = 0.56
HP_H = 0.40
BELIEF_H = 0.40
INTENT_H = 0.16
BLOCK_GAP = 0.10
LABEL_X = -0.14

FIG_W_IN = 7.0

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
})


def col_x(s):
    return s * COL_W


def turn_x(s, j):
    return s * COL_W + (j + 0.5) * TURN_W


def flatten_turns(rec):
    """Per-turn dicts: global turn idx, intent, logged team/enemy HP,
    and each player's logged action letter."""
    out = []
    t = 0
    for s in range(rec["n_stages"]):
        for j, turn in enumerate(rec["stage_turns"][s]):
            out.append({
                "s": s, "j": j, "t": t,
                "intent": rec["turn_intent"][t] if t < len(rec["turn_intent"]) else 0,
                "thp": float(turn["team_hp"]),
                "ehp": float(turn["enemy_hp"]),
                "actions": {pid: ACTION_NAME_TO_LETTER.get(a, "?")
                            for pid, a in turn["actions"].items()},
            })
            t += 1
    return out


# ──────────────────────────────────────────────────────────────────────
# Renderer pieces
# ──────────────────────────────────────────────────────────────────────

def _draw_stage_headers(ax, n_stages, turns, y_top, y_bottom):
    for s in range(n_stages):
        ax.text(col_x(s) + COL_W / 2, y_top + 0.30, f"Stage {s + 1}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                color="#333")
    for s in range(n_stages + 1):
        ax.plot([col_x(s), col_x(s)], [y_bottom, y_top + 0.26],
                color="#ddd", linewidth=0.6, zorder=0)
    for tu in turns:
        ax.text(turn_x(tu["s"], tu["j"]), y_top + 0.15, f"{tu['t'] + 1}",
                ha="center", va="bottom", fontsize=5, color="#aaa")
    ax.text(LABEL_X, y_top + 0.15, "turn", ha="right", va="bottom",
            fontsize=5, color="#aaa")


def _draw_hp_strip(ax, rec, turns, hp_y, ranks):
    max_thp, max_ehp = rec["team_max_hp"], rec["enemy_max_hp"]
    bw = 0.15
    last_turn_of_stage = {}
    for tu in turns:
        last_turn_of_stage[tu["s"]] = tu["t"]
    for tu in turns:
        x = turn_x(tu["s"], tu["j"])
        for k, (val, mx, color) in enumerate(
                [(tu["thp"], max_thp, TEAM_HP_COLOR),
                 (tu["ehp"], max_ehp, ENEMY_HP_COLOR)]):
            bx = x - bw - 0.015 if k == 0 else x + 0.015
            h = max(val / mx * HP_H, 0.004)
            ax.add_patch(Rectangle((bx, hp_y), bw, HP_H, facecolor="#f2f2f2",
                                   edgecolor="none"))
            ax.add_patch(Rectangle((bx, hp_y), bw, h, facecolor=color,
                                   edgecolor="none", alpha=0.9))
            if tu["t"] == last_turn_of_stage[tu["s"]]:
                ax.text(bx + bw / 2, hp_y + HP_H + 0.03, f"{val:.0f}",
                        ha="center", va="bottom", fontsize=5, color=color)
    ax.plot([col_x(0), col_x(turns[-1]["s"]) + COL_W], [hp_y, hp_y],
            color="#bbb", linewidth=0.6)
    ax.text(LABEL_X, hp_y + HP_H * 0.72, "team HP", ha="right", va="center",
            fontsize=6.5, color=TEAM_HP_COLOR, fontweight="bold")
    ax.text(LABEL_X, hp_y + HP_H * 0.28, "boss HP", ha="right", va="center",
            fontsize=6.5, color=ENEMY_HP_COLOR, fontweight="bold")
    # value-rank of the played combo, per stage, just under the headers
    for s in range(rec["n_stages"]):
        ax.text(col_x(s) + COL_W / 2, hp_y + HP_H + 0.165,
                f"combo rank {ranks[s]}/27", ha="center", va="bottom",
                fontsize=5.5, color="#666")
    if rec["outcome"] == "WIN":
        x_end = turn_x(turns[-1]["s"], turns[-1]["j"]) + TURN_W * 0.62
        ax.text(x_end, hp_y + HP_H / 2, "WIN", ha="left", va="center",
                fontsize=8, fontweight="bold", color=RELENT_COLOR)


def _draw_role_card(ax, s, ty, role, actions_this_stage, turns_this_stage):
    x0 = col_x(s) + CELL_PAD
    w = COL_W - 2 * CELL_PAD
    ax.add_patch(FancyBboxPatch(
        (x0, ty), w, TRACK_H, boxstyle="round,pad=0,rounding_size=0.05",
        facecolor=ROLE_COLORS[role], edgecolor="white", linewidth=1.0,
        alpha=0.92, zorder=3))
    e = 0.15
    cx, cy = col_x(s) + COL_W / 2, ty + TRACK_H * 0.62
    draw_role_icon(ax, role, cx, cy, size=2 * e, zorder=5)
    ax.text(x0 + w - 0.04, ty + TRACK_H - 0.04, ROLE_SHORT[role],
            ha="right", va="top", fontsize=5.5, color="white",
            fontweight="bold", zorder=6)
    for j, tu in enumerate(turns_this_stage):
        a = actions_this_stage[j]
        ax.text(turn_x(s, j), ty + 0.095, a, ha="center", va="center",
                fontsize=6, color="white", fontweight="bold", zorder=6)


def _draw_role_track(ax, rec, turns, ty, pid, br_by_stage, label_player_color):
    ax.text(LABEL_X, ty + TRACK_H / 2 + 0.06, f"P{pid + 1}", ha="right",
            va="center", fontsize=7.5, color=label_player_color,
            fontweight="bold")
    ax.text(LABEL_X, ty + TRACK_H / 2 - 0.10, "human", ha="right",
            va="center", fontsize=5.5, color="#888")
    stage_turns = rec["stage_turns"]
    for s in range(rec["n_stages"]):
        role = rec["role_seq"][s][pid]
        acts = [t["actions"].get(pid, "?") for t in turns if t["s"] == s]
        _draw_role_card(ax, s, ty, role, acts, stage_turns[s])
        # best-response flag: only when the model would switch this player
        br = br_by_stage.get((s, pid))
        if br is not None and br != role:
            stayed = (s >= 1 and rec["role_seq"][s - 1][pid] == role)
            fx = col_x(s) + COL_W / 2
            fy = ty + TRACK_H + 0.045
            ax.scatter([fx], [fy], marker="v", s=11, color=BR_COLOR, zorder=6)
            ax.text(fx, fy + 0.05, f"best-resp: {ROLE_SHORT[br]}", ha="center",
                    va="bottom", fontsize=5, color=BR_COLOR, fontweight="bold")
            if stayed:
                # emphasize the stuck card + arc to where it relents (next stage)
                x0 = col_x(s) + CELL_PAD
                ax.add_patch(FancyBboxPatch(
                    (x0, ty), COL_W - 2 * CELL_PAD, TRACK_H,
                    boxstyle="round,pad=0,rounding_size=0.05",
                    facecolor="none", edgecolor=BR_COLOR, linewidth=1.4,
                    linestyle=(0, (2, 1.4)), zorder=7))


def _draw_relent_arrow(ax, ty, s_from):
    """Short in-lane arrow from a stuck card into the stage where the
    player relents — kept at the card's mid-height so it unambiguously
    belongs to this player's row."""
    s_to = s_from + 1
    x_from = col_x(s_from) + COL_W - CELL_PAD + 0.01
    x_to = col_x(s_to) + CELL_PAD + 0.18
    y = ty + TRACK_H * 0.58
    ax.annotate("", xy=(x_to, y), xytext=(x_from, y),
                arrowprops=dict(arrowstyle="-|>", color=RELENT_COLOR,
                                linewidth=1.2, connectionstyle="arc3,rad=-0.12",
                                shrinkA=0, shrinkB=0), zorder=8)


def _draw_belief_row(ax, rec, posteriors, ry, target_pid, reports_by_stage,
                     true_prev_role, player_color, show_prior_label):
    """One target player's posterior marginal per stage + teammates' reports."""
    bw, gap = 0.155, 0.05
    max_h = BELIEF_H - 0.14
    ax.text(LABEL_X, ry + BELIEF_H / 2 + 0.05, "belief about", ha="right",
            va="center", fontsize=5.5, color="#888")
    ax.text(LABEL_X, ry + BELIEF_H / 2 - 0.08, f"P{target_pid + 1}", ha="right",
            va="center", fontsize=7, color=player_color, fontweight="bold")
    for s in range(rec["n_stages"]):
        marg = target_marginal(posteriors[s], target_pid)
        base_y = ry + 0.10
        ax.plot([col_x(s) + 0.16, col_x(s) + COL_W - 0.16], [base_y, base_y],
                color="#ccc", linewidth=0.5)
        bar_x = {}
        for role in range(3):
            bx = col_x(s) + COL_W / 2 + (role - 1) * (bw + gap) - bw / 2
            bar_x[role] = bx + bw / 2
            h = max(float(marg[role]) * max_h, 0.012)
            ax.add_patch(Rectangle((bx, base_y), bw, h,
                                   facecolor=ROLE_COLORS[role], edgecolor="none",
                                   alpha=0.80, zorder=4))
        # teammates' reports about this target, made AT stage s (about s-1)
        reports = reports_by_stage.get(s, {})       # {reporter_pid: guessed}
        truth = true_prev_role.get(s)               # target's role at s-1
        for k, (reporter, guessed) in enumerate(sorted(reports.items())):
            mx = bar_x[guessed]
            dx = (-0.5 + k) * 0.13                   # offset the (≤2) reporters
            correct = (truth is not None and guessed == truth)
            ax.scatter([mx + dx], [base_y - 0.055], marker="^", s=12,
                       facecolor="#222" if correct else "white",
                       edgecolor="#222", linewidths=0.6, zorder=6)
            ax.text(mx + dx, base_y - 0.135, f"P{reporter + 1}", ha="center",
                    va="top", fontsize=4.3, color="#444", zorder=6)
        if show_prior_label and s == 0:
            ax.text(col_x(s) + COL_W / 2, ry + BELIEF_H + 0.0,
                    "prior", ha="center", va="bottom", fontsize=5.5,
                    color="#999", style="italic")


def _draw_intent_row(ax, turns, iy):
    ax.text(LABEL_X, iy + INTENT_H / 2, "boss\nattacks", ha="right",
            va="center", fontsize=6, color=ENEMY_HP_COLOR)
    for tu in turns:
        if tu["intent"] == 1:
            ax.scatter([turn_x(tu["s"], tu["j"])], [iy + INTENT_H / 2],
                       marker="v", s=14, color=ENEMY_HP_COLOR, zorder=5)


def _draw_legend(ax, ly, n_stages):
    # Role icons (vector) + names — the colorblind-safe role key.
    lx = LABEL_X
    ax.text(lx, ly + 0.16, "Roles:", ha="left", va="center", fontsize=6.5,
            color="#333", fontweight="bold")
    lx2 = lx + 0.42
    for r in range(3):
        ax.add_patch(Rectangle((lx2, ly + 0.16 - 0.085), 0.17, 0.17,
                               facecolor=ROLE_COLORS[r], edgecolor="none",
                               zorder=3))
        draw_role_icon(ax, r, lx2 + 0.085, ly + 0.16, size=0.165, zorder=5)
        ax.text(lx2 + 0.25, ly + 0.16, ROLE_NAMES[r], ha="left", va="center",
                fontsize=6, color="#333")
        lx2 += 0.25 + 0.10 * len(ROLE_NAMES[r]) + 0.30
    ax.text(lx2 + 0.05, ly + 0.16, "(A / B / H = attack / block / heal)",
            ha="left", va="center", fontsize=5.5, color="#666")

    # marker meanings, second line
    ax.scatter([lx + 0.05], [ly - 0.16], marker="v", s=11, color=BR_COLOR)
    ax.text(lx + 0.14, ly - 0.16,
            "model best-response (dashed = stayed against it: sticky)",
            ha="left", va="center", fontsize=5.5, color="#555")
    rx = lx + 3.45
    ax.annotate("", xy=(rx + 0.16, ly - 0.16), xytext=(rx, ly - 0.16),
                arrowprops=dict(arrowstyle="-|>", color=RELENT_COLOR,
                                linewidth=1.0))
    ax.text(rx + 0.22, ly - 0.16, "relents (rational switch)", ha="left",
            va="center", fontsize=5.5, color="#555")
    # belief markers, third line
    ax.scatter([lx + 0.05], [ly - 0.40], marker="^", s=12, facecolor="#222",
               edgecolor="#222", linewidths=0.6)
    ax.scatter([lx + 0.30], [ly - 0.40], marker="^", s=12, facecolor="white",
               edgecolor="#222", linewidths=0.6)
    ax.text(lx + 0.42, ly - 0.40,
            "a teammate's reported guess (filled = correct, hollow = wrong); "
            "bars = Bayesian observer posterior (stage 1 = prior)",
            ha="left", va="center", fontsize=5.5, color="#555")


# ──────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────

def render(rec, posteriors, br_by_stage, name="R3_team_case"):
    n_stages = rec["n_stages"]
    turns = flatten_turns(rec)
    ranks = [stage_value_rank(rec, s) for s in range(n_stages)]

    # vertical layout, top → bottom (kept compact: minimal inter-row gaps)
    hp_y = -HP_H
    track_ys = []
    yy = hp_y - 0.40
    for _ in range(3):
        yy -= TRACK_H
        track_ys.append(yy)
        yy -= BLOCK_GAP
    belief_ys = []
    yy -= 0.13
    for _ in range(3):
        yy -= BELIEF_H
        belief_ys.append(yy)
        yy -= 0.10
    intent_y = yy - 0.02 - INTENT_H
    legend_y = intent_y - 0.40

    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    _draw_stage_headers(ax, n_stages, turns, y_top=0.0, y_bottom=intent_y)
    _draw_hp_strip(ax, rec, turns, hp_y, ranks)

    player_label_colors = {0: "#222", 1: "#222", 2: "#222"}
    for pid in range(3):
        _draw_role_track(ax, rec, turns, track_ys[pid], pid, br_by_stage,
                         player_label_colors[pid])
    # relent arrows: stuck-against-BR at stage s, switched at s+1
    for (s, pid), br in br_by_stage.items():
        played = rec["role_seq"][s][pid]
        stayed = (s >= 1 and rec["role_seq"][s - 1][pid] == played)
        if br != played and stayed and s + 1 < n_stages \
                and rec["role_seq"][s + 1][pid] != played:
            _draw_relent_arrow(ax, track_ys[pid], s)

    # belief rows: one per TARGET player
    for row, target in enumerate(range(3)):
        # reports about `target`, by stage: {stage: {reporter: guessed}}
        reports_by_stage = {}
        for si, obs_map in rec["inferred"].items():
            for reporter, guesses in obs_map.items():
                if target in guesses:
                    reports_by_stage.setdefault(si, {})[reporter] = guesses[target]
        # target's true role at the PREVIOUS stage (report timing)
        true_prev = {s: rec["role_seq"][s - 1][target]
                     for s in range(1, n_stages)}
        _draw_belief_row(ax, rec, posteriors, belief_ys[row], target,
                         reports_by_stage, true_prev, ROLE_COLORS[0] if False
                         else "#222", show_prior_label=(row == 0))

    _draw_intent_row(ax, turns, intent_y)
    _draw_legend(ax, legend_y, n_stages)

    x_lo, x_hi = LABEL_X - 1.15, n_stages * COL_W + 0.45
    y_lo, y_hi = legend_y - 0.26, 0.55
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.axis("off")
    fig.set_size_inches(FIG_W_IN, FIG_W_IN * (y_hi - y_lo) / (x_hi - x_lo))

    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kw)
        print(f"[team-case] wrote {path}")
    plt.close(fig)


def trim_to_live(rec):
    """Drop trailing stages after the boss/team is already dead.

    The pinned round logs a spurious 4th stage (a duplicate frame with
    the boss already at 0 HP and one player's actions unrecorded). The
    win happens during the stage whose final turn first reaches 0 HP;
    everything after is a dead frame, so we render only the live stages.
    """
    n_live = rec["n_stages"]
    for s in range(rec["n_stages"]):
        turns = rec["stage_turns"][s]
        if turns and (turns[-1]["enemy_hp"] <= 0 or turns[-1]["team_hp"] <= 0):
            n_live = s + 1
            break
    if n_live == rec["n_stages"]:
        return rec
    rec = dict(rec)
    rec["n_stages"] = n_live
    rec["role_seq"] = rec["role_seq"][:n_live]
    rec["stage_turns"] = rec["stage_turns"][:n_live]
    rec["roles"] = {pid: rs[:n_live] for pid, rs in rec["roles"].items()}
    n_turns = sum(len(t) for t in rec["stage_turns"])
    rec["turn_intent"] = rec["turn_intent"][:n_turns]
    rec["inferred"] = {si: m for si, m in rec["inferred"].items() if si < n_live}
    return rec


def main():
    s1, strat = load_stage1_canonical()
    records = load_human_records()
    matches = [r for r in records if r["game_id"] == CASE_GAME_ID
               and r["round_number"] == CASE_ROUND]
    assert len(matches) == 1, f"pinned case not unique ({len(matches)})"
    rec = matches[0]
    assert rec["outcome"] == "WIN", rec["outcome"]
    assert rec["stat_profile_id"] == "222_222_222", rec["stat_profile_id"]
    rec = trim_to_live(rec)

    posteriors = human_posteriors(rec, s1, strat)
    br_by_stage = {}
    for s in range(1, rec["n_stages"]):
        for pid in range(3):
            br, _ev = best_response(rec, posteriors, s, pid)
            br_by_stage[(s, pid)] = br

    traj = " → ".join("".join(ROLE_SHORT[r] for r in rec["role_seq"][s])
                      for s in range(rec["n_stages"]))
    print(f"[team-case] {rec['game_id']} r{rec['round_number']} "
          f"({rec['stat_profile_id']}): {traj}, {rec['outcome']}")
    render(rec, posteriors, br_by_stage)


if __name__ == "__main__":
    main()
