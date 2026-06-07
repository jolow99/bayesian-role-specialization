"""Figure 2 — single bot-round storyboard v3 (successful adaptation).

Replaces the v1/v2 "qualitative_best_respond" storyboards with a
professional layout consistent with the 06-07 epistemic/instrumental
PNAS figures, showing the 2 TURNS within each stage and using the game
UI role emoji 🤺 / 💂 / 👩🏻‍⚕️ (needs Raqm-enabled Pillow — see common_bot).

Pinned case (case_search.py → case_candidates.md "Pinned" rule): the
human starts on their stat-optimal role, watches two stubborn bots for
one stage, then switches to the deviate-optimal role and holds it
through a WIN.

Layout (single equal-aspect axes, data units; top → bottom):
  1. stage headers + faint stage dividers + per-turn ticks
  2. HP strip — per-turn paired mini-bars from LOGGED turn data
  3. human role track — one role card per stage, logged A/B/H letters
     per turn sub-column, stat-opt/dev-opt margin ticks, "switches" note
  4. two bot role tracks — constant cards ("stubborn"), actions
     RECONSTRUCTED via preferred_action(bot_role, intent, hp_before)
  5. belief mini-rows (one per bot) — Bayesian observer posterior
     marginal per stage (stage 1 = utility prior); ▲/outline = the
     human's report (made at stage s about stage s-1)
  6. enemy-intent row — red marker per attack turn
  7. compact legend

Run from analysis/:
    uv run python experiments/2026-06-07-bot-adaptation/storyboard_v3.py
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

from common_bot import (  # noqa: E402
    OUT_DIR, bot_posteriors, emoji_glyph, load_bot_records,
    load_stage1_canonical, target_marginal,
)
from shared.constants import ROLE_NAMES, ROLE_SHORT  # noqa: E402
from shared.inference import preferred_action  # noqa: E402

# Pinned case — see case_candidates.md "Pinned for storyboard_v3".
# (game_id, round_number) alone is ambiguous: each player in a game has
# their own bot rounds, so the participant id is part of the key.
CASE_GAME_ID = "01KRBKSTM48HJWYZ0J4SRBRN0Z"
CASE_ROUND = 4
CASE_PARTICIPANT = "01KRBKZ6Z204TWFGJKMSNVMH02"

ROLE_COLORS = {0: "#e74c3c", 1: "#3498db", 2: "#2ecc71"}
TEAM_HP_COLOR = "#3498db"
ENEMY_HP_COLOR = "#e74c3c"
STAT_COLOR = "#c0392b"
DEV_COLOR = "#27ae60"
GREY = "#999999"
ACTION_LETTER = {0: "A", 1: "B", 2: "H"}
ACTION_NAME_TO_LETTER = {"ATTACK": "A", "BLOCK": "B", "HEAL": "H"}

# ── geometry (data units; axes aspect is equal) ──
COL_W = 1.0          # stage column width
TURN_W = COL_W / 2   # turn sub-column width
CELL_PAD = 0.05      # padding inside a stage column
TRACK_H = 0.60       # role-card height
HP_H = 0.42          # HP strip height
BELIEF_H = 0.46      # one belief mini-row height
INTENT_H = 0.18      # enemy-intent row height
BLOCK_GAP = 0.14     # vertical gap between role tracks
LABEL_X = -0.12      # right edge of the row-label margin

FIG_W_IN = 7.0       # PNAS full width

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
    """Per-turn dicts with logged human action / post-turn HP and the
    start-of-turn HP the players acted on (= previous turn's logged HP)."""
    out = []
    thp_before = float(rec["team_max_hp"])
    t = 0
    for s, turns in enumerate(rec["stage_turns"]):
        for j, turn in enumerate(turns):
            out.append({
                "s": s, "j": j, "t": t,
                "intent": rec["turn_intent"][t],
                "action_h": ACTION_NAME_TO_LETTER[turn["action"]],
                "thp_after": float(turn["teamHealth"]),
                "ehp_after": float(turn["enemyHealth"]),
                "thp_before": thp_before,
            })
            thp_before = float(turn["teamHealth"])
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
    # faint stage dividers
    for s in range(n_stages + 1):
        ax.plot([col_x(s), col_x(s)], [y_bottom, y_top + 0.26],
                color="#ddd", linewidth=0.6, zorder=0)
    # per-turn ticks (global turn numbers) — above the HP value labels
    for tu in turns:
        x = turn_x(tu["s"], tu["j"])
        ax.text(x, y_top + 0.15, f"{tu['t'] + 1}", ha="center", va="bottom",
                fontsize=5, color="#aaa")
    ax.text(LABEL_X, y_top + 0.15, "turn", ha="right", va="bottom",
            fontsize=5, color="#aaa")


def _draw_hp_strip(ax, rec, turns, hp_y):
    """Per-turn paired mini-bars of LOGGED post-turn team/enemy HP."""
    max_thp, max_ehp = rec["team_max_hp"], rec["enemy_max_hp"]
    bw = 0.15
    last_turn_of_stage = {tu["s"]: tu["t"] for tu in turns}
    for tu in turns:
        x = turn_x(tu["s"], tu["j"])
        for k, (val, mx, color) in enumerate(
                [(tu["thp_after"], max_thp, TEAM_HP_COLOR),
                 (tu["ehp_after"], max_ehp, ENEMY_HP_COLOR)]):
            bx = x - bw - 0.015 if k == 0 else x + 0.015
            h = max(val / mx * HP_H, 0.004)
            ax.add_patch(Rectangle((bx, hp_y), bw, HP_H, facecolor="#f2f2f2",
                                   edgecolor="none"))
            ax.add_patch(Rectangle((bx, hp_y), bw, h, facecolor=color,
                                   edgecolor="none", alpha=0.9))
            if tu["t"] == last_turn_of_stage[tu["s"]]:
                ax.text(bx + bw / 2, hp_y + HP_H + 0.035, f"{val:.0f}",
                        ha="center", va="bottom", fontsize=5,
                        color=color)
    ax.plot([col_x(0), col_x(turns[-1]["s"]) + COL_W],
            [hp_y, hp_y], color="#bbb", linewidth=0.6)
    ax.text(LABEL_X, hp_y + HP_H * 0.72, "team HP", ha="right", va="center",
            fontsize=6.5, color=TEAM_HP_COLOR, fontweight="bold")
    ax.text(LABEL_X, hp_y + HP_H * 0.28, "boss HP", ha="right", va="center",
            fontsize=6.5, color=ENEMY_HP_COLOR, fontweight="bold")
    if rec["outcome"] == "WIN":
        x_end = turn_x(turns[-1]["s"], turns[-1]["j"]) + TURN_W * 0.62
        ax.text(x_end, hp_y + HP_H / 2, "WIN", ha="left", va="center",
                fontsize=7, fontweight="bold", color=DEV_COLOR)


def _draw_role_track(ax, rec, turns, ty, roles_per_stage, label_lines,
                     actions_by_turn, emoji_cache, label_color="#222"):
    """One role track: a role card per stage + per-turn action letters."""
    n_stages = len(roles_per_stage)
    for i, line in enumerate(label_lines):
        text, color, weight = line
        ax.text(LABEL_X, ty + TRACK_H / 2 + (len(label_lines) - 1) * 0.07
                - i * 0.14, text, ha="right", va="center", fontsize=6.5,
                color=color, fontweight=weight)
    for s in range(n_stages):
        r = roles_per_stage[s]
        x0 = col_x(s) + CELL_PAD
        w = COL_W - 2 * CELL_PAD
        ax.add_patch(FancyBboxPatch(
            (x0, ty), w, TRACK_H,
            boxstyle="round,pad=0,rounding_size=0.05",
            facecolor=ROLE_COLORS[r], edgecolor="white",
            linewidth=1.0, alpha=0.92))
        e = 0.155
        cx, cy = col_x(s) + COL_W / 2, ty + TRACK_H * 0.60
        ax.imshow(emoji_cache[r], extent=(cx - e, cx + e, cy - e, cy + e),
                  zorder=5, interpolation="bilinear")
        ax.text(x0 + w - 0.04, ty + TRACK_H - 0.04, ROLE_SHORT[r],
                ha="right", va="top", fontsize=5.5, color="white",
                fontweight="bold", zorder=6)
    # per-turn action letters in the turn sub-columns
    for tu in turns:
        a = actions_by_turn[tu["t"]]
        ax.text(turn_x(tu["s"], tu["j"]), ty + 0.10, a, ha="center",
                va="center", fontsize=6, color="white", fontweight="bold",
                zorder=6)


def _draw_switch_annotation(ax, rec, ty, switch_stage):
    x = col_x(switch_stage) + COL_W / 2
    ax.annotate("switches to deviate-optimal",
                xy=(x, ty + TRACK_H + 0.015), xytext=(x, ty + TRACK_H + 0.21),
                ha="center", va="bottom", fontsize=6.5, color=DEV_COLOR,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=DEV_COLOR,
                                linewidth=0.8))


def _draw_belief_rows(ax, rec, posteriors, belief_y, n_stages):
    """One mini-row per bot: 3-bar posterior marginal per stage; the
    human's report (made AT stage s, about stage s-1) marked ▲/outline."""
    bw, gap = 0.16, 0.05
    max_h = BELIEF_H - 0.14
    for row, pos in enumerate(rec["others"]):
        ry = belief_y[row]
        ax.text(LABEL_X, ry + BELIEF_H / 2, f"belief\nabout P{pos + 1}",
                ha="right", va="center", fontsize=6, color="#666")
        for s in range(n_stages):
            marg = target_marginal(posteriors[s], pos)
            guess = rec["inferred"].get(s, {}).get(pos)
            base_y = ry + 0.10
            ax.plot([col_x(s) + 0.16, col_x(s) + COL_W - 0.16],
                    [base_y, base_y], color="#ccc", linewidth=0.5)
            for role in range(3):
                bx = (col_x(s) + COL_W / 2
                      + (role - 1) * (bw + gap) - bw / 2)
                h = max(float(marg[role]) * max_h, 0.012)
                ax.add_patch(Rectangle(
                    (bx, base_y), bw, h, facecolor=ROLE_COLORS[role],
                    edgecolor="black" if role == guess else "none",
                    linewidth=1.0, alpha=0.95 if role == guess else 0.72,
                    zorder=4))
                if role == guess:
                    # scatter marker, not a text glyph — Helvetica has no
                    # "▲" and falls back to a tofu box
                    ax.scatter([bx + bw / 2], [base_y - 0.055], marker="^",
                               s=9, color="#222", zorder=5)
            if row == 0 and s == 0:
                ax.text(col_x(s) + COL_W / 2, ry + BELIEF_H + 0.02,
                        "prior", ha="center", va="bottom", fontsize=5.5,
                        color="#999", style="italic")


def _draw_intent_row(ax, turns, iy):
    ax.text(LABEL_X, iy + INTENT_H / 2, "boss\nattacks", ha="right",
            va="center", fontsize=6, color=ENEMY_HP_COLOR)
    for tu in turns:
        if tu["intent"] == 1:
            ax.scatter([turn_x(tu["s"], tu["j"])], [iy + INTENT_H / 2],
                       marker="v", s=14, color=ENEMY_HP_COLOR, zorder=5)


def _draw_legend(ax, ly):
    lx = LABEL_X - 1.0
    for r in range(3):
        ax.add_patch(Rectangle((lx, ly - 0.07), 0.14, 0.14,
                               facecolor=ROLE_COLORS[r], edgecolor="none"))
        ax.text(lx + 0.18, ly, ROLE_NAMES[r], ha="left", va="center",
                fontsize=6, color="#333")
        lx += 0.18 + 0.105 * len(ROLE_NAMES[r]) + 0.16
    ax.text(lx + 0.10, ly, "A/B/H = attack / block / heal", ha="left",
            va="center", fontsize=6, color="#555")
    ax.scatter([lx + 1.92], [ly], marker="v", s=14, color=ENEMY_HP_COLOR)
    ax.text(lx + 2.00, ly, "= boss attacks", ha="left", va="center",
            fontsize=6, color="#555")
    ax.scatter([LABEL_X - 0.96], [ly - 0.22], marker="^", s=9, color="#222")
    ax.text(LABEL_X - 0.88, ly - 0.22,
            "/outline = the human's report, made at stage s about stage "
            "s−1;  mini-bars = Bayesian observer posterior over the "
            "bot's role (stage 1 = prior from stats)",
            ha="left", va="center", fontsize=6, color="#555")


# ──────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────

def render_storyboard_v3(rec, posteriors, name="adaptation_case"):
    n_stages = len(rec["human_roles"])
    turns = flatten_turns(rec)
    emoji_cache = {r: emoji_glyph(r) for r in range(3)}

    # bot actions are NOT logged — reconstruct from the fixed role, the
    # enemy intent, and the LOGGED start-of-turn team HP
    bot_actions = {
        pos: [ACTION_LETTER[preferred_action(
            rec["bot_role_map"][pos], tu["intent"], tu["thp_before"],
            rec["team_max_hp"])] for tu in turns]
        for pos in rec["others"]
    }
    human_actions = [tu["action_h"] for tu in turns]

    # vertical layout, top → bottom
    y = 0.0
    hp_y = y - HP_H
    human_y = hp_y - 0.40 - TRACK_H
    bot_y = []
    yy = human_y
    for _ in rec["others"]:
        bot_y.append(yy - BLOCK_GAP - TRACK_H)
        yy = bot_y[-1]
    belief_y = []
    yy -= 0.22
    for _ in rec["others"]:
        belief_y.append(yy - BELIEF_H)
        yy = belief_y[-1] - 0.10
    intent_y = yy - 0.10 - INTENT_H
    legend_y = intent_y - 0.42

    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    _draw_stage_headers(ax, n_stages, turns, y_top=0.0, y_bottom=intent_y)
    _draw_hp_strip(ax, rec, turns, hp_y)
    _draw_role_track(
        ax, rec, turns, human_y, rec["human_roles"],
        [(f"P{rec['pid'] + 1} · human", "#222", "bold"),
         (f"stat-opt {ROLE_SHORT[rec['human_stat_opt']]}", STAT_COLOR,
          "normal"),
         (f"dev-opt {ROLE_SHORT[rec['human_dev_opt']]}", DEV_COLOR,
          "normal")],
        human_actions, emoji_cache)
    switch_stage = next(s for s in range(1, n_stages)
                        if rec["human_roles"][s] == rec["human_dev_opt"])
    _draw_switch_annotation(ax, rec, human_y, switch_stage)
    for row, pos in enumerate(rec["others"]):
        _draw_role_track(
            ax, rec, turns, bot_y[row],
            [rec["bot_role_map"][pos]] * n_stages,
            [(f"P{pos + 1} · bot", "#666", "bold")],
            bot_actions[pos], emoji_cache)
    _draw_belief_rows(ax, rec, posteriors, belief_y, n_stages)
    _draw_intent_row(ax, turns, intent_y)
    _draw_legend(ax, legend_y)

    x_lo, x_hi = LABEL_X - 1.05, n_stages * COL_W + 0.42
    y_lo, y_hi = legend_y - 0.35, 0.62
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.axis("off")
    fig.set_size_inches(FIG_W_IN, FIG_W_IN * (y_hi - y_lo) / (x_hi - x_lo))

    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kw)
        print(f"[storyboard-v3] wrote {path}")
    plt.close(fig)


def main():
    s1, strat = load_stage1_canonical()
    records = load_bot_records()
    matches = [r for r in records
               if r["game_id"] == CASE_GAME_ID
               and r["round_number"] == CASE_ROUND
               and r["participant_id"] == CASE_PARTICIPANT]
    assert len(matches) == 1, (
        f"Pinned case not found uniquely ({len(matches)} matches) — "
        "re-run case_search.py and re-pin.")
    rec = matches[0]

    # the storyboard's story must still hold for the pinned round
    assert rec["outcome"] == "WIN", rec["outcome"]
    assert rec["human_roles"][0] == rec["human_stat_opt"], rec["human_roles"]
    assert rec["human_roles"][-1] == rec["human_dev_opt"], rec["human_roles"]

    posteriors = bot_posteriors(rec, s1, strat)
    traj = " → ".join(ROLE_SHORT[r] for r in rec["human_roles"])
    print(f"[storyboard-v3] case {rec['game_id']} r{rec['round_number']} "
          f"({rec['treatment_id']}): {traj}, human at P{rec['pid'] + 1}, "
          f"bots {{P{rec['others'][0] + 1}: "
          f"{ROLE_SHORT[rec['bot_role_map'][rec['others'][0]]]}, "
          f"P{rec['others'][1] + 1}: "
          f"{ROLE_SHORT[rec['bot_role_map'][rec['others'][1]]]}}}")
    render_storyboard_v3(rec, posteriors)


if __name__ == "__main__":
    main()
