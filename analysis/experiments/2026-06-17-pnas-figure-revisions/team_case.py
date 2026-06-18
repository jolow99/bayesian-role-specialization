"""R3_team_case (2026-06-17 revision) — compact human-team case study for
the "role choice = rational value-seeking + sticky inertia" claim.

Successor to 2026-06-16-pnas-figure-revisions/team_case.py. Changes asked
for by the advisor:

  1. NEW CASE with >= 4 LIVE stages. The 06-16 case
     (01KRBT30...X48RYB r4) logged 4 stages but the 4th was a spurious
     post-win duplicate, leaving only 3 live stages after trimming. The
     new case is a genuine 4-live-stage round.

  2. MUCH MORE COMPACT (less whitespace):
     * each player's role track is MERGED with the "belief about that
       player" row into a single labelled group (was 3 + 3 = 6 separate
       rows; now 3 stacked groups);
     * the "boss attacks" markers are folded into the team/boss HP strip
       (was its own row);
     * the green relent transition arrows are removed (the role card
       simply changing in the next stage already shows the relent).

Pinned case: game 01KQ6YDF6T28MRGBK1E911B0J4, round 6 — a fully symmetric
team (222_222_222), 4 live stages, WIN:

  | stage | combo | rank | what happens                                  |
  |------:|-------|-----:|-----------------------------------------------|
  |   1   | MTF   | 17   | mis-coordinated start                         |
  |   2   | MMF   | 18   | P3 already best-responding (Fighter); P1 STAYS Medic though best-response says Fighter — stickiness |
  |   3   | MFF   |  9   | P2 relents -> Fighter; P1 STILL stuck on Medic |
  |   4   | FFF   |  1   | P1 finally relents -> Fighter; value-optimal, WIN |

  P3 is the rational anchor (Fighter throughout); P1 is the sticky
  laggard who holds Medic against best-response for two stages then
  relents — the visual signature of a rational + sticky mixture
  (cf. Bayesian Walk-PS in R4_individual_fitting).

Role icons are TRUE VECTOR Twemoji (svg_icons.draw_role_icon). Posteriors
use the 05-25 Stage-1 params (cross-checked vs common.py's 05-12 fit).

Run from analysis/:
    uv run python experiments/2026-06-17-pnas-figure-revisions/team_case.py
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
# Reuse the 06-16 scaffolding (data join, posteriors, best-response) — same
# cross-folder import pattern the 06-16 experiment itself uses. Icons are now
# local (icons.py) so the action/stat glyphs render vector like the roles.
PREV_DIR = SCRIPT_DIR.parent / "2026-06-16-pnas-figure-revisions"
sys.path.insert(0, str(PREV_DIR))

from common_human import (  # noqa: E402
    human_posteriors, load_human_records,
    load_stage1_canonical, stage_value_rank,
)
from icons import draw_action_icon, draw_role_icon, draw_svg  # noqa: E402
from shared.constants import ROLE_NAMES, ROLE_SHORT  # noqa: E402

OUT_DIR = SCRIPT_DIR / "stuff to incorporate"
OUT_DIR.mkdir(exist_ok=True)

# Pinned case — symmetric 222 team, 4 live stages, MTF->MMF->MFF->FFF, WIN.
CASE_GAME_ID = "01KQ6YDF6T28MRGBK1E911B0J4"
CASE_ROUND = 6

ROLE_COLORS = {0: "#e74c3c", 1: "#3498db", 2: "#2ecc71"}   # F / T / M
TEAM_HP_COLOR = "#3498db"
ENEMY_HP_COLOR = "#e74c3c"
WIN_COLOR = "#27ae60"
CARET_OK_COLOR = "#27ae60"     # green caret: observer correctly infers the role
CARET_BAD_COLOR = "#e74c3c"    # red caret: observer wrongly infers the role
BOSS_SVG = "boss_attack.svg"   # 👹 Twemoji, vector, marks the boss-attack turns
ACTION_NAME_TO_LETTER = {"ATTACK": "A", "BLOCK": "B", "HEAL": "H"}

# ── geometry (data units; axes aspect is equal) ──
COL_W = 1.0
TURN_W = COL_W / 2
START_W = 0.95             # leading "Start" column (initial HP + player id/stats)
CELL_PAD = 0.05
TRACK_H = 0.50
BELIEF_H = 0.30
SUB_GAP = 0.03              # role card <-> its belief sub-row
GROUP_GAP = 0.06           # between player groups
TOP_GAP = 0.05             # small headroom above each role card
HP_H = 0.40

FIG_W_IN = 7.0

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
})


def col_x(s):
    """Left edge of stage `s` (stages sit to the right of the Start column)."""
    return START_W + s * COL_W


def turn_x(s, j):
    return START_W + s * COL_W + (j + 0.5) * TURN_W


def start_cx():
    """Centre of the leading Start column."""
    return START_W / 2


def conditional_role_belief(joint, target, obs, obs_val):
    """Observer `obs`'s belief about `target`'s role from the observer model's
    joint posterior: P(role_target | role_obs = obs_val), marginalizing the
    third player. Conditioning on the observer's own (known) role is what makes
    the two teammates' beliefs about a given player differ (the joint is
    correlated via the memory drift-to-prior step)."""
    third = [a for a in range(3) if a not in (target, obs)][0]
    idx = [slice(None)] * 3
    idx[obs] = obs_val
    sub = joint[tuple(idx)]                       # 2-D over the two non-obs axes
    rem = [a for a in range(3) if a != obs]       # axis order within `sub`
    m = sub.sum(axis=rem.index(third))            # marginalize the third player
    t = m.sum()
    return m / t if t > 0 else np.ones(3) / 3.0


def flatten_turns(rec):
    """Per-turn dicts: global turn idx, intent, logged team/enemy HP, and
    each player's logged action letter."""
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
    ax.text(start_cx(), y_top + 0.26, "Start", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color="#777")
    for s in range(n_stages):
        ax.text(col_x(s) + COL_W / 2, y_top + 0.26, f"Stage {s + 1}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                color="#333")
    # interior column separators only (Start|Stage1, then each stage boundary);
    # the outer left/right edges are left open to keep the figure uncluttered
    for x in [col_x(s) for s in range(n_stages)]:
        ax.plot([x, x], [y_bottom, y_top + 0.22], color="#ddd",
                linewidth=0.6, zorder=0)


def _draw_hp_strip(ax, rec, turns, hp_y, ranks):
    """Team/boss HP mini-bars per turn, with the boss-attack markers folded
    into the same band (small red carets just below the HP baseline)."""
    max_thp, max_ehp = rec["team_max_hp"], rec["enemy_max_hp"]
    bw = 0.15
    last_turn_of_stage = {}
    for tu in turns:
        last_turn_of_stage[tu["s"]] = tu["t"]
    attack_y = hp_y - 0.07

    # ---- Start column: the initial (full) team/boss HP, before any turn ----
    for k, (val, mx, color) in enumerate(
            [(max_thp, max_thp, TEAM_HP_COLOR),
             (max_ehp, max_ehp, ENEMY_HP_COLOR)]):
        bx = start_cx() - bw - 0.015 if k == 0 else start_cx() + 0.015
        ax.add_patch(Rectangle((bx, hp_y), bw, HP_H, facecolor="#f2f2f2",
                               edgecolor="none"))
        ax.add_patch(Rectangle((bx, hp_y), bw, val / mx * HP_H, facecolor=color,
                               edgecolor="none", alpha=0.9))
        ax.text(bx + bw / 2, hp_y + HP_H + 0.03, f"{val:.0f}", ha="center",
                va="bottom", fontsize=5, color=color)
    ax.text(start_cx(), hp_y - 0.15, "initial", ha="center", va="top",
            fontsize=5, color="#999")

    for tu in turns:
        x = turn_x(tu["s"], tu["j"])
        # label HP values on the first turn of each stage as well as the last
        # (single-turn stages collapse to one label) so every stage is read-off
        label_turn = (tu["j"] == 0 or tu["t"] == last_turn_of_stage[tu["s"]])
        for k, (val, mx, color) in enumerate(
                [(tu["thp"], max_thp, TEAM_HP_COLOR),
                 (tu["ehp"], max_ehp, ENEMY_HP_COLOR)]):
            bx = x - bw - 0.015 if k == 0 else x + 0.015
            h = max(val / mx * HP_H, 0.004)
            ax.add_patch(Rectangle((bx, hp_y), bw, HP_H, facecolor="#f2f2f2",
                                   edgecolor="none"))
            ax.add_patch(Rectangle((bx, hp_y), bw, h, facecolor=color,
                                   edgecolor="none", alpha=0.9))
            if label_turn:
                ax.text(bx + bw / 2, hp_y + HP_H + 0.03, f"{val:.0f}",
                        ha="center", va="bottom", fontsize=5, color=color)
        # boss-attack marker for this turn, in the HP band's lower margin
        # (👹 emoji, keyed in the legend; no inline row label)
        if tu["intent"] == 1:
            draw_svg(ax, BOSS_SVG, x, attack_y, size=0.13, zorder=5)
        # per-turn label under the HP strip (team/boss HP key now in legend)
        ax.text(x, hp_y - 0.15, f"turn {tu['j'] + 1}", ha="center", va="top",
                fontsize=5, color="#999")
    ax.plot([0.0, col_x(turns[-1]["s"]) + COL_W], [hp_y, hp_y],
            color="#bbb", linewidth=0.6)
    # value-rank of the played combo, per stage, just under the headers
    for s in range(rec["n_stages"]):
        ax.text(col_x(s) + COL_W / 2, hp_y + HP_H + 0.155,
                f"combo rank {ranks[s]}/27", ha="center", va="bottom",
                fontsize=5.5, color="#666")


def _draw_role_card(ax, s, ty, role, actions_this_stage, turns_this_stage):
    x0 = col_x(s) + CELL_PAD
    w = COL_W - 2 * CELL_PAD
    bs = "round,pad=0,rounding_size=0.05"
    # light role-tinted card (transparent fill + crisp role-coloured border) so
    # the role/action icons read on a near-white ground instead of a solid block
    ax.add_patch(FancyBboxPatch((x0, ty), w, TRACK_H, boxstyle=bs,
                                facecolor=ROLE_COLORS[role], edgecolor="none",
                                alpha=0.15, zorder=2))
    ax.add_patch(FancyBboxPatch((x0, ty), w, TRACK_H, boxstyle=bs,
                                facecolor="none", edgecolor=ROLE_COLORS[role],
                                linewidth=1.1, alpha=0.85, zorder=3))
    # role glyph (no text label — the icon + card tint carry the role)
    cx, cy = col_x(s) + COL_W / 2, ty + TRACK_H * 0.67
    draw_role_icon(ax, role, cx, cy, size=0.30, zorder=5)
    # per-turn action emojis directly on the light card (no white chip needed)
    for j in range(len(turns_this_stage)):
        a = actions_this_stage[j]
        if a not in ("A", "B", "H"):
            continue
        ax_, ay = turn_x(s, j), ty + 0.12
        draw_action_icon(ax, a, ax_, ay, size=0.125, zorder=6)


def _draw_player_group(ax, rec, turns, posteriors, role_y, bel_y, pid):
    """One merged group: P{pid}'s role track on top, the team's belief about
    P{pid} directly beneath, sharing one left label."""
    n_stages = rec["n_stages"]
    stage_turns = rec["stage_turns"]

    # Start column (x in [0, START_W]): a player id at the left, then a compact
    # 3-row stat panel mirroring the game UI's PlayerStats (STR/DEF/SUP label +
    # short bar of value/6 + value), colour-linked to the role each favours
    # (STR->F red, DEF->T blue, SUP->M green).
    ax.text(0.07, role_y + TRACK_H / 2, f"P{pid + 1}", ha="center",
            va="center", fontsize=7, color="#222", fontweight="bold")
    st = [int(v) for v in rec["player_stats"][pid]]      # STR, DEF, SUP
    lab_x = 0.18
    bar_x0, bar_x1 = 0.43, 0.73
    stat_dy = 0.105                                      # tight row spacing
    for k, name in enumerate(("STR", "DEF", "SUP")):
        row_cy = role_y + TRACK_H / 2 + (1 - k) * stat_dy
        ax.text(lab_x, row_cy, name, ha="left", va="center", fontsize=4.2,
                color=ROLE_COLORS[k], fontweight="bold")
        ax.add_patch(Rectangle((bar_x0, row_cy - 0.022), bar_x1 - bar_x0, 0.044,
                               facecolor="#e6e6e6", edgecolor="none", zorder=3))
        frac = st[k] / 6.0
        ax.add_patch(Rectangle((bar_x0, row_cy - 0.022),
                               (bar_x1 - bar_x0) * frac, 0.044,
                               facecolor=ROLE_COLORS[k], edgecolor="none",
                               alpha=0.9, zorder=4))
        ax.text(bar_x1 + 0.04, row_cy, f"{st[k]}", ha="left", va="center",
                fontsize=4.4, color="#444", fontweight="bold")

    # ---- role track ----
    for s in range(n_stages):
        role = rec["role_seq"][s][pid]
        acts = [t["actions"].get(pid, "?") for t in turns if t["s"] == s]
        _draw_role_card(ax, s, role_y, role, acts, stage_turns[s])

    # ---- belief sub-row: one mini posterior per TEAMMATE-observer, i.e. each
    # other player's belief about P{pid}'s role conditioned on that observer's
    # own (known) role — P(r_pid | r_obs = obs's role), marginalizing the third.
    # Two observers => two mini bar-charts per stage, each carrying that
    # observer's correct/wrong inference caret (filled = correct, hollow =
    # wrong) and an observer label.
    #
    # Alignment: the chart under stage column `s` is the belief at the END of
    # stage `s` (= posteriors[s + 1], after observing stage s's actions),
    # conditioned on the observer's role *during* stage s, and the caret is the
    # human report ABOUT stage s — which the game logs at the NEXT stage
    # (s + 1), since an inference made at stage N is about stage N-1. The last
    # stage therefore shows its end-of-stage posterior but no caret (no report
    # is ever logged after it). The leading Start column carries the INITIAL
    # belief (the prior, posteriors[0]) before any actions — no caret. ----
    observers = [o for o in range(3) if o != pid]          # the two teammates
    reports_by_stage = {}   # {stage logged: {reporter: guessed role about pid}}
    for si, obs_map in rec["inferred"].items():
        for reporter, guesses in obs_map.items():
            if pid in guesses:
                reports_by_stage.setdefault(si, {})[reporter] = guesses[pid]
    true_now = {s: rec["role_seq"][s][pid] for s in range(n_stages)}

    mbw, mgap, max_h, half_dx = 0.06, 0.02, 0.13, 0.24
    base_y = bel_y + 0.115

    def draw_cell(hc, belief, obs, guessed=None, correct=False):
        """One observer's mini posterior bar-chart at centre `hc`, with an
        optional correct/wrong caret over the guessed-role bar."""
        ax.plot([hc - 0.13, hc + 0.13], [base_y, base_y],
                color="#ccc", linewidth=0.5)
        bar_x = {}
        for role in range(3):
            bx = hc + (role - 1) * (mbw + mgap) - mbw / 2
            bar_x[role] = bx + mbw / 2
            h = max(float(belief[role]) * max_h, 0.010)
            ax.add_patch(Rectangle((bx, base_y), mbw, h,
                                   facecolor=ROLE_COLORS[role],
                                   edgecolor="none", alpha=0.80, zorder=4))
        if guessed is not None:
            ax.scatter([bar_x[guessed]], [base_y - 0.05], marker="^", s=12,
                       facecolor=CARET_OK_COLOR if correct else CARET_BAD_COLOR,
                       edgecolor="none", zorder=6)
        # label as Pr(target | observer): observer's belief over this row's
        # player's role given the observer's own (known) role
        ax.text(hc, base_y - 0.10, f"Pr(P{pid + 1} | P{obs + 1})",
                ha="center", va="top", fontsize=4.2, color="#444", zorder=6)

    # ---- Start column: the initial belief (the prior) before any actions ----
    for oi, obs in enumerate(observers):
        hc = start_cx() + (oi - 0.5) * 2 * half_dx
        obs_role = rec["role_seq"][0][obs]   # observer's chosen stage-1 role
        belief = conditional_role_belief(posteriors[0], pid, obs, obs_role)
        draw_cell(hc, belief, obs)

    # ---- stage columns: end-of-stage-s belief ----
    for s in range(n_stages):
        truth = true_now.get(s)               # P{pid}'s actual role in stage s
        reports = reports_by_stage.get(s + 1, {})   # reports ABOUT stage s
        for oi, obs in enumerate(observers):
            hc = col_x(s) + COL_W / 2 + (oi - 0.5) * 2 * half_dx
            obs_role = rec["role_seq"][s][obs]   # observer's own role in stage s
            belief = conditional_role_belief(posteriors[s + 1], pid, obs, obs_role)
            guessed = reports.get(obs)
            correct = (guessed is not None and truth is not None
                       and guessed == truth)
            draw_cell(hc, belief, obs, guessed=guessed, correct=correct)


def _draw_legend(ax, cx_center, y_top):
    """Compact horizontal legend band along the bottom of the figure, laid out
    across two centered rows: (1) the role icons + action icons (keyed by their
    lowercase names) + the green "correctly infers" caret; (2) the team/boss HP
    bar, the 👹 boss-attack marker, the posterior-bar key, and the red "wrongly
    infers" caret. Returns the centre y of the last row so the caller can size
    the canvas to it."""
    CW = 0.052          # approx data-unit width per character at fontsize 6
    TIGHT = 0.10        # gap WITHIN a cluster (fighter/tank/medic, A/B/H)
    INTER = 0.22        # baseline gap BETWEEN clusters
    ROW_H = 0.185

    def role_item(r):
        name = ROLE_NAMES[r].lower()

        def draw(x, y):
            cx = x + 0.05
            ax.add_patch(Rectangle((cx - 0.05, y - 0.046), 0.10, 0.092,
                                   facecolor=ROLE_COLORS[r],
                                   edgecolor=ROLE_COLORS[r],
                                   linewidth=0.7, alpha=0.20, zorder=3))
            draw_role_icon(ax, r, cx, y, size=0.098, zorder=5)
            ax.text(x + 0.135, y, name, ha="left", va="center", fontsize=6,
                    color="#333")
        return 0.135 + CW * len(name), draw

    def action_item(a, name):
        def draw(x, y):
            draw_action_icon(ax, a, x + 0.05, y, size=0.105, zorder=5)
            ax.text(x + 0.12, y, name, ha="left", va="center", fontsize=6,
                    color="#333")
        return 0.12 + CW * len(name), draw

    def hp_item():
        text = "team / boss HP"

        def draw(x, y):
            hbw, hh = 0.05, 0.13
            ax.add_patch(Rectangle((x, y - hh / 2), hbw, hh,
                                   facecolor=TEAM_HP_COLOR, edgecolor="none",
                                   alpha=0.9, zorder=5))
            ax.add_patch(Rectangle((x + hbw + 0.015, y - hh / 2), hbw, hh * 0.6,
                                   facecolor=ENEMY_HP_COLOR, edgecolor="none",
                                   alpha=0.9, zorder=5))
            ax.text(x + 0.135, y, text, ha="left", va="center", fontsize=6,
                    color="#333")
        return 0.135 + CW * len(text), draw

    def boss_item():
        text = "boss attacks"

        def draw(x, y):
            draw_svg(ax, BOSS_SVG, x + 0.05, y, size=0.13, zorder=5)
            ax.text(x + 0.135, y, text, ha="left", va="center", fontsize=6,
                    color="#333")
        return 0.135 + CW * len(text), draw

    def posterior_item():
        text = "Pr(Px | Py) = Py's belief of Px"
        bw, gap, h_max = 0.040, 0.020, 0.12
        bars_w = 3 * bw + 2 * gap

        def draw(x, y):
            demo = [0.55, 0.30, 0.15]
            base = y - h_max / 2
            ax.plot([x - 0.008, x + bars_w + 0.008], [base, base],
                    color="#ccc", linewidth=0.5, zorder=4)
            for r in range(3):
                ax.add_patch(Rectangle((x + r * (bw + gap), base), bw,
                                       max(demo[r] * h_max, 0.010),
                                       facecolor=ROLE_COLORS[r], edgecolor="none",
                                       alpha=0.80, zorder=5))
            ax.text(x + bars_w + 0.06, y, text, ha="left", va="center",
                    fontsize=6, color="#333")
        return bars_w + 0.06 + CW * len(text), draw

    def caret_item(text, color, width):
        # width is passed in (shared max) so both carets get an identical-width
        # cluster — with the rows justified, equal last-cluster widths make the
        # two carets left-align in the same column.
        def draw(x, y):
            ax.scatter([x + 0.05], [y], marker="^", s=12, facecolor=color,
                       edgecolor="none", zorder=5)
            ax.text(x + 0.13, y, text, ha="left", va="center", fontsize=6,
                    color="#333")
        return width, draw

    # Each row is a list of CLUSTERS; items inside a cluster are packed tight
    # (TIGHT gap) and clusters are spread apart. Both rows are justified to a
    # common width so their left/right edges line up and the two inference
    # carets sit in the same column on the right.
    ok_text, bad_text = "Py correctly infers Px", "Py wrongly infers Px"
    caret_w = 0.13 + CW * max(len(ok_text), len(bad_text))
    rows = [
        [[role_item(0), role_item(1), role_item(2)],
         [action_item("A", "attack"), action_item("B", "block"),
          action_item("H", "heal")],
         [caret_item(ok_text, CARET_OK_COLOR, caret_w)]],
        [[hp_item()], [boss_item()], [posterior_item()],
         [caret_item(bad_text, CARET_BAD_COLOR, caret_w)]],
    ]

    def cluster_w(cluster):
        return sum(w for w, _ in cluster) + TIGHT * (len(cluster) - 1)

    leg_w = max(sum(cluster_w(c) for c in row) + INTER * (len(row) - 1)
                for row in rows)
    x_left = cx_center - leg_w / 2

    y = y_top
    for row in rows:
        cws = [cluster_w(c) for c in row]
        inter = (leg_w - sum(cws)) / (len(row) - 1) if len(row) > 1 else 0.0
        gx = x_left
        for cluster, cw in zip(row, cws):
            ix = gx
            for w, draw in cluster:
                draw(ix, y)
                ix += w + TIGHT
            gx += cw + inter
        y -= ROW_H
    return y + ROW_H     # centre y of the last drawn row


# ──────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────

def render(rec, posteriors, name="R3_team_case"):
    n_stages = rec["n_stages"]
    turns = flatten_turns(rec)
    ranks = [stage_value_rank(rec, s) for s in range(n_stages)]

    # vertical layout, top -> bottom (extra headroom below the HP strip for
    # the per-turn labels)
    hp_y = -HP_H - 0.04
    group_role_ys, group_bel_ys = [], []
    yy = hp_y - 0.22
    for _ in range(3):
        yy -= TOP_GAP
        role_y = yy - TRACK_H
        group_role_ys.append(role_y)
        bel_y = role_y - SUB_GAP - BELIEF_H
        group_bel_ys.append(bel_y)
        yy = bel_y - GROUP_GAP

    # the player-id + (compact) stat panels now live inside the Start column and
    # the legend sits in a band at the bottom, so the grid (Start + stage
    # columns) can start at x ~ 0 with only a thin left margin.
    x_lo = -0.05
    x_hi = START_W + n_stages * COL_W + 0.35

    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    _draw_stage_headers(ax, n_stages, turns, y_top=0.0,
                        y_bottom=group_bel_ys[-1] + 0.02)
    _draw_hp_strip(ax, rec, turns, hp_y, ranks)

    for pid in range(3):
        _draw_player_group(ax, rec, turns, posteriors, group_role_ys[pid],
                           group_bel_ys[pid], pid)

    # dashed separators in the gaps between player groups, so each player's
    # belief sub-row reads with the card ABOVE it (its own) rather than being
    # mis-grouped with the beliefs of the player above.
    sep_x0, sep_x1 = 0.0, col_x(n_stages - 1) + COL_W
    for i in range(2):
        y_sep = (group_bel_ys[i] + group_role_ys[i + 1] + TRACK_H) / 2
        ax.plot([sep_x0, sep_x1], [y_sep, y_sep], color="#bbb", linewidth=0.7,
                linestyle=(0, (4, 3)), zorder=1)

    last_row_y = _draw_legend(ax, (x_lo + x_hi) / 2,
                              group_bel_ys[-1] - 0.30)

    y_lo, y_hi = last_row_y - 0.11, 0.30
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
    """Drop trailing stages logged after the boss/team is already dead."""
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
    assert rec["n_stages"] >= 4, f"need >=4 live stages, got {rec['n_stages']}"

    posteriors = human_posteriors(rec, s1, strat)

    traj = " -> ".join("".join(ROLE_SHORT[r] for r in rec["role_seq"][s])
                       for s in range(rec["n_stages"]))
    print(f"[team-case] {rec['game_id']} r{rec['round_number']} "
          f"({rec['stat_profile_id']}): {traj}, {rec['outcome']}, "
          f"{rec['n_stages']} live stages")
    render(rec, posteriors)


if __name__ == "__main__":
    main()
