"""R3_team_case (2026-09-02 revision) — the human-team case study WITH the
best-fitting model's per-player role-choice predictions.

Successor to 2026-06-17-pnas-figure-revisions/team_case.py. Advisor
feedback (Tan Zhi-Xuan, 2026-08-31): the 06-17 figure shows humans adapting
but nothing in it shows that the *computational model* explains that
adaptation — the only model content is the observer posterior, and it is
not labelled as coming from the model. Changes:

  1. NEW "model prediction" sub-row per player. Between each player's role
     card and the belief sub-row, one mini bar-chart per stage gives the
     best-fitting model's (Bayesian-Walk-BR, agg_ll fit from the 05-25
     pipeline / 05-28 results.json) predicted distribution over that
     player's role at that stage, P(r_i | history). The bar of the role the
     player ACTUALLY chose is outlined and its probability printed, so the
     reader can see at a glance whether the model "called" each choice.
     The prediction under stage column s is made from the start-of-stage-s
     posterior (= the end-of-stage-(s-1) belief drawn in the column to its
     left) and the stage-(s-1) roles (stickiness term).
  2. Belief bars relabelled in the legend as the MODEL's inference of a
     teammate's belief ("fitted observer" was unclear). The green/red
     carets in BOTH model rows now mark whether the MODEL's most likely
     role is the actual one (chosen role for predictions, played role for
     beliefs); the human inference reports are no longer drawn.
  3. Legend moved from a band along the bottom to a column on the LEFT of
     the diagram (paper figure is full-width, so horizontal space is the
     cheap dimension); the in-diagram "Bayesian-Walk prediction" row label
     is gone (the legend carries it).
  4. Case is a parameter (CASES below) so the same script renders the
     current pinned case plus candidate replacements that end SPECIALIZED
     (three distinct roles) — the advisor flagged the all-Fighter ending as
     a poor illustration of "role specialization".

Prediction mechanics (mirrors models._walk_predict in the 05-25 pipeline):
    switch_i = softmax_{tau_v}( E_{r_-i ~ posterior}[ V(r_i, r_-i, s) ] )
    P(r_i)   = (1 - eps_s) * 1[r_i = r_i^prev] + eps_s * switch_i
with the trajectory (posterior, simulated HP, intent) recomputed by the
pipeline's own precompute_trajectories, so the numbers are exactly what
the fitted model was scored on.

Run from analysis/:
    uv run python experiments/2026-09-02-pnas-figure-revisions/team_case_model.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
DIR_0617 = EXP_DIR / "2026-06-17-pnas-figure-revisions"
DIR_0616 = EXP_DIR / "2026-06-16-pnas-figure-revisions"
DIR_0525 = EXP_DIR / "2026-05-25-full-pipeline"
RESULTS_0528 = EXP_DIR / "2026-05-28-paper-figures" / "results.json"

# Reuse the 06-17 renderer pieces + its vector icons (icons.py resolves its
# assets/ relative to its own file) and the 06-16 data scaffolding.
sys.path.insert(0, str(DIR_0617))
sys.path.insert(0, str(DIR_0616))

from common_human import (  # noqa: E402
    human_posteriors, load_human_records, load_stage1_canonical,
    stage_value_rank,
)
from icons import draw_action_icon, draw_role_icon, draw_svg  # noqa: E402
from shared.constants import ROLE_NAMES, ROLE_SHORT  # noqa: E402
from shared.data_loading import load_all_exports  # noqa: E402
from shared.inference import softmax_role_dist  # noqa: E402



def _load_module(name: str, path: Path):
    """Import a script by path under a unique module name. Both the 06-16 and
    06-17 folders have a `team_case.py`, and the 05-25 pipeline module is
    called `pipeline.py` like the 05-12 one that common_human -> common already
    imported as `pipeline`, so plain `import` would pick the wrong file."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tc = _load_module("team_case_0617", DIR_0617 / "team_case.py")   # renderer pieces
p0525 = _load_module("pipeline_0525", DIR_0525 / "pipeline.py")   # trajectories

OUT_DIR = SCRIPT_DIR / "stuff to incorporate"
OUT_DIR.mkdir(exist_ok=True)
OUT_MD = SCRIPT_DIR / "summary.md"

BEST_MODEL = "Bayesian Walk"          # results.json key; paper name Bayesian-Walk-BR
MODEL_LABEL = "Bayesian-Walk"

# (game-id suffix, round) -> output stem. The first entry is the PRIMARY
# candidate and is ALSO written as plain R3_team_case.{pdf,png}.
CASES = [
    ("2F8H1E", 2, "R3_team_case_2F8H1E_r2"),   # 114_222_222, FTT -> MFT x3, WIN
    ("11B0J4", 8, "R3_team_case_11B0J4_r8"),   # 141_222_222, TFF -> TMF x3, WIN
    ("K72DV0", 5, "R3_team_case_K72DV0_r5"),   # 141_222_222, 5 stages, TFF -> TMF x4
    ("RBRN0Z", 8, "R3_team_case_RBRN0Z_r8"),   # 411_141_114, FMM -> FTM x3
    ("11B0J4", 6, "R3_team_case_11B0J4_r6"),   # the 06-17 pinned case (all-F end)
]
PRIMARY_NAME = "R3_team_case"

# geometry: reuse 06-17 values; add the model sub-row
COL_W, TURN_W, START_W = tc.COL_W, tc.TURN_W, tc.START_W
TRACK_H, BELIEF_H = tc.TRACK_H, tc.BELIEF_H
MODEL_H = 0.38
SUB_GAP, TOP_GAP, HP_H = tc.SUB_GAP, tc.TOP_GAP, tc.HP_H
GROUP_GAP = 0.13          # wider than 06-17 so the belief labels clear the dashed separators
ROLE_COLORS = tc.ROLE_COLORS
TEAM_HP_COLOR, ENEMY_HP_COLOR = tc.TEAM_HP_COLOR, tc.ENEMY_HP_COLOR
CARET_OK_COLOR, CARET_BAD_COLOR = tc.CARET_OK_COLOR, tc.CARET_BAD_COLOR
BOSS_SVG = tc.BOSS_SVG
CHOSEN_EDGE = "#222222"
FIG_W_IN = 7.0
col_x, turn_x, start_cx = tc.col_x, tc.turn_x, tc.start_cx
conditional_role_belief = tc.conditional_role_belief

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
})


# ──────────────────────────────────────────────────────────────────────
# Best-fitting model predictions (per player, per stage)
# ──────────────────────────────────────────────────────────────────────

def load_best_model_params():
    with open(RESULTS_0528) as f:
        res = json.load(f)
    params = res["aggregate"]["params_by_model"][BEST_MODEL]
    s1 = res["scope"]["stage1"]
    return params, s1, res


def build_trajectories(s1):
    """{(game_id, round_number): (trajectory, values)} via the 05-25 pipeline,
    so the model is evaluated on exactly the trajectories it was fit on."""
    recs = p0525.load_human_team_records(verbose=False)
    strat = p0525.strategy_from_params(s1["memory_strategy"], None, None)
    trajs = p0525.precompute_trajectories(recs, s1["tau_prior"], s1["epsilon"],
                                          strat)
    return {(r["game_id"], r["round_number"]): (t, r["env_config"]["values"])
            for r, t in zip(recs, trajs)}


def walk_predictions(traj, values, params):
    """Per stage: {'pred': [3 x (3,)], 'switch': [3 x (3,)], 'prior', 'combo'}.
    pred = Bayesian-Walk-BR per-player role distribution; switch = the
    value-softmax component alone (no stickiness), kept for the summary."""
    tau, eps = params["tau_softmax"], params["epsilon_switch"]
    out = []
    for stage in traj:
        switch = [np.asarray(softmax_role_dist(i, stage["intent"], stage["thp"],
                                               stage["ehp"], stage["prior"],
                                               values, tau), dtype=float)
                  for i in range(3)]
        pred = []
        for i in range(3):
            if stage["prev_roles"] is None:
                pred.append(switch[i])
            else:
                stick = np.zeros(3)
                stick[stage["prev_roles"][i]] = 1.0
                pred.append((1.0 - eps) * stick + eps * switch[i])
        out.append({"pred": pred, "switch": switch, "prior": stage["prior"],
                    "combo": stage["human_combo"], "thp": stage["thp"],
                    "ehp": stage["ehp"], "intent": stage["intent"]})
    return out


# ──────────────────────────────────────────────────────────────────────
# Renderer: player group with the extra model sub-row
# ──────────────────────────────────────────────────────────────────────

def _fmt_p(p):
    s = f"{p:.2f}"
    return s[1:] if s.startswith("0") else s


def _caret(ax, x, y, ok):
    ax.scatter([x], [y], marker="^", s=12,
               facecolor=CARET_OK_COLOR if ok else CARET_BAD_COLOR,
               edgecolor="none", zorder=6)


def _draw_player_group(ax, rec, turns, posteriors, preds, role_y, mod_y,
                       bel_y, pid):
    """P{pid}'s role track; beneath it the best-fitting model's predicted
    role distribution for P{pid} at each stage; beneath that the model's
    inference of each teammate's belief about P{pid}. In both model rows a
    caret under the model's MOST LIKELY role is green if that role is the
    actual one (the chosen role for predictions, the played role for
    beliefs) and red otherwise."""
    n_stages = rec["n_stages"]
    stage_turns = rec["stage_turns"]

    # ---- Start column: player id + compact STR/DEF/SUP stat panel (06-17) ----
    ax.text(0.07, role_y + TRACK_H / 2, f"P{pid + 1}", ha="center",
            va="center", fontsize=7, color="#222", fontweight="bold")
    st = [int(v) for v in rec["player_stats"][pid]]
    lab_x, bar_x0, bar_x1, stat_dy = 0.18, 0.43, 0.73, 0.105
    for k, name in enumerate(("STR", "DEF", "SUP")):
        row_cy = role_y + TRACK_H / 2 + (1 - k) * stat_dy
        ax.text(lab_x, row_cy, name, ha="left", va="center", fontsize=4.2,
                color=ROLE_COLORS[k], fontweight="bold")
        ax.add_patch(Rectangle((bar_x0, row_cy - 0.022), bar_x1 - bar_x0, 0.044,
                               facecolor="#e6e6e6", edgecolor="none", zorder=3))
        ax.add_patch(Rectangle((bar_x0, row_cy - 0.022),
                               (bar_x1 - bar_x0) * st[k] / 6.0, 0.044,
                               facecolor=ROLE_COLORS[k], edgecolor="none",
                               alpha=0.9, zorder=4))
        ax.text(bar_x1 + 0.04, row_cy, f"{st[k]}", ha="left", va="center",
                fontsize=4.4, color="#444", fontweight="bold")

    # ---- role track (06-17 card renderer) ----
    for s in range(n_stages):
        role = rec["role_seq"][s][pid]
        acts = [t["actions"].get(pid, "?") for t in turns if t["s"] == s]
        tc._draw_role_card(ax, s, role_y, role, acts, stage_turns[s])

    # ---- MODEL sub-row: Bayesian-Walk predicted role distribution ----
    pbw, pgap, pmax_h = 0.075, 0.025, 0.15
    pbase_y = mod_y + 0.14
    for s in range(n_stages):
        hc = col_x(s) + COL_W / 2
        pred = preds[s]["pred"][pid]
        chosen = rec["role_seq"][s][pid]
        ax.plot([hc - 0.16, hc + 0.16], [pbase_y, pbase_y], color="#ccc",
                linewidth=0.5)
        bar_cx = {}
        for role in range(3):
            bx = hc + (role - 1) * (pbw + pgap) - pbw / 2
            bar_cx[role] = bx + pbw / 2
            h = max(float(pred[role]) * pmax_h, 0.010)
            is_chosen = role == chosen
            ax.add_patch(Rectangle((bx, pbase_y), pbw, h,
                                   facecolor=ROLE_COLORS[role],
                                   edgecolor=CHOSEN_EDGE if is_chosen else "none",
                                   linewidth=0.8 if is_chosen else 0,
                                   alpha=0.85 if is_chosen else 0.55,
                                   zorder=5 if is_chosen else 4))
            if is_chosen:
                ax.text(bx + pbw / 2, pbase_y + h + 0.012, _fmt_p(pred[role]),
                        ha="center", va="bottom", fontsize=4.4,
                        color="#222", fontweight="bold", zorder=6)
        top = int(np.argmax(pred))
        _caret(ax, bar_cx[top], pbase_y - 0.05, ok=(top == chosen))
        ax.text(hc, pbase_y - 0.10, f"Pr(P{pid + 1} | model)", ha="center",
                va="top", fontsize=4.2, color="#444", zorder=6)

    # ---- BELIEF sub-row: the model's inference of each teammate's belief
    # about P{pid}, P(r_pid | r_obs = obs's role) read off the joint
    # posterior (06-17 semantics and timing). Caret = whether the model's
    # most likely role for P{pid} is the role P{pid} actually played that
    # stage (the human reports are no longer drawn). ----
    observers = [o for o in range(3) if o != pid]
    mbw, mgap, max_h, half_dx = 0.06, 0.02, 0.13, 0.24
    base_y = bel_y + 0.115

    def draw_cell(hc, belief, obs, truth=None):
        ax.plot([hc - 0.13, hc + 0.13], [base_y, base_y],
                color="#ccc", linewidth=0.5)
        bar_cx = {}
        for role in range(3):
            bx = hc + (role - 1) * (mbw + mgap) - mbw / 2
            bar_cx[role] = bx + mbw / 2
            h = max(float(belief[role]) * max_h, 0.010)
            ax.add_patch(Rectangle((bx, base_y), mbw, h,
                                   facecolor=ROLE_COLORS[role],
                                   edgecolor="none", alpha=0.80, zorder=4))
        if truth is not None:
            top = int(np.argmax(belief))
            _caret(ax, bar_cx[top], base_y - 0.05, ok=(top == truth))
        ax.text(hc, base_y - 0.10, f"Pr(P{pid + 1} | P{obs + 1})",
                ha="center", va="top", fontsize=4.2, color="#444", zorder=6)

    # Start column: the prior, before any action — no caret (nothing to be
    # right or wrong about yet)
    for oi, obs in enumerate(observers):
        hc = start_cx() + (oi - 0.5) * 2 * half_dx
        obs_role = rec["role_seq"][0][obs]
        draw_cell(hc, conditional_role_belief(posteriors[0], pid, obs, obs_role),
                  obs)
    for s in range(n_stages):
        truth = rec["role_seq"][s][pid]
        for oi, obs in enumerate(observers):
            hc = col_x(s) + COL_W / 2 + (oi - 0.5) * 2 * half_dx
            obs_role = rec["role_seq"][s][obs]
            belief = conditional_role_belief(posteriors[s + 1], pid, obs, obs_role)
            draw_cell(hc, belief, obs, truth=truth)


# ──────────────────────────────────────────────────────────────────────
# Legend: a vertical column to the LEFT of the diagram (uses the paper's
# full-width horizontal space instead of a band along the bottom)
# ──────────────────────────────────────────────────────────────────────

LEG_W = 1.95            # data-unit width reserved for the legend column
LEG_GAP = 0.20          # gap between the legend column and the Start column
LEG_FS = 5.6            # legend font size


def _draw_legend_left(ax, x_left, y_center):
    """Draw the legend as one left-aligned column whose left edge is x_left,
    vertically centred on y_center. Returns (y_top, y_bottom)."""
    ICON_DX = 0.13              # icon-to-text offset for a 0.10-wide glyph
    LINE = 0.125                # baseline-to-baseline for wrapped text lines
    SEC_GAP = 0.13              # extra gap between legend sections

    entries = []                # (height, draw(y_top))

    def text_lines(x, y_top, lines, color="#333"):
        for k, ln in enumerate(lines):
            ax.text(x, y_top - k * LINE, ln, ha="left", va="top",
                    fontsize=LEG_FS, color=color)

    def inline_row(items, gap=0.16):
        """Several (icon_draw, label) pairs on one line."""
        def draw(y_top):
            cy = y_top - 0.06
            x = x_left
            for icon, label in items:
                icon(x + 0.05, cy)
                ax.text(x + ICON_DX, cy, label, ha="left", va="center",
                        fontsize=LEG_FS, color="#333")
                x += ICON_DX + 0.05 * len(label) + gap
        return 0.14, draw

    def role_icon(r):
        def icon(cx, cy):
            ax.add_patch(Rectangle((cx - 0.05, cy - 0.046), 0.10, 0.092,
                                   facecolor=ROLE_COLORS[r],
                                   edgecolor=ROLE_COLORS[r],
                                   linewidth=0.7, alpha=0.20, zorder=3))
            draw_role_icon(ax, r, cx, cy, size=0.098, zorder=5)
        return icon

    def action_icon(a):
        return lambda cx, cy: draw_action_icon(ax, a, cx, cy, size=0.105,
                                               zorder=5)

    def hp_icon(cx, cy):
        hbw, hh = 0.045, 0.12
        ax.add_patch(Rectangle((cx - 0.05, cy - hh / 2), hbw, hh,
                               facecolor=TEAM_HP_COLOR, edgecolor="none",
                               alpha=0.9, zorder=5))
        ax.add_patch(Rectangle((cx + 0.005, cy - hh / 2), hbw, hh * 0.6,
                               facecolor=ENEMY_HP_COLOR, edgecolor="none",
                               alpha=0.9, zorder=5))

    def boss_icon(cx, cy):
        draw_svg(ax, BOSS_SVG, cx, cy, size=0.13, zorder=5)

    def bars_entry(lines, demo, outline_idx=None, alpha=0.80, caret=None):
        bw, gap, h_max = 0.040, 0.020, 0.12
        bars_w = 3 * bw + 2 * gap
        n = len(lines)
        height = max(0.12 + (n - 1) * LINE + 0.02, 0.22)

        def draw(y_top):
            base = y_top - 0.14
            ax.plot([x_left - 0.008, x_left + bars_w + 0.008], [base, base],
                    color="#ccc", linewidth=0.5, zorder=4)
            for r in range(3):
                is_o = r == outline_idx
                ax.add_patch(Rectangle((x_left + r * (bw + gap), base), bw,
                                       max(demo[r] * h_max, 0.010),
                                       facecolor=ROLE_COLORS[r],
                                       edgecolor=CHOSEN_EDGE if is_o else "none",
                                       linewidth=0.8 if is_o else 0,
                                       alpha=0.85 if is_o else alpha, zorder=5))
            if caret is not None:
                r, ok = caret
                _caret(ax, x_left + r * (bw + gap) + bw / 2, base - 0.045, ok)
            text_lines(x_left + bars_w + 0.08, y_top - 0.01, lines)
        return height, draw

    def caret_entry(ok, lines):
        def draw(y_top):
            _caret(ax, x_left + 0.05, y_top - 0.05, ok)
            text_lines(x_left + ICON_DX, y_top - 0.01, lines)
        return 0.04 + LINE * len(lines), draw

    def spacer(h):
        return h, (lambda y_top: None)

    entries += [
        inline_row([(role_icon(0), "fighter"), (role_icon(1), "tank"),
                    (role_icon(2), "medic")]),
        spacer(0.03),
        inline_row([(action_icon("A"), "attack"), (action_icon("B"), "block"),
                    (action_icon("H"), "heal")]),
        spacer(0.03),
        inline_row([(hp_icon, "team / boss HP"), (boss_icon, "boss attacks")],
                   gap=0.20),
        spacer(SEC_GAP),
        bars_entry(["Pr(Px | Py): the model's inference",
                    "of Py's belief about Px's role"],
                   [0.55, 0.30, 0.15], caret=(0, True)),
        spacer(SEC_GAP),
        bars_entry([f"Pr(Px | model): {MODEL_LABEL}'s",
                    "predicted role for Px; outlined",
                    "bar = the role actually chosen"],
                   [0.20, 0.62, 0.18], outline_idx=1, alpha=0.55,
                   caret=(1, True)),
        spacer(SEC_GAP),
        caret_entry(True, ["model's most likely role is the",
                           "actual one (played / chosen)"]),
        spacer(0.04),
        caret_entry(False, ["model's most likely role differs",
                            "from the actual one"]),
    ]

    total_h = sum(h for h, _ in entries)
    y = y_center + total_h / 2
    y_top = y
    for h, draw in entries:
        draw(y)
        y -= h
    return y_top, y


# ──────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────

def render(rec, posteriors, preds, names):
    n_stages = rec["n_stages"]
    turns = tc.flatten_turns(rec)
    ranks = [stage_value_rank(rec, s) for s in range(n_stages)]

    hp_y = -HP_H - 0.04
    role_ys, mod_ys, bel_ys = [], [], []
    yy = hp_y - 0.22
    for _ in range(3):
        yy -= TOP_GAP
        role_y = yy - TRACK_H
        mod_y = role_y - SUB_GAP - MODEL_H
        bel_y = mod_y - SUB_GAP - BELIEF_H
        role_ys.append(role_y)
        mod_ys.append(mod_y)
        bel_ys.append(bel_y)
        yy = bel_y - GROUP_GAP

    # legend column on the left (x < 0), then Start column + stages
    leg_x_left = -(LEG_GAP + LEG_W)
    x_lo = leg_x_left - 0.05
    x_hi = START_W + n_stages * COL_W + 0.30

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    tc._draw_stage_headers(ax, n_stages, turns, y_top=0.0,
                           y_bottom=bel_ys[-1] + 0.02)
    tc._draw_hp_strip(ax, rec, turns, hp_y, ranks)
    for pid in range(3):
        _draw_player_group(ax, rec, turns, posteriors, preds, role_ys[pid],
                           mod_ys[pid], bel_ys[pid], pid)

    sep_x0, sep_x1 = 0.0, col_x(n_stages - 1) + COL_W
    for i in range(2):
        y_sep = (bel_ys[i] + role_ys[i + 1] + TRACK_H) / 2
        ax.plot([sep_x0, sep_x1], [y_sep, y_sep], color="#bbb", linewidth=0.7,
                linestyle=(0, (4, 3)), zorder=1)

    diagram_top, diagram_bottom = 0.30, bel_ys[-1] - 0.02
    _draw_legend_left(ax, leg_x_left, (diagram_top + diagram_bottom) / 2)

    y_lo, y_hi = diagram_bottom - 0.06, diagram_top
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.axis("off")
    fig.set_size_inches(FIG_W_IN, FIG_W_IN * (y_hi - y_lo) / (x_hi - x_lo))
    for name in names:
        for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
            path = OUT_DIR / f"{name}.{ext}"
            fig.savefig(path, bbox_inches="tight", facecolor="white", **kw)
            print(f"[team-case-model] wrote {path}")
    plt.close(fig)


_PRS_CACHE = []


def _all_player_rounds():
    if not _PRS_CACHE:
        _PRS_CACHE.extend(load_all_exports())
    return _PRS_CACHE


def participant_dominant_models(rec, res):
    """[(pid, participant_id, dominant model, P(Bayesian Walk))] per position."""
    post = res["individual"]["posteriors"]
    out = []
    for pr in _all_player_rounds():
        if (pr.game_id == rec["game_id"]
                and pr.round.round_number == rec["round_number"]
                and pr.round.round_type == "human"):
            p = post.get(pr.participant_id, {})
            dom = max(p.items(), key=lambda kv: kv[1])[0] if p else "n/a"
            out.append((pr.player_id, pr.participant_id, dom,
                        p.get(BEST_MODEL, float("nan"))))
    return sorted(out)


def model_inference_accuracy(rec, posteriors):
    """Fraction of (stage, observer, target) cells where the model's most
    likely role for the target (conditioned on the observer's own role)
    equals the role the target played that stage."""
    hits = tot = 0
    for s in range(rec["n_stages"]):
        for obs in range(3):
            for tgt in range(3):
                if tgt == obs:
                    continue
                b = conditional_role_belief(posteriors[s + 1], tgt, obs,
                                            rec["role_seq"][s][obs])
                hits += int(np.argmax(b) == rec["role_seq"][s][tgt])
                tot += 1
    return hits, tot


def case_summary_md(rec, preds, ranks, dom, inf_acc):
    n = rec["n_stages"]
    lines = [
        f"### `{rec['game_id']}` round {rec['round_number']} — "
        f"`{rec['stat_profile_id']}`, {rec['outcome']}, {n} live stages",
        "",
        "| Stage | combo | value rank | P(chosen) P1 | P2 | P3 | mean | "
        "model top-role = chosen? | value-softmax only P1 | P2 | P3 |",
        "|--:|---|--:|--:|--:|--:|--:|---|--:|--:|--:|",
    ]
    all_p = []
    top_hits = 0
    for s in range(n):
        combo = "".join(ROLE_SHORT[r] for r in rec["role_seq"][s])
        pc = [float(preds[s]["pred"][i][rec["role_seq"][s][i]]) for i in range(3)]
        pv = [float(preds[s]["switch"][i][rec["role_seq"][s][i]]) for i in range(3)]
        hit = ["Y" if int(np.argmax(preds[s]["pred"][i])) == rec["role_seq"][s][i]
               else "n" for i in range(3)]
        top_hits += hit.count("Y")
        all_p.extend(pc)
        lines.append(f"| {s + 1} | {combo} | {ranks[s]}/27 | "
                     + " | ".join(f"{p:.2f}" for p in pc)
                     + f" | **{np.mean(pc):.2f}** | {' '.join(hit)} | "
                     + " | ".join(f"{p:.2f}" for p in pv) + " |")
    lines += [
        "",
        f"Mean P(chosen) over all player-stages: **{np.mean(all_p):.2f}** "
        f"(chance 0.33); min {np.min(all_p):.2f}. Model's most likely role = "
        f"chosen role in **{top_hits}/{3 * n}** player-stages (green carets, "
        "prediction row). Model's most likely inferred role = played role in "
        f"**{inf_acc[0]}/{inf_acc[1]}** observer-target-stage cells (green "
        "carets, belief row).",
        "",
        "| Position | participant | dominant model (R4 posterior) | "
        "P(Bayesian-Walk-BR) |",
        "|---|---|---|--:|",
    ]
    for pid, part, d, pw in dom:
        lines.append(f"| P{pid + 1} | `{part}` | {d} | {pw:.2f} |")
    lines.append("")
    return "\n".join(lines)


def main():
    params, s1_res, res = load_best_model_params()
    s1, strat = load_stage1_canonical()
    for k in ("tau_prior", "epsilon", "memory_strategy"):
        assert s1[k] == s1_res[k], (k, s1[k], s1_res[k])
    print(f"[team-case-model] {BEST_MODEL}: {params}")

    records = load_human_records()
    traj_by_key = build_trajectories(s1)

    md = [
        "# R3_team_case with best-fitting model predictions (2026-09-02)",
        "",
        f"Model: **{BEST_MODEL}** (paper: {MODEL_LABEL}-BR), agg_ll fit from "
        f"`2026-05-28-paper-figures/results.json`: "
        f"tau_softmax = {params['tau_softmax']:.4f}, "
        f"epsilon_switch = {params['epsilon_switch']:.4f}. Stage-1: "
        f"tau_prior = {s1['tau_prior']:.4f}, epsilon = {s1['epsilon']:.4f}, "
        f"memory `{s1['memory_strategy']}`.",
        "",
        "P(chosen) = the model's predicted probability of the role the player "
        "actually chose at that stage (from the start-of-stage posterior + the "
        "previous stage's roles). 'value-softmax only' drops the stickiness "
        "term (what Bayesian-BR alone would say). In a round's final stage the "
        "value matrix is often flat (every combo wins), so the softmax is "
        "uniform and P(chosen) collapses to the stickiness floor "
        f"(1 - eps + eps/3 = {1 - params['epsilon_switch'] + params['epsilon_switch'] / 3:.2f} "
        "for a repeated role).",
        "",
        f"Primary output `{PRIMARY_NAME}.pdf` = `{CASES[0][2]}`.",
        "",
    ]

    for suffix, rnd, name in CASES:
        matches = [r for r in records if r["game_id"].endswith(suffix)
                   and r["round_number"] == rnd]
        assert len(matches) == 1, f"case {suffix} r{rnd}: {len(matches)} matches"
        rec = tc.trim_to_live(matches[0])
        key = (rec["game_id"], rec["round_number"])
        traj, values = traj_by_key[key]
        assert len(traj) >= rec["n_stages"], (len(traj), rec["n_stages"])
        preds = walk_predictions(traj[:rec["n_stages"]], values, params)
        posteriors = human_posteriors(rec, s1, strat)

        # cross-checks: the pipeline's combos + start-of-stage posteriors must
        # match the 06-16 scaffolding's (same Stage-1 params, same sim)
        for s in range(rec["n_stages"]):
            combo = "".join(ROLE_SHORT[r] for r in rec["role_seq"][s])
            assert preds[s]["combo"] == combo, (s, preds[s]["combo"], combo)
            d = float(np.abs(preds[s]["prior"] - posteriors[s]).max())
            assert d < 1e-9, f"posterior mismatch at stage {s}: {d}"

        ranks = [stage_value_rank(rec, s) for s in range(rec["n_stages"])]
        dom = participant_dominant_models(rec, res)
        inf_acc = model_inference_accuracy(rec, posteriors)
        traj_str = " -> ".join("".join(ROLE_SHORT[r] for r in rec["role_seq"][s])
                               for s in range(rec["n_stages"]))
        print(f"[team-case-model] {rec['game_id']} r{rec['round_number']} "
              f"({rec['stat_profile_id']}): {traj_str}, {rec['outcome']}, "
              f"{rec['n_stages']} live stages; ranks {ranks}; "
              f"model inference acc {inf_acc[0]}/{inf_acc[1]}")
        names = [name] + ([PRIMARY_NAME] if name == CASES[0][2] else [])
        render(rec, posteriors, preds, names)
        md.append(case_summary_md(rec, preds, ranks, dom, inf_acc))

    OUT_MD.write_text("\n".join(md))
    print(f"[team-case-model] wrote {OUT_MD}")


if __name__ == "__main__":
    main()
