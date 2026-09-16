"""R3_team_case (2026-09-16 revision) — the human-team case study WITH the
best-fitting model's per-player role-choice predictions.

Fourth pass, after the advisor's (Tan Zhi-Xuan) written feedback of
2026-09-14 on the 09-02/09-09 figure, plus the user's 09-16 calls (outline
not caret, π not Pr, label once in the Start column, subheaders, legend row,
boss icon centred over its turn, no variants). Standardised choices:

  a. In-card model chart ("Pr(Px | Model)", formerly "Pr(Px | BW)"): the
     green/red correct/wrong caret under the model's top pick is GONE (with
     near-uniform predictions the model cannot really be "wrong"). Instead a
     bold dark OUTLINE marks the bar of the role Px actually chose, and every
     bar carries its probability. The chart is NOT labelled per card: the
     Start column carries "π(Px | Model)" once per row, at the strip's
     height, under a small "Model prediction" subheader; the belief
     sub-row's Start column carries a "Model beliefs & player inferences"
     subheader above the labelled prior charts. Legend is terse (the
     subheaders carry the explanation).
  b. Legend: the model block now sits ABOVE the belief block (matching the
     vertical order in the figure); keys read
       π(Px | Model)  = Prediction of Px's role selection by best-fitting
                        model (Bayesian-Walk)
       Pr(Px | Py)    = Py's belief about Px's role if Py were a Bayesian
                        observer
     plus an outline key ("role Px actually chose"). "team / boss HP" and
     "boss attacks" share one legend row. The belief-chart green/red carets
     (Py's own report, correct/wrong) are unchanged.
  c. Notation: π(Px | Model) for the model's role-selection policy (main text
     uses π for policies, Pr for beliefs); Pr(Px | Py) for beliefs.
  e. Start column: "P1" sits LEFT of its STR/DEF/SUP panel (as in 06-17),
     not above it. The boss-attack icon is centred ABOVE its "turn j" label
     in the header row (it used to sit to the right, ambiguously between
     turns).
  d. Start-column belief labels use spaced pipes ("Pr(P1 | P2)") — the
     unspaced pipe read as a capital I at 4.2 pt.

Previous (2026-09-02) notes, still accurate for everything else:

Successor to 2026-06-17-pnas-figure-revisions/team_case.py. Advisor
feedback (Tan Zhi-Xuan, 2026-08-31): the 06-17 figure shows humans adapting
but nothing in it shows that the *computational model* explains that
adaptation — the only model content is the observer posterior, and it is
not labelled as coming from the model. Changes:

  1. NEW "Pr(Px | BW)" chart per player per stage: the best-fitting
     model's (Bayesian-Walk-BR, agg_ll fit from the 05-25 pipeline / 05-28
     results.json) predicted distribution over that player's role at that
     stage, P(r_i | history). It sits INSIDE the role card, on a white strip
     along the card's bottom (the card is taller; the role icon and the
     per-turn action icons share ONE line above it, the actions aligned
     under the HP strip's turn columns), so the sub-row below the card
     holds only the two belief charts. P(chosen) is printed directly above
     the chosen bar (wide bar spacing keeps it clear). Labels are
     NOT repeated per column: the Pr(Px|Py) labels appear once per row under
     the Start column's prior charts, and the in-card chart is unlabelled
     (the legend names both). The "turn 1 / turn 2" labels and the boss-attack marker moved
     to a header row ABOVE the HP bars. A green/red caret
     under BW's most likely role says whether that is the role Px actually
     chose (= the card's role); P(chosen) is printed above the chosen bar.
  2. Belief charts keep the 06-17 timing (belief at the END of stage s in
     column s; the prior in the Start column) and the 06-17 carets: Py's
     OWN reported guess about Px over the guessed role, green = correct,
     red = wrong. Keyed "Pr(Px | Py) = Py's belief of Px". The prediction
     in column s is computed from the belief drawn in the column to its LEFT.
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
    uv run python experiments/2026-09-16-pnas-figure-revisions/team_case_model.py
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

# ---- style (standardised 2026-09-16; kept as switches so the legend and
# the card renderer cannot disagree) ----
NOTATION = "pi"           # "pi" -> π(Px | Model); "Pr" -> Pr(Px | Model)
CHOSEN_MARK = "outline"   # "outline" = bold dark outline round the chosen
                          # role's bar; "caret" = grey triangle under it
CARET_CHOSEN_COLOR = "#7f7f7f"
OUTLINE_COLOR = "#222222"
SUBHEAD_COLOR = "#666"    # "Prediction" / "Beliefs" subheaders (Start column)


def model_label(pid):
    """Per-player label of the model chart, e.g. 'Pr(P1 | Model)'."""
    head = "Pr" if NOTATION == "Pr" else "π"
    return f"{head}(P{pid + 1} | Model)"


def model_key():
    """Generic legend form, e.g. 'Pr(Px | Model)'."""
    head = "Pr" if NOTATION == "Pr" else "π"
    return f"{head}(Px | Model)"

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

# geometry: reuse 06-17 values, but WIDER stage columns so the three mini
# charts per stage have room. The 06-17 renderer pieces read COL_W/TURN_W
# from their module globals at call time, so patch them there first.
tc.COL_W = 1.25
tc.TURN_W = tc.COL_W / 2
tc.START_W = 1.30         # wider than 06-17 (0.95): P-id beside the stats, and the
                          # "Prediction" / "Beliefs" subheaders + per-row labels
COL_W, TURN_W, START_W = tc.COL_W, tc.TURN_W, tc.START_W
TRACK_H = 0.76            # taller than 06-17 (0.50): the card also holds the model chart strip (bars + numbers)
STRIP_Y0, STRIP_H = 0.04, 0.34   # model-chart strip inside the card (offset from card bottom, height)
BELIEF_H = 0.48           # 06-17: 0.30; +0.18 for the Start column's two-line subheader above the prior charts
SUB_GAP, TOP_GAP, HP_H = tc.SUB_GAP, tc.TOP_GAP, tc.HP_H
GROUP_GAP = 0.16          # wider than 06-17 so the belief labels clear the dashed separators
ROLE_COLORS = tc.ROLE_COLORS
TEAM_HP_COLOR, ENEMY_HP_COLOR = tc.TEAM_HP_COLOR, tc.ENEMY_HP_COLOR
CARET_OK_COLOR, CARET_BAD_COLOR = tc.CARET_OK_COLOR, tc.CARET_BAD_COLOR
BOSS_SVG = tc.BOSS_SVG
BW_PANEL = "#ffffff"      # backdrop behind the Pr(Px|BW) chart inside the role card
CELL_PAD = tc.CELL_PAD
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
# Renderer: player group with the model sub-row
# ──────────────────────────────────────────────────────────────────────

def _fmt_p(p):
    s = f"{p:.2f}"
    return s[1:] if s.startswith("0") else s


def _caret(ax, x, y, ok=None, color=None):
    """Caret under/over a bar. ok=True/False -> green (correct) / red (wrong):
    the human's reported guess over a belief chart. color overrides (the
    grey 'role actually chosen' caret in the model chart)."""
    if color is None:
        color = CARET_OK_COLOR if ok else CARET_BAD_COLOR
    ax.scatter([x], [y], marker="^", s=12, facecolor=color,
               edgecolor="none", zorder=6)


def _draw_player_group(ax, rec, turns, posteriors, preds, role_y, bel_y, pid):
    """P{pid}'s role track; beneath it ONE model sub-row holding, per stage,
    the two teammates' (model-inferred) beliefs about P{pid} entering the
    stage -- with the teammate's OWN reported guess as a green/red caret --
    and the Bayesian-Walk prediction of P{pid}'s role, with P{pid}'s actual
    choice outlined in black and its predicted probability printed."""
    n_stages = rec["n_stages"]
    stage_turns = rec["stage_turns"]

    # ---- Start column, card row: player id at the LEFT, compact STR/DEF/SUP
    # panel to its right (both on the card's icon line); below them, at the
    # height of the cards' white strips, the "Prediction" subheader and the
    # row's single π(Px | Model) label (not repeated per card). ----
    icon_cy = role_y + TRACK_H - 0.19          # same line as the role icons
    ax.text(0.16, icon_cy, f"P{pid + 1}", ha="center", va="center",
            fontsize=7, color="#222", fontweight="bold")
    st = [int(v) for v in rec["player_stats"][pid]]
    lab_x, bar_x0, bar_x1, stat_dy = 0.40, 0.66, 0.98, 0.105
    for k, name in enumerate(("STR", "DEF", "SUP")):
        row_cy = icon_cy + (1 - k) * stat_dy
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
    strip_mid = role_y + STRIP_Y0 + STRIP_H / 2
    ax.text(0.08, strip_mid + 0.045, "Model prediction", ha="left", va="center",
            fontsize=4.6, color=SUBHEAD_COLOR, fontweight="bold")
    ax.text(0.08, strip_mid - 0.065, model_label(pid), ha="left",
            va="center", fontsize=4.4, color="#444")

    # ---- role track: the 06-17 card (light role tint + border, role icon,
    # per-turn action icons) with the Bayesian-Walk prediction chart INSIDE
    # the card on its right-hand side, on a white panel. Bars = BW's predicted
    # distribution over P{pid}'s role at this stage (computed from the belief
    # charts in the column to the LEFT + P{pid}'s previous role). A caret
    # under BW's MOST LIKELY role is green if that is the role P{pid} actually
    # chose (= the card's own role) and red otherwise; P(chosen) is printed
    # above the chosen role's bar. ----
    for s in range(n_stages):
        role = rec["role_seq"][s][pid]
        acts = [t["actions"].get(pid, "?") for t in turns if t["s"] == s]
        _draw_role_card_with_pred(ax, s, role_y, role, acts,
                                  preds[s]["pred"][pid], pid)

    # ---- belief sub-row: two mini charts per stage column, Pr(P{pid} | Py)
    # for each teammate Py = the model's Bayesian inference of what Py
    # believes about P{pid}'s role at the END of stage s (posteriors[s + 1],
    # conditioned on Py's role during stage s) -- the 06-17 timing. The Start
    # column carries the prior (posteriors[0]). The caret over a chart is Py's
    # OWN reported guess about P{pid} (logged at stage s + 1, about stage s),
    # The Pr(Px|Py) labels appear ONCE per row, under the Start column's prior
    # charts (leftmost instance of each small-multiple row); stage columns are
    # unlabelled.
    # placed over the guessed role: green if the guess equals P{pid}'s stage-s
    # role, red if not. The prior and the last stage carry no caret (no report
    # exists). ----
    observers = [o for o in range(3) if o != pid]
    reports_by_stage = {}   # {stage logged (0-based): {reporter: guess of pid}}
    for si, obs_map in rec["inferred"].items():
        for reporter, guesses in obs_map.items():
            if pid in guesses:
                reports_by_stage.setdefault(si, {})[reporter] = guesses[pid]

    base_y = bel_y + 0.115
    slot_cx = [0.30 * COL_W, 0.70 * COL_W]
    # Start column: subheader on its own line above the two labelled prior
    # charts (labels ~0.45 units wide each at 4.0 pt)
    for k, ln in enumerate(("Model beliefs &", "player inferences")):
        ax.text(0.08, bel_y + BELIEF_H - 0.02 - k * 0.085, ln, ha="left",
                va="top", fontsize=4.6, color=SUBHEAD_COLOR, fontweight="bold")
    start_slots = [0.36, 0.92]

    for oi, obs in enumerate(observers):            # Start column: the prior
        belief = conditional_role_belief(posteriors[0], pid, obs,
                                         rec["role_seq"][0][obs])
        _mini_bars(ax, start_slots[oi], base_y, belief,
                   f"Pr(P{pid + 1} | P{obs + 1})", fs=4.0)
    for s in range(n_stages):
        truth = rec["role_seq"][s][pid]
        reports = reports_by_stage.get(s + 1, {})       # reports ABOUT stage s
        for oi, obs in enumerate(observers):
            belief = conditional_role_belief(posteriors[s + 1], pid, obs,
                                             rec["role_seq"][s][obs])
            bar_cx = _mini_bars(ax, col_x(s) + slot_cx[oi], base_y, belief,
                                "")                 # labelled once, at Start
            guess = reports.get(obs)
            if guess is not None:
                _caret(ax, bar_cx[guess], base_y - 0.05, ok=(guess == truth))


def _mini_bars(ax, hc, base_y, dist, label, number=None, max_h=0.13,
               mbw=0.06, mgap=0.02, label_dy=0.10, fs=4.2, number_y=None,
               outline=None, number_fs=4.2):
    """A 3-bar role distribution centred at `hc` on baseline `base_y`, with a
    label beneath. number=role prints that bar's value above it; number="all"
    prints every bar's value (all in the same plain style) -- directly above
    each bar, or on the fixed line `number_y` when given. outline=role draws a
    bold dark outline round that bar. Returns {role: bar centre x}."""
    ax.plot([hc - 0.13, hc + 0.13], [base_y, base_y], color="#ccc",
            linewidth=0.5, zorder=3)
    bar_cx = {}
    for role in range(3):
        bx = hc + (role - 1) * (mbw + mgap) - mbw / 2
        bar_cx[role] = bx + mbw / 2
        h = max(float(dist[role]) * max_h, 0.010)
        ax.add_patch(Rectangle((bx, base_y), mbw, h,
                               facecolor=ROLE_COLORS[role],
                               edgecolor="none", alpha=0.80, zorder=4))
        if outline is not None and role == outline:
            ax.add_patch(Rectangle((bx, base_y), mbw, h, facecolor="none",
                                   edgecolor=OUTLINE_COLOR, linewidth=0.9,
                                   joinstyle="miter", zorder=5))
        if number is not None and (number == "all" or role == number):
            ny = base_y + h + 0.012 if number_y is None else number_y
            ax.text(bx + mbw / 2, ny, _fmt_p(dist[role]),
                    ha="center", va="bottom", fontsize=number_fs, color="#222",
                    zorder=6)
    if label:
        ax.text(hc, base_y - label_dy, label, ha="center", va="top",
                fontsize=fs, color="#444", zorder=6)
    return bar_cx


def _draw_role_card_with_pred(ax, s, ty, role, actions_this_stage, pred, pid):
    """06-17 role card (light tint + border), taller: ONE icon line on top --
    the role icon centred with the per-turn action icons on the same line,
    aligned under the HP strip's turn columns -- and the model's prediction
    chart on a white strip below. 09-16: every bar carries its probability and
    the role Px actually chose is marked by a bold outline round its bar (or a
    grey caret, CHOSEN_MARK = "caret"). No correct/wrong colouring, no per-card
    label (the Start column labels the row)."""
    from matplotlib.patches import FancyBboxPatch
    x0 = col_x(s) + CELL_PAD
    w = COL_W - 2 * CELL_PAD
    bs = "round,pad=0,rounding_size=0.05"
    ax.add_patch(FancyBboxPatch((x0, ty), w, TRACK_H, boxstyle=bs,
                                facecolor=ROLE_COLORS[role], edgecolor="none",
                                alpha=0.15, zorder=2))
    ax.add_patch(FancyBboxPatch((x0, ty), w, TRACK_H, boxstyle=bs,
                                facecolor="none", edgecolor=ROLE_COLORS[role],
                                linewidth=1.1, alpha=0.85, zorder=3))
    # icon line: role icon centred, action icons under their turn columns
    icon_y = ty + TRACK_H - 0.19
    draw_role_icon(ax, role, col_x(s) + COL_W / 2, icon_y, size=0.30, zorder=5)
    for j, a in enumerate(actions_this_stage):
        if a in ("A", "B", "H"):
            draw_action_icon(ax, a, turn_x(s, j), icon_y, size=0.125, zorder=6)
    # bottom strip: white panel with the model's prediction chart (unlabelled
    # here -- the Start column carries the row's π(Px | Model) label)
    strip_y, strip_h = ty + STRIP_Y0, STRIP_H
    ax.add_patch(FancyBboxPatch((x0 + 0.04, strip_y), w - 0.08, strip_h,
                                boxstyle="round,pad=0,rounding_size=0.03",
                                facecolor=BW_PANEL, edgecolor="none",
                                alpha=0.9, zorder=3))
    base, max_h = strip_y + 0.075, 0.145
    hc = x0 + w / 2
    bar_cx = _mini_bars(ax, hc, base, pred, "", number="all", max_h=max_h,
                        mbw=0.09, mgap=0.065, number_fs=4.0,
                        outline=role if CHOSEN_MARK == "outline" else None)
    if CHOSEN_MARK == "caret":
        _caret(ax, bar_cx[role], base - 0.045, color=CARET_CHOSEN_COLOR)


def _draw_hp_strip(ax, rec, turns, hp_y, ranks):
    """Team/boss HP mini-bars per turn (06-17), but with the per-turn
    "turn 1 / turn 2" labels and the boss-attack marker moved ABOVE the bars,
    as a header row under the combo rank -- so each turn column is titled once
    at the top and nothing sits in the gap above P1's cards."""
    max_thp, max_ehp = rec["team_max_hp"], rec["enemy_max_hp"]
    bw = 0.15
    last_turn_of_stage = {tu["s"]: tu["t"] for tu in turns}
    head_y = hp_y + HP_H + 0.16          # baseline of the turn-header row
    boss_cy = head_y + 0.21              # boss-attack icon centred ABOVE "turn j"
    rank_y = head_y + 0.33               # combo-rank line above the boss icons

    # Start column: initial (full) HP
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
    ax.text(start_cx(), head_y, "initial", ha="center", va="bottom",
            fontsize=5, color="#999")

    for tu in turns:
        x = turn_x(tu["s"], tu["j"])
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
        # header row: "turn j" with the boss-attack marker beside it
        ax.text(x, head_y, f"turn {tu['j'] + 1}", ha="center", va="bottom",
                fontsize=5, color="#999")
        if tu["intent"] == 1:      # centred on the turn column, above its label
            draw_svg(ax, BOSS_SVG, x, boss_cy, size=0.13, zorder=5)
    ax.plot([0.0, col_x(turns[-1]["s"]) + COL_W], [hp_y, hp_y],
            color="#bbb", linewidth=0.6)
    # value-rank of the played combo, per stage, just under the stage headers
    for s in range(rec["n_stages"]):
        ax.text(col_x(s) + COL_W / 2, rank_y, f"combo rank {ranks[s]}/27",
                ha="center", va="bottom", fontsize=5.5, color="#666")


# ──────────────────────────────────────────────────────────────────────
# Legend: a vertical column to the LEFT of the diagram (uses the paper's
# full-width horizontal space instead of a band along the bottom)
# ──────────────────────────────────────────────────────────────────────

LEG_W = 1.60            # data-unit width reserved for the legend column
LEG_GAP = 0.65          # gap between the legend column and the Start column (text is in pt, layout in units: 5-stage renders need the extra room)
LEG_FS = 4.9            # legend font size


def _draw_legend_left(ax, x_left, y_center):
    """Draw the legend as one left-aligned column whose left edge is x_left,
    vertically centred on y_center. Returns (y_top, y_bottom)."""
    ICON_DX = 0.13              # icon-to-text offset for a 0.10-wide glyph
    LINE = 0.125                # baseline-to-baseline for wrapped text lines
    SEC_GAP = 0.42              # extra gap between the three legend sections
    ITEM_GAP = 0.08             # gap between items within a section
    CW = 0.042                  # approx text width per character at LEG_FS

    entries = []                # (height, draw(y_top))

    def text_lines(x, y_top, lines, color="#333"):
        for k, ln in enumerate(lines):
            ax.text(x, y_top - k * LINE, ln, ha="left", va="top",
                    fontsize=LEG_FS, color=color)

    def inline_row(items, gap=0.14):
        def draw(y_top):
            cy = y_top - 0.06
            x = x_left
            for icon, label in items:
                icon(x + 0.05, cy)
                ax.text(x + ICON_DX, cy, label, ha="left", va="center",
                        fontsize=LEG_FS, color="#333")
                x += ICON_DX + CW * len(label) + gap
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

    def bars_entry(lines, demo, caret=None, number=None, panel=False,
                   chosen=None, bw=0.040, gap=0.020, h_max=0.12,
                   number_fs=3.6):
        """Mini bar-chart key. caret=(role, ok) draws a green/red caret under
        that bar; number=role / "all" prints bar values above; panel draws the
        white in-card backdrop; chosen=role marks that bar the way the cards
        do (grey caret or bold outline, per CHOSEN_MARK). bw/gap/h_max let the
        demo match the real chart's geometry (the model key uses the in-card
        bar pitch so its three numbers have room)."""
        bars_w = 3 * bw + 2 * gap
        height = max(0.14 + (len(lines) - 1) * LINE + 0.02, 0.24)

        def draw(y_top):
            base = y_top - 0.16
            if panel:
                ax.add_patch(Rectangle((x_left - 0.03, base - 0.06), bars_w + 0.06,
                                       h_max + 0.13, facecolor=BW_PANEL,
                                       edgecolor="#ddd", linewidth=0.5, zorder=4))
            ax.plot([x_left - 0.008, x_left + bars_w + 0.008], [base, base],
                    color="#ccc", linewidth=0.5, zorder=4)
            for r in range(3):
                h = max(demo[r] * h_max, 0.010)
                bx = x_left + r * (bw + gap)
                ax.add_patch(Rectangle((bx, base), bw, h,
                                       facecolor=ROLE_COLORS[r],
                                       edgecolor="none", alpha=0.80, zorder=5))
                if chosen == r and CHOSEN_MARK == "outline":
                    ax.add_patch(Rectangle((bx, base), bw, h, facecolor="none",
                                           edgecolor=OUTLINE_COLOR, linewidth=0.9,
                                           joinstyle="miter", zorder=6))
                if number is not None and (number == "all" or r == number):
                    ax.text(bx + bw / 2, base + h + 0.012, _fmt_p(demo[r]),
                            ha="center", va="bottom", fontsize=number_fs,
                            color="#222", zorder=6)
            if caret is not None:
                r, ok = caret
                _caret(ax, x_left + r * (bw + gap) + bw / 2, base - 0.045, ok)
            if chosen is not None and CHOSEN_MARK == "caret":
                _caret(ax, x_left + chosen * (bw + gap) + bw / 2, base - 0.045,
                       color=CARET_CHOSEN_COLOR)
            text_lines(x_left + bars_w + 0.07, y_top - 0.01, lines)
        return height, draw

    def caret_entry(ok, lines, color=None):
        def draw(y_top):
            _caret(ax, x_left + 0.05, y_top - 0.05, ok, color=color)
            text_lines(x_left + ICON_DX, y_top - 0.01, lines)
        return 0.04 + LINE * len(lines), draw

    def outline_entry(lines):
        def draw(y_top):
            ax.add_patch(Rectangle((x_left + 0.03, y_top - 0.10), 0.04, 0.09,
                                   facecolor=ROLE_COLORS[1], edgecolor="none",
                                   alpha=0.8, zorder=5))
            ax.add_patch(Rectangle((x_left + 0.03, y_top - 0.10), 0.04, 0.09,
                                   facecolor="none", edgecolor=OUTLINE_COLOR,
                                   linewidth=0.9, joinstyle="miter", zorder=6))
            text_lines(x_left + ICON_DX, y_top - 0.01, lines)
        return 0.04 + LINE * len(lines), draw

    def spacer(h):
        return h, (lambda y_top: None)

    entries += [
        inline_row([(role_icon(0), "fighter"), (role_icon(1), "tank"),
                    (role_icon(2), "medic")]),
        spacer(ITEM_GAP),
        inline_row([(action_icon("A"), "attack"), (action_icon("B"), "block"),
                    (action_icon("H"), "heal")]),
        spacer(ITEM_GAP),
        inline_row([(hp_icon, "team / boss HP"), (boss_icon, "boss attacks")],
                   gap=0.45),
        spacer(SEC_GAP),
        # model block FIRST (it sits above the belief charts in the figure)
        bars_entry([f"{model_key()}: model's predicted",
                    "role of Px"],
                   [0.20, 0.62, 0.18], number="all", panel=True, chosen=1,
                   bw=0.09, gap=0.065, h_max=0.145, number_fs=4.0),
        spacer(ITEM_GAP),
        (caret_entry(None, ["role Px chose"], color=CARET_CHOSEN_COLOR)
         if CHOSEN_MARK == "caret" else
         outline_entry(["role Px chose"])),
        spacer(SEC_GAP),
        bars_entry(["Pr(Px | Py): model of Py's",
                    "belief about Px's role"],
                   [0.55, 0.30, 0.15], caret=(0, True)),
        spacer(ITEM_GAP),
        caret_entry(True, ["Py correctly infers Px's role",
                           "after seeing their actions"]),
        spacer(ITEM_GAP),
        caret_entry(False, ["Py wrongly infers Px's role",
                            "after seeing their actions"]),
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

    hp_y = -HP_H - 0.47          # room above the bars for turn labels, boss icons, combo rank
    role_ys, bel_ys = [], []
    yy = hp_y - 0.08
    for _ in range(3):
        yy -= TOP_GAP
        role_y = yy - TRACK_H
        bel_y = role_y - SUB_GAP - BELIEF_H
        role_ys.append(role_y)
        bel_ys.append(bel_y)
        yy = bel_y - GROUP_GAP

    # legend column on the left (x < 0), then Start column + stages
    leg_x_left = -(LEG_GAP + LEG_W)
    x_lo = leg_x_left - 0.05
    x_hi = START_W + n_stages * COL_W + 0.25

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    tc._draw_stage_headers(ax, n_stages, turns, y_top=0.0,
                           y_bottom=bel_ys[-1] + 0.02)
    _draw_hp_strip(ax, rec, turns, hp_y, ranks)
    for pid in range(3):
        _draw_player_group(ax, rec, turns, posteriors, preds, role_ys[pid],
                           bel_ys[pid], pid)

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
    """(hits, total) over (stage, observer, target): does the model's most
    likely role for the target under the END-of-stage-s belief
    (posteriors[s + 1], conditioned on the observer's stage-s role) equal the
    role the target played in stage s? Not drawn on the figure; kept for the
    summary."""
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


def human_report_accuracy(rec):
    """(hits, total) over the humans' logged inference reports: a report
    logged at stage s about target t is correct iff it equals t's stage-(s-1)
    role. These are the green/red carets over the belief charts."""
    hits = tot = 0
    for si, obs_map in rec["inferred"].items():
        if si < 1 or si >= rec["n_stages"]:
            continue
        for reporter, guesses in obs_map.items():
            for tgt, g in guesses.items():
                if tgt == reporter:
                    continue
                hits += int(g == rec["role_seq"][si - 1][tgt])
                tot += 1
    return hits, tot


def case_summary_md(rec, preds, ranks, dom, inf_acc, rep_acc):
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
        f"chosen role in **{top_hits}/{3 * n}** player-stages (not marked on "
        "the figure since 09-16; read off the in-card numbers). Humans' own reported inferences "
        f"correct in **{rep_acc[0]}/{rep_acc[1]}** reports (green carets, "
        "belief charts). Model's most likely inferred role "
        f"(end-of-stage belief) = played role in {inf_acc[0]}/{inf_acc[1]} "
        "cells (not drawn).",
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
        "# R3_team_case with best-fitting model predictions (2026-09-16)",
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
        rep_acc = human_report_accuracy(rec)
        traj_str = " -> ".join("".join(ROLE_SHORT[r] for r in rec["role_seq"][s])
                               for s in range(rec["n_stages"]))
        print(f"[team-case-model] {rec['game_id']} r{rec['round_number']} "
              f"({rec['stat_profile_id']}): {traj_str}, {rec['outcome']}, "
              f"{rec['n_stages']} live stages; ranks {ranks}; "
              f"model inference acc {inf_acc[0]}/{inf_acc[1]}; "
              f"human reports correct {rep_acc[0]}/{rep_acc[1]}")
        names = [name] + ([PRIMARY_NAME] if name == CASES[0][2] else [])
        render(rec, posteriors, preds, names)
        md.append(case_summary_md(rec, preds, ranks, dom, inf_acc, rep_acc))

    OUT_MD.write_text("\n".join(md))
    print(f"[team-case-model] wrote {OUT_MD}")


if __name__ == "__main__":
    main()
