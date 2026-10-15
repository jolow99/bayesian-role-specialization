"""Fig. 4 (R3_team_case) legend / whitespace DESIGN OPTIONS (2026-10-15).

Imports the 2026-09-16 generator unchanged (data, model predictions, diagram
drawing) and replaces ONLY the legend placement / figure layout.

Layouts
  current : the paper's figure (09-16 `render`, left legend column)
  A       : compact left column (stacked key text, tight, justified to height)
  B       : two-row band BELOW the diagram
  C       : split band: icon/header keys ABOVE the diagram, chart keys BELOW
  D       : B-style bottom band but diagram kept at the ORIGINAL text scale
            with tighter columns (shorter figure)  [see README]

Every option adds a short key for the "value v (best b)" header.

Scale handling: the figure is always 7 in wide. When the legend leaves the
left column the diagram gets bigger on the page, so all diagram text/marker
sizes are multiplied by K (= the on-page magnification vs. the paper figure,
optionally capped) by wrapping Axes.text / Axes.scatter. K = 1 reproduces the
paper's absolute font sizes.

Run:  cd analysis && .venv/bin/python experiments/2026-10-15-fig4-legend-options/legend_options.py [--case 2F8H1E:2] [--only A,B]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text
import matplotlib.collections
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
DIR_0916 = HERE.parent / "2026-09-16-pnas-figure-revisions"
sys.path.insert(0, str(DIR_0916))
import team_case_model as tm  # noqa: E402  (imports 06-16/06-17/05-25 pieces)
from shared.constants import ROLE_SHORT  # noqa: E402

tc = tm.tc
ROLE_COLORS = tm.ROLE_COLORS
TEAM_HP_COLOR, ENEMY_HP_COLOR = tm.TEAM_HP_COLOR, tm.ENEMY_HP_COLOR
OUTLINE_COLOR = tm.OUTLINE_COLOR

FIG_W = 7.0
ORIG_SUB_GAP = tm.SUB_GAP
ORIG_HP_H = tm.HP_H
OLD_IN_PER_UNIT = 0.658     # NOMINAL scale of the paper figure's code (5.59 in native / ~8.5 units); the PDF is then placed at 7 in (x1.25)

# ---- text / marker magnification (K) -------------------------------------
_orig_text = Axes.text
_orig_scatter = Axes.scatter
K = 1.0


def _text(self, x, y, s, *a, **kw):
    kw["fontsize"] = kw.get("fontsize", 7) * K
    return _orig_text(self, x, y, s, *a, **kw)


def _scatter(self, *a, **kw):
    if "s" in kw:
        kw["s"] = kw["s"] * K * K
    return _orig_scatter(self, *a, **kw)


Axes.text = _text
Axes.scatter = _scatter


# ---- text measuring -------------------------------------------------------
_scratch = plt.figure(figsize=(2, 2), dpi=72)
_rend = _scratch.canvas.get_renderer()


def tw_pt(s, fs, weight="normal"):
    t = _orig_text(_scratch.gca(), 0, 0, s, fontsize=fs, fontweight=weight)
    w = t.get_window_extent(_rend).width       # px at 72 dpi == pt
    t.remove()
    return w


# ---- case preparation (mirrors tm.main) ----------------------------------
def prepare_case(suffix, rnd):
    params, s1_res, res = tm.load_best_model_params()
    s1, strat = tm.load_stage1_canonical()
    records = tm.load_human_records()
    traj_by_key = tm.build_trajectories(s1)
    m = [r for r in records if r["game_id"].endswith(suffix)
         and r["round_number"] == rnd]
    assert len(m) == 1
    rec = tc.trim_to_live(m[0])
    traj, values = traj_by_key[(rec["game_id"], rec["round_number"])]
    preds = tm.walk_predictions(traj[:rec["n_stages"]], values, params)
    posteriors = tm.human_posteriors(rec, s1, strat)
    return rec, posteriors, preds


# ---- the diagram (identical to tm.render, minus the legend) --------------
def draw_diagram(ax, rec, posteriors, preds):
    n_stages = rec["n_stages"]
    turns = tc.flatten_turns(rec)
    vals = tm.stage_values(rec)
    hp_y = -tm.HP_H - 0.47
    role_ys, bel_ys = [], []
    yy = hp_y - 0.08
    for _ in range(3):
        yy -= tm.TOP_GAP
        role_y = yy - tm.TRACK_H
        bel_y = role_y - tm.SUB_GAP - tm.BELIEF_H
        role_ys.append(role_y)
        bel_ys.append(bel_y)
        yy = bel_y - tm.GROUP_GAP
    tc._draw_stage_headers(ax, n_stages, turns, y_top=0.0,
                           y_bottom=bel_ys[-1] + 0.02)
    tm._draw_hp_strip(ax, rec, turns, hp_y, vals)
    for pid in range(3):
        tm._draw_player_group(ax, rec, turns, posteriors, preds, role_ys[pid],
                              bel_ys[pid], pid)
    x1 = tm.col_x(n_stages - 1) + tm.COL_W
    for i in range(2):
        y_sep = (bel_ys[i] + role_ys[i + 1] + tm.TRACK_H) / 2
        ax.plot([0.0, x1], [y_sep, y_sep], color="#bbb", linewidth=0.7,
                linestyle=(0, (4, 3)), zorder=1)


# ---- legend widgets ------------------------------------------------------
class Legend:
    """Widgets are (width, height, draw(x_left, y_center)) in data units."""

    def __init__(self, ax, upp, fs):
        self.ax, self.upp, self.fs = ax, upp, fs
        self.LH = fs * 1.22 * upp          # line height

    def P(self, pt):
        return pt * self.upp

    def tw(self, s, fs=None):
        return self.P(tw_pt(s, fs or self.fs))

    def T(self, x, y, s, **kw):
        kw.setdefault("color", "#333")
        kw.setdefault("ha", "left")
        kw.setdefault("va", "center")
        return _orig_text(self.ax, x, y, s, fontsize=self.fs, **kw)

    def lines(self, x, yc, lines):
        n = len(lines)
        for k, ln in enumerate(lines):
            self.T(x, yc + ((n - 1) / 2 - k) * self.LH, ln)

    # -- icon + label
    def icon_label(self, icon, label):
        P = self.P
        w = P(9) + self.tw(label)

        def draw(x, yc):
            icon(x + P(3.6), yc)
            self.T(x + P(8.6), yc, label)
        return w, P(8), draw

    def role_icon(self, r):
        P, ax = self.P, self.ax

        def icon(cx, cy):
            s = P(7.4)
            ax.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s,
                                   facecolor=ROLE_COLORS[r],
                                   edgecolor=ROLE_COLORS[r], linewidth=0.6,
                                   alpha=0.20, zorder=3))
            tm.draw_role_icon(ax, r, cx, cy, size=P(7.4), zorder=5)
        return icon

    def action_icon(self, a):
        return lambda cx, cy: tm.draw_action_icon(self.ax, a, cx, cy,
                                                  size=self.P(7.4), zorder=5)

    def hp_icon(self, cx, cy):
        P, ax = self.P, self.ax
        hbw, hh = P(2.8), P(7.2)
        ax.add_patch(Rectangle((cx - P(3.2), cy - hh / 2), hbw, hh,
                               facecolor=TEAM_HP_COLOR, edgecolor="none",
                               alpha=0.9, zorder=5))
        ax.add_patch(Rectangle((cx + P(0.3), cy - hh / 2), hbw, hh * 0.6,
                               facecolor=ENEMY_HP_COLOR, edgecolor="none",
                               alpha=0.9, zorder=5))

    def boss_icon(self, cx, cy):
        tm.draw_svg(self.ax, tm.BOSS_SVG, cx, cy, size=self.P(7.8), zorder=5)

    # -- value header key: "value v (best b)" styled like the header
    def value_key(self, lines_after):
        P = self.P
        head = "value (best)"
        hw = self.P(tw_pt(head, self.fs * 1.1))
        tail_w = max(self.tw(l) for l in lines_after)
        w = hw + P(3) + tail_w
        n = len(lines_after)

        def draw(x, yc):
            _orig_text(self.ax, x, yc, head, fontsize=self.fs * 1.1,
                       color="#666", ha="left", va="center")
            self.lines(x + hw + P(3), yc, lines_after)
        return w, max(P(8), n * self.LH), draw

    # -- mini-bar chart key + text
    def bars_key(self, demo, lines, *, bw, gap, hmax, number=False,
                 panel=False, chosen=None, caret=None, num_fs=4.0):
        P = self.P
        cw = 3 * bw + 2 * gap
        pad = P(2.2) if panel else 0.0
        text_w = max(self.tw(l) for l in lines)
        w = cw + 2 * pad + P(3.5) + text_w
        h = max(hmax + P(8) + (P(5) if caret else 0), len(lines) * self.LH)

        def draw(x, yc):
            ax = self.ax
            base = yc - hmax / 2 - P(1.5)
            bx0 = x + pad
            if panel:
                ax.add_patch(Rectangle((x, base - P(2.6)), cw + 2 * pad,
                                       hmax + P(9.6), facecolor="white",
                                       edgecolor="#ddd", linewidth=0.5,
                                       zorder=4))
            ax.plot([bx0 - P(0.5), bx0 + cw + P(0.5)], [base, base],
                    color="#ccc", linewidth=0.5, zorder=4)
            for r in range(3):
                hh = max(demo[r] * hmax, 0.010)
                bx = bx0 + r * (bw + gap)
                ax.add_patch(Rectangle((bx, base), bw, hh,
                                       facecolor=ROLE_COLORS[r],
                                       edgecolor="none", alpha=0.8, zorder=5))
                if chosen == r:
                    ax.add_patch(Rectangle((bx, base), bw, hh, facecolor="none",
                                           edgecolor=OUTLINE_COLOR,
                                           linewidth=0.9, joinstyle="miter",
                                           zorder=6))
                if number:
                    _orig_text(ax, bx + bw / 2, base + hh + P(0.7),
                               tm._fmt_p(demo[r]), fontsize=num_fs * K,
                               ha="center", va="bottom", color="#222",
                               zorder=6)
            if caret is not None:
                r, ok = caret
                tm._caret(ax, bx0 + r * (bw + gap) + bw / 2, base - P(3.2), ok)
            self.lines(x + cw + 2 * pad + P(3.5), yc, lines)
        return w, h, draw

    def outline_key(self, lines):
        P = self.P
        w = P(9) + max(self.tw(l) for l in lines)

        def draw(x, yc):
            ax = self.ax
            bw_, bh = P(3.4), P(7.2)
            ax.add_patch(Rectangle((x + P(1), yc - bh / 2), bw_, bh,
                                   facecolor=ROLE_COLORS[1], edgecolor="none",
                                   alpha=0.8, zorder=5))
            ax.add_patch(Rectangle((x + P(1), yc - bh / 2), bw_, bh,
                                   facecolor="none", edgecolor=OUTLINE_COLOR,
                                   linewidth=0.9, joinstyle="miter", zorder=6))
            self.lines(x + P(7.6), yc, lines)
        return w, max(P(8), len(lines) * self.LH), draw

    def caret_key(self, oks, lines):
        """one or two carets (ok flags) followed by text"""
        P = self.P
        cw = P(5.0) * len(oks)
        w = cw + P(2) + max(self.tw(l) for l in lines)

        def draw(x, yc):
            for i, ok in enumerate(oks):
                tm._caret(self.ax, x + P(2.4) + i * P(5.0), yc, ok)
            self.lines(x + cw + P(2), yc, lines)
        return w, max(P(7), len(lines) * self.LH), draw

    # -- layout helpers
    def row(self, widgets, x0, x1, yc, max_gap=None, min_gap=None):
        P = self.P
        tot = sum(w for w, _, _ in widgets)
        n = len(widgets)
        gap = (x1 - x0 - tot) / (n - 1) if n > 1 else 0
        if min_gap is None:
            min_gap = P(3)
        if gap < min_gap:
            print(f"   !! legend row overflows by {(min_gap - gap) * (n - 1) / self.upp:.1f} pt")
        if max_gap is not None and gap > max_gap:
            gap = max_gap
        x = x0
        for w, _, draw in widgets:
            draw(x, yc)
            x += w + gap
        return gap

    def row_height(self, widgets):
        return max(h for _, h, _ in widgets)


def keys(L, mode):
    """Build all key widgets. mode 'col' -> wrapped multi-line text,
    'band' -> single-line text."""
    col = mode == "col"
    P = L.P
    k = {}
    k["fighter"] = L.icon_label(L.role_icon(0), "fighter")
    k["tank"] = L.icon_label(L.role_icon(1), "tank")
    k["medic"] = L.icon_label(L.role_icon(2), "medic")
    k["attack"] = L.icon_label(L.action_icon("A"), "attack")
    k["block"] = L.icon_label(L.action_icon("B"), "block")
    k["heal"] = L.icon_label(L.action_icon("H"), "heal")
    k["hp"] = L.icon_label(L.hp_icon, "team / boss HP")
    k["boss"] = L.icon_label(L.boss_icon, "boss attacks")
    k["value"] = L.value_key(
        ["expected return of", "chosen roles", "(best possible)"] if col else
        ["expected return of chosen", "roles (best possible)"])
    k["pi"] = L.bars_key(
        [0.20, 0.62, 0.18],
        ["π(Px | Model):", "model-predicted", "role of Px"] if col else
        ["π(Px | Model): predicted role of Px"],
        number=True, panel=True, chosen=1, bw=0.09, gap=0.065, hmax=0.145)
    k["chose"] = L.outline_key(["role Px chose"])
    k["pr"] = L.bars_key(
        [0.55, 0.30, 0.15],
        ["Pr(Px | Py):", "Py's belief about", "Px's role"] if col else
        ["Pr(Px | Py): Py's belief about Px's role"],
        bw=0.04, gap=0.02, hmax=0.12, caret=(0, True))
    if col:
        k["carets"] = L.caret_key([True], ["Py correctly infers", "Px's role after", "seeing their actions"])
        k["carets2"] = L.caret_key([False], ["Py wrongly infers", "Px's role after", "seeing their actions"])
    else:
        k["carets"] = L.caret_key([True, False],
                                  ["Py correctly / wrongly infers Px's role"])
    return k


# ---- layouts -------------------------------------------------------------
def legend_A(ax, upp, fs, ext, K_, compact=True):
    """compact left column, justified to the diagram's height"""
    L = Legend(ax, upp, fs)
    k = keys(L, "col")
    P = L.P
    secs = [
        [[k["fighter"], k["tank"], k["medic"]],
         [k["attack"], k["block"], k["heal"]],
         [k["hp"]], [k["boss"]], [k["value"]]],
        [[k["pi"]], [k["chose"]]],
        [[k["pr"]], [k["carets"]], [k["carets2"]]],
    ]
    col_w = max(sum(w for w, _, _ in r) + P(7) * (len(r) - 1)
                for s in secs for r in s)
    x_left = ext["x0"] - L.P(5) - col_w
    ITEM = P(3.5)
    heights = []
    for s in secs:
        heights.append(sum(L.row_height(r) for r in s) + ITEM * (len(s) - 1))
    top = ext["y1"] - P(2)
    bot = ext["y0"] + P(2)
    free = (top - bot) - sum(heights)
    sec_gap = P(12) if compact else free / (len(secs) - 1)
    y = top
    for s, h in zip(secs, heights):
        for r in s:
            rh = L.row_height(r)
            x = x_left
            for w, _, draw in r:
                draw(x, y - rh / 2)
                x += w + P(7)
            y -= rh + ITEM
        y += ITEM
        y -= sec_gap
    return {"col_w_pt": col_w / upp}


def band_rows(ax, upp, fs, ext, which):
    L = Legend(ax, upp, fs)
    k = keys(L, "band")
    r1 = [k["fighter"], k["tank"], k["medic"], k["attack"], k["block"],
          k["heal"], k["hp"], k["boss"], k["value"]]
    r2 = [k["pi"], k["chose"], k["pr"], k["carets"]]
    return L, r1, r2


def legend_B(ax, upp, fs, ext, K_):
    L, r1, r2 = band_rows(ax, upp, fs, ext, "B")
    P = L.P
    x0, x1 = ext["x0"], ext["x1"]
    h1, h2 = L.row_height(r1), L.row_height(r2)
    y = ext["y0"] - P(7)
    # hairline separating legend from the diagram
    ax.plot([x0, x1], [y + P(2.5), y + P(2.5)], color="#ccc", linewidth=0.5)
    y1 = y - P(2) - h1 / 2
    y2 = y1 - h1 / 2 - P(5) - h2 / 2
    L.row(r1, x0, x1, y1, max_gap=P(28))
    L.row(r2, x0, x1, y2, max_gap=P(28))


def legend_C(ax, upp, fs, ext, K_):
    """icon/header keys above the diagram, chart keys below"""
    L, r1, r2 = band_rows(ax, upp, fs, ext, "C")
    P = L.P
    x0, x1 = ext["x0"], ext["x1"]
    h1, h2 = L.row_height(r1), L.row_height(r2)
    yt = ext["y1"] + P(6) + h1 / 2
    L.row(r1, x0, x1, yt, max_gap=P(28))
    yb = ext["y0"] - P(8) - h2 / 2
    ax.plot([x0, x1], [yb + h2 / 2 + P(4), yb + h2 / 2 + P(4)],
            color="#ccc", linewidth=0.5)
    L.row(r2, x0, x1, yb, max_gap=P(28))


LAYOUTS = {
    "A": dict(fn=legend_A, fs=5.6, kcap=3.0),
    "B": dict(fn=legend_B, fs=5.1, kcap=3.0),
    "C": dict(fn=legend_C, fs=5.1, kcap=3.0),
    "D": dict(fn=legend_B, fs=5.1, kcap=3.0, belief_h=0.45, sub_gap=0.02, hp_h=0.30),   # B + tightened vertical rhythm
}


# ---- driver ---------------------------------------------------------------
def tight_data_bbox(fig, ax):
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    xs, ys = [], []
    for a in list(ax.texts) + list(ax.patches) + list(ax.lines) + list(ax.collections):
        try:
            bb = a.get_window_extent(r)
        except Exception:
            continue
        if bb is None or not np.isfinite(bb.get_points()).all():
            continue
        if isinstance(a, matplotlib.text.Text) and not a.get_text():
            continue
        p = inv.transform(bb.get_points())
        if isinstance(a, matplotlib.collections.PathCollection):
            o = np.asarray(a.get_offsets())
            xs += list(o[:, 0] - 0.09) + list(o[:, 0] + 0.09)
            ys += list(o[:, 1] - 0.09) + list(o[:, 1] + 0.09)
            continue
        xs += [p[0][0], p[1][0]]
        ys += [p[0][1], p[1][1]]
    return min(xs), min(ys), max(xs), max(ys)


def new_fig(xr, yc, xc):
    fig = plt.figure(figsize=(FIG_W, FIG_W), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_aspect("equal")
    ax.set_xlim(xc - xr / 2, xc + xr / 2)
    ax.set_ylim(yc - xr / 2, yc + xr / 2)
    ax.axis("off")
    return fig, ax


def render_layout(name, rec, posteriors, preds, out_stems):
    global K
    cfg = LAYOUTS[name]
    tm.BELIEF_H = cfg.get("belief_h", 0.48)
    tm.HP_H = cfg.get("hp_h", ORIG_HP_H)
    tm.SUB_GAP = cfg.get("sub_gap", ORIG_SUB_GAP)
    n = rec["n_stages"]
    xr = 9.0
    PAD_IN = 0.03
    for it in range(8):
        in_per_unit = FIG_W / xr
        K = max(1.0, min(cfg["kcap"], in_per_unit / OLD_IN_PER_UNIT))
        upp = xr / (FIG_W * 72)                      # data units per pt
        # 1) diagram extents
        fig, ax = new_fig(xr, -2, 3.3)
        draw_diagram(ax, rec, posteriors, preds)
        x0, y0, x1, y1 = tight_data_bbox(fig, ax)
        plt.close(fig)
        ext = dict(x0=x0, x1=x1, y0=y0, y1=y1)
        # 2) diagram + legend
        fig, ax = new_fig(xr, -2, 3.3)
        draw_diagram(ax, rec, posteriors, preds)
        info = cfg["fn"](ax, upp, cfg["fs"], ext, K)
        bx0, by0, bx1, by1 = tight_data_bbox(fig, ax)
        pad = PAD_IN * xr / FIG_W
        xr_new = (bx1 - bx0) + 2 * pad
        if abs(xr_new - xr) < 0.004 or it == 7:
            ax.set_xlim(bx0 - pad, bx1 + pad)
            ax.set_ylim(by0 - pad, by1 + pad)
            h = FIG_W * (by1 - by0 + 2 * pad) / (bx1 - bx0 + 2 * pad)
            fig.set_size_inches(FIG_W, h)
            for stem in out_stems:
                fig.savefig(HERE / f"{stem}.png", dpi=300, facecolor="white")
                fig.savefig(HERE / f"{stem}.pdf", facecolor="white")
            plt.close(fig)
            dgm_w = (ext["x1"] - ext["x0"]) / (bx1 - bx0 + 2 * pad)
            dgm_h = (ext["y1"] - ext["y0"]) / (by1 - by0 + 2 * pad)
            print(f"[{name}] {out_stems[0]}: {FIG_W:.2f} x {h:.2f} in "
                  f"(aspect {FIG_W / h:.2f}:1), K={K:.2f}, diagram covers "
                  f"{dgm_w * dgm_h * 100:.0f}% of area; info={info}")
            return dict(h=h, K=K, dgm_frac=dgm_w * dgm_h)
        plt.close(fig)
        xr = xr_new


def render_current(rec, posteriors, preds, stem):
    global K
    K = 1.0
    saved = tm.OUT_DIR
    tm.OUT_DIR = HERE
    tm.render(rec, posteriors, preds, [stem])
    tm.OUT_DIR = saved


def main():
    global K
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="2F8H1E:2")
    ap.add_argument("--only", default="")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--current", action="store_true")
    a = ap.parse_args()
    suffix, rnd = a.case.split(":")
    rec, posteriors, preds = prepare_case(suffix, int(rnd))
    tag = a.suffix
    if a.current or not a.only:
        render_current(rec, posteriors, preds, f"current{tag}")
    K = 1.0
    for nm in (a.only.split(",") if a.only else LAYOUTS):
        render_layout(nm, rec, posteriors, preds, [f"option_{nm}{tag}"])


if __name__ == "__main__":
    main()
