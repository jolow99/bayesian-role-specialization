"""D2 / D3 follow-ups (2026-10-15): grouped legend band + height-reduction levers.

Imports legend_options (unchanged; A-D outputs untouched) and adds
  * legend_grouped: G1 roles | G2 actions | G3 HP+boss (+value key) | G4 pi + chose | G5 Pr + carets
  * parametrised copies of the 09-16 HP-strip and player-group drawers (source
    patched at import time, so the drawing logic stays the 09-16 code)
  * render_variant(name, cfg): same two-stage fit-to-7in driver as legend_options

Run: cd analysis && .venv/bin/python experiments/2026-10-15-fig4-legend-options/grouped_variants.py [--case K72DV0:5 --suffix _5stage] [--ablate]
"""
from __future__ import annotations
import argparse, inspect
import matplotlib.pyplot as plt
import numpy as np
import legend_options as lo
from legend_options import tm, tc, HERE, FIG_W

# ---- parametrised copies of the 09-16 drawers ------------------------------
def _patched(fn, name, reps):
    src = inspect.getsource(fn)
    src = src.replace(f"def {fn.__name__}", f"def {name}", 1)
    for a, b in reps:
        assert a in src, a
        src = src.replace(a, b)
    exec(src, tm.__dict__)
    return tm.__dict__[name]

_hp = _patched(tm._draw_hp_strip, "_draw_hp_strip_v2", [
    ("head_y = hp_y + HP_H + 0.16", "head_y = hp_y + HP_H + HEAD_OFF"),
    ("rank_y = head_y + 0.33", "rank_y = head_y + RANK_OFF"),
    ("    for s in range(rec[\"n_stages\"]):\n        vp, vb = vals[s]", "    for s in range(0 if MERGE_HDR else rec[\"n_stages\"]):\n        vp, vb = vals[s]"),
    ("draw_svg(ax, BOSS_SVG, x, boss_cy, size=0.13, zorder=5)",
     "draw_svg(ax, BOSS_SVG, x + BOSS_DX, head_y + BOSS_DY, size=BOSS_S, zorder=5)"),
])
_pg = _patched(tm._draw_player_group, "_draw_player_group_v2", [
    ("base_y = bel_y + 0.115", "base_y = bel_y + BASE_OFF"),
    ('f"Pr(P{pid + 1} | P{obs + 1})", fs=4.0)', 'f"Pr(P{pid + 1} | P{obs + 1})", fs=4.0, label_dy=LABEL_DY)'),
])

def center_line(ax, y, parts, fs):
    """One line centred in the Start column. parts = [(text, bold)]; fs nominal."""
    cx = tm.start_cx()
    ws = [lo.tw_pt(t, fs * lo.K, "bold" if b else "normal") * tm.UPP for t, b in parts]
    gap = 0.45 * fs * lo.K * tm.UPP
    tot = sum(ws) + gap * (len(parts) - 1)
    tm.LAST_LINE_W = max(getattr(tm, "LAST_LINE_W", 0), tot)
    x = cx - tot / 2
    for (t, b), w in zip(parts, ws):
        ax.text(x, y, t, ha="left", va="center", fontsize=fs if b else fs - 0.2,
                color=tm.SUBHEAD_COLOR if b else "#444", fontweight="bold" if b else "normal")
        x += w + gap
tm._center_line = center_line

_pg3_src_reps = [
    ("base_y = bel_y + 0.115", "base_y = bel_y + BASE_OFF"),
    ('f"Pr(P{pid + 1} | P{obs + 1})", fs=4.0)', 'f"Pr(P{pid + 1} | P{obs + 1})", fs=4.0, label_dy=LABEL_DY)'),
    ("start_slots = [0.36, 0.92]", "start_slots = [start_cx() - 0.28, start_cx() + 0.28]"),
]
_src = inspect.getsource(tm._draw_player_group)
_a = _src.index('    ax.text(0.08, strip_mid + 0.045')
_b = _src.index('fontsize=4.4, color="#444")', _a) + len('fontsize=4.4, color="#444")')
_src = _src[:_a] + '    _center_line(ax, strip_mid, [("Model prediction", True), (model_label(pid), False)], SUB_FS)' + _src[_b:]
_a = _src.index('    for k, ln in enumerate(("Model beliefs &"')
_b = _src.index('fontweight="bold")', _a) + len('fontweight="bold")')
_src = _src[:_a] + '    _center_line(ax, bel_y + BELIEF_H - SH_TOP - 0.05, [(BEL_TEXT, True)], SUB_FS)' + _src[_b:]
_src = _src.replace("def _draw_player_group", "def _draw_player_group_v3", 1)
for a_, b_ in _pg3_src_reps:
    assert a_ in _src, a_
    _src = _src.replace(a_, b_)
exec(_src, tm.__dict__)
_pg3 = tm._draw_player_group_v3

DEFAULTS = dict(track_h=0.76, belief_h=0.45, sub_gap=0.02, top_gap=0.05, group_gap=0.16,
                hp_h=0.30, head_off=0.16, rank_off=0.33, boss_dx=0.0, boss_dy=0.21,
                boss_s=0.13, title_y=0.0, sh_top=0.02, sh_lh=0.085, base_off=0.115,
                label_dy=0.10, v3=False, sub_fs=4.6, start_w=1.30, bel_text="Model beliefs & player inferences")


def set_globals(c):
    tm.TRACK_H = c["track_h"]; tm.BELIEF_H = c["belief_h"]; tm.SUB_GAP = c["sub_gap"]
    tm.HP_H = c["hp_h"]
    tm.HEAD_OFF = c["head_off"]; tm.RANK_OFF = c["rank_off"]
    tm.BOSS_DX = c["boss_dx"]; tm.BOSS_DY = c["boss_dy"]; tm.BOSS_S = c["boss_s"]
    tm.SH_TOP = c["sh_top"]; tm.SH_LH = c["sh_lh"]; tm.BASE_OFF = c["base_off"]
    tm.LABEL_DY = c["label_dy"]
    tm.SUB_FS = c["sub_fs"]; tm.BEL_TEXT = c["bel_text"]
    tc.START_W = tm.START_W = c["start_w"]
    tm.LAST_LINE_W = 0
    tm.MERGE_HDR = c.get('merge_hdr', False)
    tc.COL_W = tm.COL_W = c.get('col_w', 1.25); tc.TURN_W = tm.TURN_W = tm.COL_W / 2


def _merged_header(ax, vals, n, c):
    """'Stage j' (bold, larger) and 'value v (best b)' (lighter, smaller) on ONE line."""
    K = lo.K
    W = tm.COL_W * c.get("hdr_fill", 0.95)
    sp, vp_ = c.get("stage_pt", 10.0), c.get("value_pt", 6.5)
    gap_pt = c.get("hdr_gap_pt", 4.0)
    texts = [(f"Stage {s + 1}", f"value {vals[s][0]:.0f} (best {vals[s][1]:.0f})") for s in range(n)]
    f = 1.0
    while True:
        wid = max((lo.tw_pt(a, sp * f, "bold") + gap_pt * f + lo.tw_pt(b, vp_ * f)) * tm.UPP for a, b in texts)
        if wid <= W or vp_ * f <= c.get("min_value_pt", 5.0):
            break
        f -= 0.01
    sp_f, vp_f = sp * f, vp_ * f
    tm.HDR_INFO = (sp_f, vp_f, wid / W)
    for s, (a, b) in enumerate(texts):
        wa = lo.tw_pt(a, sp_f, "bold") * tm.UPP
        wb = lo.tw_pt(b, vp_f) * tm.UPP
        gp = gap_pt * f * tm.UPP
        x = tm.col_x(s) + tm.COL_W / 2 - (wa + gp + wb) / 2
        ax.text(x, 0.0, a, ha="left", va="baseline", fontsize=sp_f / K, fontweight="bold", color="#333")
        ax.text(x + wa + gp, 0.0, b, ha="left", va="baseline", fontsize=vp_f / K, color="#666")
    ax.text(tm.start_cx(), 0.0, "Start", ha="center", va="baseline", fontsize=sp_f / K,
            fontweight="bold", color="#777")


def draw_diagram2(ax, rec, posteriors, preds, c):
    n = rec["n_stages"]
    turns = tc.flatten_turns(rec)
    vals = tm.stage_values(rec)
    merge = c.get("merge_hdr", False)
    hp_hdr = (c["head_off"] + c.get("hdr_line", 0.34)) if merge else (c["head_off"] + c["rank_off"] - 0.02)
    hp_y = -c["hp_h"] - hp_hdr
    role_ys, bel_ys = [], []
    yy = hp_y - 0.08
    for _ in range(3):
        yy -= c["top_gap"]
        ry = yy - c["track_h"]
        by = ry - c["sub_gap"] - c["belief_h"]
        role_ys.append(ry); bel_ys.append(by)
        yy = by - c["group_gap"]
    n_before = len(ax.texts)
    tc._draw_stage_headers(ax, n, turns, y_top=(-0.02 if merge else c["title_y"]), y_bottom=bel_ys[-1] + 0.02)
    if merge:
        for t in list(ax.texts)[n_before:]:
            t.remove()
        _merged_header(ax, vals, n, c)
    _hp(ax, rec, turns, hp_y, vals)
    for pid in range(3):
        (_pg3 if c["v3"] else _pg)(ax, rec, turns, posteriors, preds, role_ys[pid], bel_ys[pid], pid)
    x1 = tm.col_x(n - 1) + tm.COL_W
    for i in range(2):
        y_sep = (bel_ys[i] + role_ys[i + 1] + c["track_h"]) / 2 - (c["group_gap"] - 0.16) / 2 * 0
        y_sep = (bel_ys[i] - c["group_gap"] / 2 + role_ys[i + 1] + c["track_h"] + c["top_gap"] / 2) / 1 \
            if False else (bel_ys[i] + role_ys[i + 1] + c["track_h"]) / 2
        ax.plot([0.0, x1], [y_sep, y_sep], color="#bbb", linewidth=0.7,
                linestyle=(0, (4, 3)), zorder=1)


# ---- grouped legend ---------------------------------------------------------
def legend_grouped(ax, upp, fs, ext, K_, intra_pt=6.0, inter_pt=26.0, fill=False):
    L = lo.Legend(ax, upp, fs)
    k = lo.keys(L, "band")
    P = L.P
    k["value"] = L.value_key(["expected return of chosen", "roles (best possible)"])
    G1 = [k["fighter"], k["tank"], k["medic"]]
    G2 = [k["attack"], k["block"], k["heal"]]
    G3 = [k["hp"], k["boss"]]
    GV = [k["value"]]
    G4 = [k["pi"], k["chose"]]
    G5 = [k["pr"], k["carets"]]
    intra = P(intra_pt)
    gw = lambda g: sum(w for w, _, _ in g) + intra * (len(g) - 1)
    x0, x1 = ext["x0"], ext["x1"]
    row1 = [G1, G2, G3, GV]
    row2 = [G4, G5]
    inter = P(inter_pt)               # one shared, clearly larger gap BETWEEN groups
    for nm, rw in (("row1", row1), ("row2", row2)):
        used = sum(gw(g) for g in rw) + inter * (len(rw) - 1)
        print(f"   {nm}: {used / upp:.0f} of {(x1 - x0) / upp:.0f} pt used (intra {intra_pt:.0f}, inter {inter_pt:.0f})")
        if used > x1 - x0:
            print("   !! grouped legend row overflows")
    inter1 = inter
    if fill:   # row 1 spreads its groups to the right edge; row 2 keeps the shared gap
        inter1 = max(inter, (x1 - x0 - sum(gw(g) for g in row1)) / (len(row1) - 1))
        inter = inter1
    h1 = max(L.row_height(g) for g in row1)
    h2 = max(L.row_height(g) for g in row2)
    y = ext["y0"] - P(7)
    ax.plot([x0, x1], [y + P(2.5), y + P(2.5)], color="#ccc", linewidth=0.5)
    y1 = y - P(2) - h1 / 2
    y2 = y1 - h1 / 2 - P(5) - h2 / 2

    def draw_row(groups, yc, gap):
        x = x0
        for g in groups:
            for w, _, draw in g:
                draw(x, yc)
                x += w + intra
            x += gap - intra
    draw_row(row1, y1, inter1)
    draw_row(row2, y2, inter)
    return {"inter_pt": inter / upp, "inter1_pt": inter1 / upp}


# ---- driver -----------------------------------------------------------------
def render_variant(name, cfg, rec, post, preds, fs=5.6):
    c = dict(DEFAULTS); c.update(cfg)
    fs = c.get('leg_fs', fs); INTER = c.get('inter_pt', 26.0)
    set_globals(c)
    xr = 9.0
    PAD_IN = 0.03
    for it in range(8):
        lo.K = max(1.0, min(3.0, (FIG_W / xr) / lo.OLD_IN_PER_UNIT))
        upp = xr / (FIG_W * 72)
        tm.UPP = upp
        fig, ax = lo.new_fig(xr, -2, 3.3)
        draw_diagram2(ax, rec, post, preds, c)
        ext = dict(zip(("x0", "y0", "x1", "y1"), lo.tight_data_bbox(fig, ax)))
        plt.close(fig)
        fig, ax = lo.new_fig(xr, -2, 3.3)
        draw_diagram2(ax, rec, post, preds, c)
        info = legend_grouped(ax, upp, fs, ext, lo.K, inter_pt=INTER, fill=c.get('fill', False))
        bx0, by0, bx1, by1 = lo.tight_data_bbox(fig, ax)
        pad = PAD_IN * xr / FIG_W
        xr_new = (bx1 - bx0) + 2 * pad
        if abs(xr_new - xr) < 0.004 or it == 7:
            ax.set_xlim(bx0 - pad, bx1 + pad); ax.set_ylim(by0 - pad, by1 + pad)
            h = FIG_W * (by1 - by0 + 2 * pad) / (bx1 - bx0 + 2 * pad)
            fig.set_size_inches(FIG_W, h)
            if name:
                fig.savefig(HERE / f"{name}.png", dpi=300, facecolor="white")
                fig.savefig(HERE / f"{name}.pdf", facecolor="white")
            plt.close(fig)
            df = ((ext["x1"] - ext["x0"]) / (bx1 - bx0 + 2 * pad)) * ((ext["y1"] - ext["y0"]) / (by1 - by0 + 2 * pad))
            print(f"   header sizes (pt on page; stage, value, fill): {getattr(tm, 'HDR_INFO', None)}")
            print(f"[{name or 'ablation'}] 7 x {h:.2f} in (aspect {FIG_W / h:.2f}), K={lo.K:.2f}, diagram {df * 100:.0f}% of area, {info}")
            return h, df
        plt.close(fig)
        xr = xr_new


# D2 = current D layout + grouped legend ; D3 = aggressive height cuts
D2 = dict(belief_h=0.45, sub_gap=0.02, hp_h=0.30)
D4 = dict(v3=True, sub_fs=3.9, start_w=1.30, leg_fs=5.6, fill=True, inter_pt=26.0,
          belief_h=0.34, sub_gap=0.02, hp_h=0.22, track_h=0.72, top_gap=0.03,
          group_gap=0.15, head_off=0.155, rank_off=0.17, boss_dx=-0.215, boss_dy=0.055,
          boss_s=0.10, title_y=-0.06, sh_top=0.01, sh_lh=0.09, base_off=0.10, label_dy=0.09)
D5 = dict(D4, merge_hdr=True, boss_dx=0.0, boss_dy=0.185, boss_s=0.115, hdr_line=0.31)
D3 = dict(belief_h=0.40, sub_gap=0.02, hp_h=0.22, track_h=0.72, top_gap=0.03,
          group_gap=0.15, head_off=0.155, rank_off=0.17, boss_dx=-0.215, boss_dy=0.055,
          boss_s=0.10, title_y=-0.06, sh_top=0.01, sh_lh=0.09, base_off=0.10, label_dy=0.09)
LEVERS = {
    "HP bars 0.30->0.22": dict(hp_h=0.22),
    "boss icon inline with 'turn j' + value row closer (header block)": dict(head_off=0.155, rank_off=0.17, boss_dx=-0.215, boss_dy=0.055, boss_s=0.10),
    "stage titles closer (title_y -0.06)": dict(title_y=-0.06),
    "card height 0.76->0.72": dict(track_h=0.72),
    "belief row 0.45->0.38 (+tighter subheader/labels)": dict(belief_h=0.40, sh_top=0.01, sh_lh=0.09, base_off=0.10, label_dy=0.09),
    "row padding (top_gap .05->.03, group_gap .16->.15)": dict(top_gap=0.03, group_gap=0.15),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="2F8H1E:2"); ap.add_argument("--suffix", default="")
    ap.add_argument("--ablate", action="store_true"); ap.add_argument("--only", default="D2,D3")
    a = ap.parse_args()
    s, r = a.case.split(":")
    rec, post, preds = lo.prepare_case(s, int(r))
    if a.ablate:
        base, _ = render_variant(None, D2, rec, post, preds)
        for n, cfg in LEVERS.items():
            h, _ = render_variant(None, {**D2, **cfg}, rec, post, preds)
            print(f"  LEVER {n}: {h:.2f} in (saves {base - h:.2f} in)")
        render_variant(None, D3, rec, post, preds)
    else:
        for n in a.only.split(","):
            render_variant(f"option_{n}{a.suffix}", {"D2": D2, "D3": D3, "D4": D4, "D5": D5}[n], rec, post, preds)
