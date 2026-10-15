"""All figure text >= 6 pt AS PRINTED (PNAS).  D5 layout, text sized in page points.

printed scale s = printed width / 7.0 in native:
  main figure at 0.9 linewidth: 0.9*512pt = 6.40 in -> s = 0.914
  main figure at 1.0 linewidth / SI figures (linewidth 512 pt = 7.11 in) -> s = 1.016
Every text artist must satisfy fontsize * s >= TARGET (6.05 pt, 0.05 margin).
"""
from __future__ import annotations
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.text
import grouped_variants as gv
import legend_options as lo
from legend_options import tm, tc, HERE, FIG_W

TARGET = 6.05


def _center_line(ax, y, parts, fs):
    cx = tm.start_cx()
    ws = [lo.tw_pt(t, fs * lo.K, "bold" if b else "normal") * tm.UPP for t, b in parts]
    gap = 0.45 * fs * lo.K * tm.UPP
    tot = sum(ws) + gap * (len(parts) - 1)
    x = cx - tot / 2
    for (t, b), w in zip(parts, ws):
        ax.text(x, y, t, ha="left", va="center", fontsize=fs, color=tm.SUBHEAD_COLOR if b else "#444",
                fontweight="bold" if b else "normal")
        x += w + gap


tm._center_line = _center_line
MIN_NOMINAL = 4.0           # smallest hard-coded diagram font in the 09-16 code (bar numbers, Pr labels)
NUMFS = [4.0]
_orig_bk = lo.Legend.bars_key


def _bk(self, *a, **kw):
    kw.setdefault("num_fs", NUMFS[0])
    return _orig_bk(self, *a, **kw)


lo.Legend.bars_key = _bk


def header_plan(texts, p, W, gap_pt=4.0):
    """one line if 'Stage j' (>= p+1.5 pt, <=10 pt) + value (p pt) fits, else two lines."""
    for sp in (10.0, 9.5, 9.0, 8.5, 8.0):
        if sp < p + 1.0:
            break
        wid = max((lo.tw_pt(a, sp, "bold") + gap_pt + lo.tw_pt(b, p)) * tm.UPP for a, b in texts)
        if wid <= W:
            return ("one", sp, wid / W)
    return ("two", 9.0, None)


def draw_header(ax, vals, n, plan, p):
    K = lo.K
    texts = [(f"Stage {s + 1}", f"value {vals[s][0]:.0f} (best {vals[s][1]:.0f})") for s in range(n)]
    mode, sp, _ = plan
    gap = 4.0 * tm.UPP
    for s, (a, b) in enumerate(texts):
        cx = tm.col_x(s) + tm.COL_W / 2
        if mode == "one":
            wa = lo.tw_pt(a, sp, "bold") * tm.UPP
            wb = lo.tw_pt(b, p) * tm.UPP
            x = cx - (wa + gap + wb) / 2
            ax.text(x, 0.0, a, ha="left", va="baseline", fontsize=sp / K, fontweight="bold", color="#333")
            ax.text(x + wa + gap, 0.0, b, ha="left", va="baseline", fontsize=p / K, color="#666")
        else:
            ax.text(cx, 0.0, b, ha="center", va="baseline", fontsize=p / K, color="#666")
            ax.text(cx, 0.17, a, ha="center", va="baseline", fontsize=sp / K, fontweight="bold", color="#333")
    ax.text(tm.start_cx(), 0.17 if mode == "two" else 0.0, "Start", ha="center", va="baseline",
            fontsize=sp / K, fontweight="bold", color="#777")


def draw_diagram3(ax, rec, post, preds, c, plan, p):
    n = rec["n_stages"]
    turns = tc.flatten_turns(rec)
    vals = tm.stage_values(rec)
    extra = 0.17 if plan[0] == "two" else 0.0
    hp_y = -c["hp_h"] - (c["head_off"] + c["hdr_line"])
    role_ys, bel_ys = [], []
    yy = hp_y - 0.08
    for _ in range(3):
        yy -= c["top_gap"]
        ry = yy - c["track_h"]
        by = ry - c["sub_gap"] - c["belief_h"]
        role_ys.append(ry); bel_ys.append(by)
        yy = by - c["group_gap"]
    n0 = len(ax.texts)
    tc._draw_stage_headers(ax, n, turns, y_top=-0.02 + extra, y_bottom=bel_ys[-1] + 0.02)
    for t in list(ax.texts)[n0:]:
        t.remove()
    draw_header(ax, vals, n, plan, p)
    gv._hp(ax, rec, turns, hp_y, vals)
    for pid in range(3):
        gv._pg3(ax, rec, turns, post, preds, role_ys[pid], bel_ys[pid], pid)
    x1 = tm.col_x(n - 1) + tm.COL_W
    for i in range(2):
        y_sep = (bel_ys[i] + role_ys[i + 1] + c["track_h"]) / 2
        ax.plot([0.0, x1], [y_sep, y_sep], color="#bbb", linewidth=0.7, linestyle=(0, (4, 3)), zorder=1)


def render6(name, rec, post, preds, s_print, case_label=""):
    c = dict(gv.DEFAULTS); c.update(gv.D5)
    c["v3"] = True
    p = TARGET / s_print                  # minimum text size in native page pt
    gv.set_globals(c)
    tm.MERGE_HDR = True
    xr = 9.0
    PAD_IN = 0.03
    for it in range(12):
        nat = (FIG_W / xr) / lo.OLD_IN_PER_UNIT
        lo.K = max(nat, p / MIN_NOMINAL)
        upp = xr / (FIG_W * 72)
        tm.UPP = upp
        NUMFS[0] = p / lo.K
        # Start column wide enough for the one-line subheaders at p pt
        need = max(lo.tw_pt("Model beliefs & player inferences", p, "bold"),
                   lo.tw_pt("Model prediction", p, "bold") + lo.tw_pt("π(P1 | Model)", p) + 0.45 * p) * upp
        sw = max(1.30, need / 0.95)
        tc.START_W = tm.START_W = sw
        tm.SUB_FS = p / lo.K
        tm.BEL_TEXT = "Model beliefs & player inferences"
        texts = [(f"Stage {s + 1}", "value 88 (best 88)") for s in range(rec["n_stages"])]
        plan = header_plan(texts, p, tm.COL_W * 0.95)
        fig, ax = lo.new_fig(xr, -2, 3.3)
        draw_diagram3(ax, rec, post, preds, c, plan, p)
        ext = dict(zip(("x0", "y0", "x1", "y1"), lo.tight_data_bbox(fig, ax)))
        plt.close(fig)
        fig, ax = lo.new_fig(xr, -2, 3.3)
        draw_diagram3(ax, rec, post, preds, c, plan, p)
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            info = gv.legend_grouped(ax, upp, p, ext, lo.K, inter_pt=26.0, fill=True)
        overflow = 'overflows' in buf.getvalue()
        bx0, by0, bx1, by1 = lo.tight_data_bbox(fig, ax)
        pad = PAD_IN * xr / FIG_W
        xr_new = (bx1 - bx0) + 2 * pad
        if abs(xr_new - xr) < 0.004 or it == 11:
            ax.set_xlim(bx0 - pad, bx1 + pad); ax.set_ylim(by0 - pad, by1 + pad)
            h = FIG_W * (by1 - by0 + 2 * pad) / (bx1 - bx0 + 2 * pad)
            fig.set_size_inches(FIG_W, h)
            fig.canvas.draw()
            # ---- verify: every visible, non-empty Text artist
            mn = (1e9, None)
            for t in fig.findobj(matplotlib.text.Text):
                if not t.get_visible() or not t.get_text().strip():
                    continue
                v = t.get_fontsize()
                if v < mn[0]:
                    mn = (v, t.get_text())
            fig.savefig(HERE / f"{name}.png", dpi=300, facecolor="white")
            fig.savefig(HERE / f"{name}.pdf", facecolor="white")
            plt.close(fig)
            res = dict(name=name, native_w=FIG_W, native_h=h, s=s_print, printed_w=FIG_W * s_print,
                       printed_h=h * s_print, min_native=mn[0], min_printed=mn[0] * s_print,
                       min_text=mn[1], header=plan, legend_overflow=overflow, K=lo.K, start_w=sw)
            print(f"[{name}] native 7 x {h:.2f} in; printed {FIG_W * s_print:.2f} x {h * s_print:.2f} in; "
                  f"min text {mn[0]:.2f} pt native = {mn[0] * s_print:.2f} pt printed ({mn[1]!r}); "
                  f"legend overflow={overflow}; header {plan}; K={lo.K:.2f}; start_w={sw:.2f}")
            return res
        plt.close(fig)
        xr = xr_new


CASES = [("2F8H1E", 2, "R3_team_case"), ("11B0J4", 6, "R3_team_case_11B0J4_r6"),
         ("11B0J4", 8, "R3_team_case_11B0J4_r8"), ("K72DV0", 5, "R3_team_case_K72DV0_r5"),
         ("RBRN0Z", 8, "R3_team_case_RBRN0Z_r8")]
S09 = 0.9 * 512 / 72 / 7.0     # main fig at 0.9 linewidth
S10 = 512 / 72 / 7.0           # \linewidth (SI, or main at 1.0)

if __name__ == "__main__":
    import json
    out = []
    for s, r, nm in CASES:
        rec, post, preds = lo.prepare_case(s, r)
        if nm == "R3_team_case":
            out.append(render6(f"final6pt/{nm}", rec, post, preds, S09))
            out.append(render6(f"final6pt/main_at_1.0/{nm}", rec, post, preds, S10))
        else:
            out.append(render6(f"final6pt/{nm}", rec, post, preds, S10))
    json.dump(out, open(HERE / "final6pt" / "results.json", "w"), default=str, indent=1)
