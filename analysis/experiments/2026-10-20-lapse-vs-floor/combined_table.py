"""Single aggregate-table figure: agg-LL and mean-LL under none / lapse / floor, side by side.
Reads results.json (written by lapse_vs_floor.py); no recomputation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(__file__).resolve().parent
R = json.load(open(D / "results.json"))
ORDER = ["Bayesian Walk", "Bayesian Walk-PS", "Bayesian-Belief", "Bayesian-Value", "Bayesian Threshold",
         "Bayesian Thresh-PS", "Random Walk", "Top-7", "Random-to-Optimal", "Optimal", "Copy Others",
         "Contradict Others", "Random", "Always Stick"]
NAMES = {"Bayesian Walk": "Bayesian-Walk-BR", "Bayesian Walk-PS": "Bayesian-Walk-PS", "Bayesian-Belief": "Bayesian-PS",
         "Bayesian-Value": "Bayesian-BR", "Bayesian Threshold": "Bayesian-Threshold-BR",
         "Bayesian Thresh-PS": "Bayesian-Threshold-PS"}
V = ["none", "lapse", "floor"]
fmt = lambda v: "−∞" if v is None or v == float("-inf") or v != v else f"{v:.3f}".replace("-", "−")
# json stores -inf as -Infinity (parsed to float('-inf'))
head = ["Model", "combo-r", "marg-r", "none", "lapse", "floor", "none", "lapse", "floor"]
rows = []
for m in ORDER:
    x = {v: R[v]["metrics"][m] for v in V}
    rows.append([NAMES.get(m, m), f"{x['lapse']['combo_r']:.3f}".replace("-", "−"), f"{x['lapse']['marg_r']:.3f}".replace("-", "−")]
                + [fmt(x[v]["agg_ll"]) for v in V] + [fmt(x[v]["mean_ll"]) for v in V])

fig, ax = plt.subplots(figsize=(9.2, 5.6)); ax.axis("off")
t = ax.table(cellText=rows, colLabels=head, cellLoc="right", colLoc="right", loc="center",
             colWidths=[0.27, 0.09, 0.09] + [0.09] * 6)
t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.3)
for (r, c), cell in t.get_celld().items():
    cell.set_linewidth(0.3)
    if c == 0:
        cell.set_text_props(ha="left"); cell._loc = "left"
    if r > 0 and c >= 3 and cell.get_text().get_text() == "−∞":
        cell.set_text_props(color="#b2182b")
    if r > 0 and c in (3, 6):
        cell.set_edgecolor("black")
# group headers centred over the middle (lapse) column of each LL block, from real cell extents
fig.canvas.draw()
inv = ax.transAxes.inverted()
for col, lab in ((4, "agg-LL"), (7, "mean-LL")):
    bb = t.get_celld()[(0, col)].get_window_extent(fig.canvas.get_renderer())
    xc, ytop = inv.transform(((bb.x0 + bb.x1) / 2, bb.y1))
    ax.text(xc, ytop + 0.02, lab, ha="center", transform=ax.transAxes, fontsize=9, fontweight="bold")
d = {v: R[v]["delta"] for v in V}
ax.text(0.0, 0.03, "none = no lapse/floor (−∞ if any observed choice has p = 0)   lapse = (1−λ)p + λ/K, λ = 0.05   floor = p clipped at 1e-12\n"
        f"Bayesian-Threshold-PS δ: none {d['none']:.2f} (combo-r fit; agg-LL undefined), lapse {d['lapse']:.2f}, floor {d['floor']:.3f}. "
        "combo-r / marg-r are shown for the lapse variant; they differ by ≤ 0.007 across variants except Threshold-PS (combo-r 0.355 under none, δ = 0.26).",
        transform=ax.transAxes, fontsize=6, va="top")
fig.savefig(D / "figures" / "aggregate_table_combined.png", dpi=250, bbox_inches="tight", facecolor="white")
fig.savefig(D / "figures" / "aggregate_table_combined.pdf", bbox_inches="tight", facecolor="white")
