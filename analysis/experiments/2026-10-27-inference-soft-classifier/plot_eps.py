"""Pooled-epsilon fit figure: population Beta prior and per-participant ε_i posteriors.

A. Fitted population distribution over the 21 ε nodes (bin masses of
   Beta(a, b) on ε' = 1.5·ε); the ε = 2/3 node (no updating) in orange.
B. One column per participant (sorted by posterior mean ε): P(ε_i = ε_k |
   answers) as a heatmap, with the two-type model's P(prior-default) as a
   strip above for comparison.

Run from analysis/ after fit.py and fit_eps_pooled.py:
    uv run python experiments/2026-10-27-inference-soft-classifier/plot_eps.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

BLUE, ORANGE = "#08519c", "#fb9a29"


def main():
    best = json.load(open(SCRIPT_DIR / "best_params_eps_pooled.json"))
    eps = np.array(best["eps_nodes"])
    w = np.array(best["population_weights"])
    rows = list(csv.DictReader(open(SCRIPT_DIR / "participant_eps.csv")))
    node_cols = [c for c in rows[0] if c.startswith("p_eps_")]
    post = np.array([[float(r[c]) for c in node_cols] for r in rows])
    mean_eps = np.array([float(r["post_mean_eps"]) for r in rows])
    two_type = {r["participant_id"]: float(r["p_prior_type"])
                for r in csv.DictReader(open(SCRIPT_DIR / "participant_types.csv"))}
    p_prior = np.array([two_type[r["participant_id"]] for r in rows])
    order = np.argsort(mean_eps)

    rc = {"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
          "font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 6,
          "ytick.labelsize": 6, "axes.linewidth": 0.6}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(7.0, 2.9))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 2.3], height_ratios=[0.07, 1],
                              hspace=0.08, wspace=0.42)
        ax_a = fig.add_subplot(gs[:, 0])
        cols = [ORANGE if i == len(eps) - 1 else BLUE for i in range(len(eps))]
        ax_a.bar(eps, w, width=(2 / 3) / 20 * 0.85, color=cols)
        ax_a.set_xlim(-0.03, 0.70)
        ax_a.set_xticks([0, 1 / 3, 2 / 3]); ax_a.set_xticklabels(["0", "1/3", "2/3\n(no updating)"])
        ax_a.set_xlabel("ε (action noise)")
        ax_a.set_ylabel("Share of population")
        ax_a.set_title(f"A. Fitted prior: Beta(a = {best['a']:.2f}, b = {best['b']:.2f})",
                       loc="left", fontsize=7.5)
        ax_a.spines[["top", "right"]].set_visible(False)

        ax_s = fig.add_subplot(gs[0, 1])
        ax_s.imshow(p_prior[order][None, :], aspect="auto", cmap="Oranges", vmin=0, vmax=1,
                    interpolation="nearest")
        ax_s.set_xticks([]); ax_s.set_yticks([0])
        ax_s.set_yticklabels(["Two-type model:\nP(uses role-based prior)"], fontsize=5.5)
        ax_s.set_title("B. Per-participant posterior over ε_i", loc="left", fontsize=7.5)

        ax_b = fig.add_subplot(gs[1, 1])
        im = ax_b.imshow(post[order].T, aspect="auto", origin="lower", cmap="Blues",
                         vmin=0, vmax=1, interpolation="nearest",
                         extent=(-0.5, len(rows) - 0.5, -0.5, len(eps) - 0.5))
        ax_b.plot(np.arange(len(rows)), mean_eps[order] / (2 / 3) * (len(eps) - 1),
                  color="black", lw=0.8, label="posterior mean ε")
        ax_b.set_yticks([0, 10, 20]); ax_b.set_yticklabels(["0", "1/3", "2/3"])
        ax_b.set_ylabel("ε")
        ax_b.set_xticks([])
        ax_b.set_xlabel(f"Participants (n = {len(rows)}, sorted by posterior mean ε)")
        ax_b.legend(loc="upper left", fontsize=5.5, frameon=False)
        cb = fig.colorbar(im, ax=[ax_s, ax_b], fraction=0.025, pad=0.01)
        cb.set_label("P(ε_i | answers)", fontsize=6); cb.ax.tick_params(labelsize=5.5)
        for ext, k in (("png", {"dpi": 300}), ("pdf", {})):
            fig.savefig(FIG_DIR / f"eps_pooled_posterior.{ext}", bbox_inches="tight",
                        facecolor="white", **k)
        plt.close(fig)

    hi = post[:, -1]
    print(f"P(ε = 2/3) > 0.9: {int((hi > 0.9).sum())}, posterior mean ε < 0.2: "
          f"{int((mean_eps < 0.2).sum())}, 0.2-0.5: {int(((mean_eps >= 0.2) & (mean_eps < 0.5)).sum())}, "
          f">= 0.5: {int((mean_eps >= 0.5).sum())}")
    print(f"corr(two-type P(prior), posterior mean ε) = {np.corrcoef(p_prior, mean_eps)[0, 1]:.3f}")


if __name__ == "__main__":
    main()
