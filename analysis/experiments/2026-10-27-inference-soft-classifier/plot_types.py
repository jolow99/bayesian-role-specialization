"""Per-participant type posterior from the soft-classifier fit (R4-style plot).

One stacked bar per participant: P(Bayesian | answers) and P(prior-default |
answers) at the best fit (θ is the prior over types), read from
participant_types.csv written by fit.py. Styled like the R4 individual-fitting
figure (2026-10-27-top-k-fixed-5/top5.py).

Run from analysis/ after fit.py:
    uv run python experiments/2026-10-27-inference-soft-classifier/plot_types.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

TYPES = ("Bayesian", "Prior-default")
COLORS = {"Bayesian": "#08519c", "Prior-default": "#fb9a29"}


def main():
    best = json.load(open(SCRIPT_DIR / "best_params.json"))
    rows = list(csv.DictReader(open(SCRIPT_DIR / "participant_types.csv")))
    p_prior = np.array(sorted(float(r["p_prior_type"]) for r in rows))
    post = {"Bayesian": 1.0 - p_prior, "Prior-default": p_prior}
    n_b = int((p_prior < 0.1).sum())
    n_d = int((p_prior > 0.9).sum())

    rc = {"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
          "font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 6,
          "ytick.labelsize": 6, "axes.linewidth": 0.6}
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(3.42, 2.4))
        x = np.arange(len(p_prior)); bottoms = np.zeros(len(p_prior))
        for t in TYPES:
            ax.bar(x, post[t], bottom=bottoms, width=1.0, color=COLORS[t],
                   edgecolor="white", linewidth=0.35)
            bottoms += post[t]
        assert np.allclose(bottoms, 1.0, atol=1e-9)
        ax.axhline(1.0 - best["theta"], color="black", lw=0.6, ls="--")
        ax.text(len(p_prior) - 0.5, 1.0 - best["theta"] + 0.015,
                f"population prior 1 − θ = {1.0 - best['theta']:.3f}",
                ha="right", va="bottom", fontsize=5.5)
        ax.set_xlim(-0.5, len(p_prior) - 0.5); ax.set_ylim(0, 1)
        ax.set_xlabel(f"Participants (n = {len(p_prior)}, sorted by P(Bayesian))")
        ax.set_ylabel("P(type | participant's inferences)")
        ax.set_xticks([]); ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.spines[["top", "right"]].set_visible(False)
        proxy = lambda t: mpatches.Patch(facecolor=COLORS[t], edgecolor="white", linewidth=0.35)
        labels = [f"Bayesian (P > 0.9: {n_b})", f"Prior-default (P > 0.9: {n_d})"]
        fig.tight_layout(pad=0.3, rect=(0, 0.1, 1, 1))
        fig.legend([proxy(t) for t in TYPES], labels, loc="upper center",
                   bbox_to_anchor=(0.5, 0.1), ncols=2, fontsize=5.5, frameon=False,
                   handlelength=1.2, handleheight=1.1, columnspacing=1.5)
        for ext, k in (("png", {"dpi": 300}), ("pdf", {})):
            fig.savefig(FIG_DIR / f"inference_type_posterior.{ext}", bbox_inches="tight",
                        facecolor="white", **k)
        plt.close(fig)

    print(f"Bayesian (P > 0.9): {n_b}, prior-default (P > 0.9): {n_d}, "
          f"uncertain: {len(p_prior) - n_b - n_d}")
    for lo, hi in ((0.0, 0.01), (0.01, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 1.0001)):
        print(f"  P(prior-default) in [{lo:.2f}, {min(hi, 1):.2f}): {int(((p_prior >= lo) & (p_prior < hi)).sum())}")


if __name__ == "__main__":
    main()
