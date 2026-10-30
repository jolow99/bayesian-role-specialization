"""How flat is the stat-based prior at the fitted tau values?

Fig 1 (prior_vs_tau): P(stat-optimal role) for a 4/1/1 player and the
most-likely-combo probability per human-env stat profile, as tau varies, with
the four fitted tau values marked.
Fig 2 (prior_joint_by_profile): the 27-combo joint prior per profile at
tau = 4.621 (per-person eps_i model), sorted, with the range across the four
fitted tau values as whiskers.

Run from analysis/:
    uv run python experiments/2026-10-30-prior-flatness/plot_prior_flatness.py
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shared.inference import utility_based_prior

OUT_DIR = Path(__file__).resolve().parent / "figures"
PROFILES = ["411_141_114", "411_222_222", "141_222_222", "114_222_222", "222_222_222"]
FITTED_TAUS = {  # model -> tau (Table: inference-model fits)
    "Two types": 4.053,
    "Per-person ε_i": 4.621,
    "Mixed posterior": 4.638,
    "Unmixed P_N": 5.535,
}
REF_TAU = 4.621
ROLE = "FTM"

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def stats_of(profile):
    return np.array([[int(c) for c in g] for g in profile.split("_")])


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def mark_taus(ax):
    lo, hi = min(FITTED_TAUS.values()), max(FITTED_TAUS.values())
    ax.axvspan(lo, hi, color=GRID, alpha=0.6, lw=0)
    ax.text(np.sqrt(lo * hi), ax.get_ylim()[1], "fitted τ\n4.05–5.54",
            ha="center", va="top", fontsize=8, color=INK2)


def fig_vs_tau():
    taus = np.geomspace(0.5, 30, 300)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE)

    ax = axes[0]
    p_opt = [utility_based_prior(stats_of("411_222_222"), t).sum(axis=(1, 2))[0]
             for t in taus]
    p_off = [(1 - p) / 2 for p in p_opt]
    ax.plot(taus, p_opt, color=SERIES[0], lw=2, label="stat-optimal role")
    ax.plot(taus, p_off, color=SERIES[1], lw=2, label="each other role")
    ax.axhline(1 / 3, color=INK2, lw=1, ls="--")
    ax.text(0.55, 1 / 3 + 0.01, "uniform 1/3", ha="left", va="bottom", fontsize=8, color=INK2)
    ax.set_ylim(0, 1)
    ax.set_xscale("log")
    mark_taus(ax)
    for name, t in FITTED_TAUS.items():
        p = utility_based_prior(stats_of("411_222_222"), t).sum(axis=(1, 2))[0]
        ax.plot(t, p, "o", ms=6, color=SERIES[0], mec=SURFACE, mew=1.5)
    ax.annotate(f"{utility_based_prior(stats_of('411_222_222'), 4.053).sum(axis=(1, 2))[0]:.2f}",
                (4.053, 0.51), xytext=(-18, 8), textcoords="offset points", fontsize=8, color=INK)
    ax.annotate(f"{utility_based_prior(stats_of('411_222_222'), 5.535).sum(axis=(1, 2))[0]:.2f}",
                (5.535, 0.46), xytext=(6, 8), textcoords="offset points", fontsize=8, color=INK)
    style(ax)
    ax.set_xlabel("τ (prior temperature)", color=INK)
    ax.set_ylabel("prior probability", color=INK)
    ax.set_title("Marginal prior for a player with stats 4/1/1", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1, 0.88))

    ax = axes[1]
    groups = [("411_141_114", "411_141_114 (three specialists)", SERIES[0]),
              ("411_222_222", "one specialist + two 2/2/2", SERIES[1]),
              ("222_222_222", "222_222_222 (all 2/2/2) = uniform 1/27", SERIES[2])]
    for prof, label, c in groups:
        ax.plot(taus, [utility_based_prior(stats_of(prof), t).max() for t in taus],
                color=c, lw=2, label=label)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim(0.03, 1.05)
    mark_taus(ax)
    style(ax)
    ax.set_xlabel("τ (prior temperature)", color=INK)
    ax.set_ylabel("P(most likely role combo)", color=INK)
    ax.set_title("Joint prior: probability of the top combo (27 combos)",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1, 0.88))

    fig.tight_layout()
    return fig


def fig_joint_by_profile():
    combos = ["".join(ROLE[r] for r in c) for c in product(range(3), repeat=3)]
    fig, axes = plt.subplots(1, len(PROFILES), figsize=(16, 4.2), sharey=True,
                             facecolor=SURFACE)
    for ax, prof in zip(axes, PROFILES):
        s = stats_of(prof)
        ref = utility_based_prior(s, REF_TAU).ravel()
        all_t = np.array([utility_based_prior(s, t).ravel() for t in FITTED_TAUS.values()])
        order = np.argsort(-ref, kind="stable")
        x = np.arange(27)
        ax.bar(x, ref[order], width=0.8, color=SERIES[0], edgecolor=SURFACE, lw=0.5)
        ax.vlines(x, all_t.min(0)[order], all_t.max(0)[order], color=INK, lw=1)
        ax.axhline(1 / 27, color=INK2, lw=1, ls="--")
        ax.text(26.4, 1 / 27, "uniform", ha="right", va="bottom", fontsize=8, color=INK2)
        n_top = int(np.isclose(ref, ref.max()).sum())
        top = combos[order[0]] if n_top == 1 else f"{n_top} combos tied"
        if n_top == 27:
            ax.text(0, ref.max() + 0.004, "exactly uniform: 1/27 = 0.037",
                    ha="left", va="bottom", fontsize=8, color=INK)
        else:
            ax.text(n_top - 0.5, ref.max(), f" {top}: {ref.max():.3f}",
                    ha="left", va="bottom", fontsize=8, color=INK)
        ax.set_xticks([])
        ax.set_xlim(-0.8, 26.8)
        style(ax)
        ax.set_title(prof, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("27 role combos, sorted", color=INK2, fontsize=9)
    axes[0].set_ylabel("joint prior probability", color=INK)
    axes[0].set_ylim(0, 0.16)
    fig.suptitle(f"Stat-based joint prior per human-env profile at τ = {REF_TAU} "
                 "(whiskers: range over τ = 4.05–5.54)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    return fig


def main():
    OUT_DIR.mkdir(exist_ok=True)
    for name, fig in [("prior_vs_tau", fig_vs_tau()),
                      ("prior_joint_by_profile", fig_joint_by_profile())]:
        for ext in ("png", "pdf"):
            fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=200, facecolor=SURFACE)
        print(f"Saved {OUT_DIR / name}.png/.pdf")


if __name__ == "__main__":
    main()
