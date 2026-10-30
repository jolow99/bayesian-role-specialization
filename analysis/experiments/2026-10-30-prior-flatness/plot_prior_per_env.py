"""Stat-based joint prior over the 27 role combos, per human env, across tau.

Fig 3 (prior_per_env): one panel per human env (5 profiles x 3 envs). x = the
27 combos in lexicographic order (FFF..MMM, the values.npy order); one line
per fitted model tau (inference-model fits table) plus tau = 1 as a
sharp-prior reference (clipped at YMAX, peak value annotated); dashed line = uniform 1/27; shaded column = the env's optimal combo.
The prior depends only on player stats, so panels in a row share the curves.
Fig 4 (prior_distance_vs_tau): total-variation distance to uniform and
normalized entropy H / log 27 vs tau, per profile.

Run from analysis/:
    uv run python experiments/2026-10-30-prior-flatness/plot_prior_per_env.py
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shared import HUMAN_ENVS_DIR
from shared.env_loading import load_env_config
from shared.inference import utility_based_prior

OUT_DIR = Path(__file__).resolve().parent / "figures"
PROFILES = ["411_141_114", "411_222_222", "141_222_222", "114_222_222", "222_222_222"]
COMBOS = ["".join("FTM"[r] for r in c) for c in product(range(3), repeat=3)]
TAUS = {  # tau -> model (Table: inference-model fits), ascending
    4.053: "Two types",
    4.621: "Per-person ε_i, Beta prior",
    4.638: "Mixed posterior",
    5.535: "Unmixed P_N",
}
FITTED_TAUS = list(TAUS)
REF_TAU = 1.0  # reference: what a sharp stat prior would look like
YMAX = 0.16

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4
MODEL_MARKERS = ["o", "s", "^", "D"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def human_envs():
    envs = sorted(p.name for p in HUMAN_ENVS_DIR.iterdir() if "__" in p.name)
    by_prof = {prof: [e for e in envs if e.startswith(prof)] for prof in PROFILES}
    stats = {e: np.asarray(load_env_config(HUMAN_ENVS_DIR / e)["player_stats"], dtype=int)
             for e in envs}
    return by_prof, stats


def tv_to_uniform(p):
    return 0.5 * np.abs(p - 1 / 27).sum()


def norm_entropy(p):
    return float(-(p * np.log(p)).sum() / np.log(27))


def fig_per_env(by_prof, stats):
    fig, axes = plt.subplots(len(PROFILES), 3, figsize=(17, 17), sharex=True, sharey=True,
                             facecolor=SURFACE)
    x = np.arange(27)
    for row, prof in zip(axes, PROFILES):
        for ax, env in zip(row, by_prof[prof]):
            opt = env.split("__")[1]
            ax.axvspan(COMBOS.index(opt) - 0.45, COMBOS.index(opt) + 0.45, color=GRID, lw=0)
            for (tau, label), c, m in zip(TAUS.items(), MODEL_COLORS, MODEL_MARKERS):
                p = utility_based_prior(stats[env], tau).ravel()
                ax.plot(x, p, "-", marker=m, color=c, lw=1.5, ms=4.5, mec=SURFACE, mew=0.6,
                        label=f"τ = {tau:.3f} ({label})")
            p_ref = utility_based_prior(stats[env], REF_TAU).ravel()
            ax.plot(x, np.minimum(p_ref, YMAX * 0.985), "-", marker="x", color=INK, lw=1,
                    ms=4, mew=1, label=f"τ = {REF_TAU:g} (reference)", zorder=1)
            for i in np.flatnonzero(p_ref > YMAX):
                ax.annotate(f"τ=1: {p_ref[i]:.2f} ↑", (i, YMAX * 0.985), xytext=(6, -4),
                            textcoords="offset points", ha="left", va="top",
                            fontsize=8, color=INK)
            ax.axhline(1 / 27, color=INK2, lw=1, ls="--")
            p_opt = [utility_based_prior(stats[env], t).ravel()[COMBOS.index(opt)]
                     for t in FITTED_TAUS]
            p_ref_opt = p_ref[COMBOS.index(opt)]
            ax.set_title(f"{env}\nP(optimal {opt}): fitted τ {min(p_opt):.3f}–{max(p_opt):.3f}"
                         f" ({27 * min(p_opt):.2f}–{27 * max(p_opt):.2f}× uniform);"
                         f"  τ=1 {p_ref_opt:.3f} ({27 * p_ref_opt:.2f}×)",
                         loc="left", fontsize=9, color=INK)
            style(ax)
    for ax in axes[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels(COMBOS, rotation=90, fontsize=7, family="monospace")
    for ax in axes[:, 0]:
        ax.set_ylabel("joint prior", color=INK)
    axes[0, 0].set_ylim(0, YMAX)
    axes[-1, -1].text(26.5, 1 / 27 * 1.15, "uniform 1/27", ha="right", va="bottom",
                      fontsize=8, color=INK2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.005, 0.985))
    fig.suptitle("Stat-based joint prior over 27 role combos, per human env "
                 "(shaded column = env's optimal combo; rows share stats, so curves repeat)",
                 x=0.01, ha="left", y=0.995, fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return fig


def fig_distance_vs_tau(stats_by_prof):
    taus = np.geomspace(0.5, 50, 300)
    groups = [("411_141_114", "411_141_114 (three specialists)", SERIES[0]),
              ("411_222_222", "411 / 141 / 114 _222_222 (one specialist; identical)", SERIES[1]),
              ("222_222_222", "222_222_222 (exactly uniform at any τ)", SERIES[2])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE)
    for ax, fn, ylab, title in [
        (axes[0], tv_to_uniform, "TV distance to uniform",
         "Total-variation distance to uniform (0 = uniform)"),
        (axes[1], norm_entropy, "H(prior) / log 27",
         "Normalized entropy (1 = uniform)"),
    ]:
        for prof, label, c in groups:
            s = stats_by_prof[prof]
            ax.plot(taus, [fn(utility_based_prior(s, t).ravel()) for t in taus],
                    color=c, lw=2, label=label)
            for t in FITTED_TAUS:
                ax.plot(t, fn(utility_based_prior(s, t).ravel()), "o", ms=7, color=c,
                        mec=SURFACE, mew=1.5)
            ax.plot(REF_TAU, fn(utility_based_prior(s, REF_TAU).ravel()), "x", ms=8,
                    mew=1.8, color=c)
        ax.axvspan(min(FITTED_TAUS), max(FITTED_TAUS), color=GRID, alpha=0.6, lw=0)
        ax.axvline(REF_TAU, color=INK2, lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        style(ax)
        ax.set_xlabel("τ (prior temperature)", color=INK)
        ax.set_ylabel(ylab, color=INK)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
    axes[0].text(np.sqrt(min(FITTED_TAUS) * max(FITTED_TAUS)), 0.98, "fitted τ\n4.05–5.54",
                 ha="center", va="top", fontsize=8, color=INK2)
    axes[0].text(REF_TAU * 1.05, 0.98, "τ = 1\n(reference)", ha="left", va="top",
                 fontsize=8, color=INK2)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def print_table(by_prof, stats):
    print(f"{'env':<20} {'tau':>6} {'TV':>6} {'H/log27':>8} {'max p':>7} "
          f"{'p(opt)':>7} {'p(opt)*27':>9} {'argmax':>7}")
    for prof in PROFILES:
        for env in by_prof[prof]:
            opt = env.split("__")[1]
            for tau in [REF_TAU, *FITTED_TAUS]:
                p = utility_based_prior(stats[env], tau).ravel()
                top = np.flatnonzero(np.isclose(p, p.max()))
                arg = COMBOS[top[0]] if len(top) == 1 else f"{len(top)} tied"
                po = p[COMBOS.index(opt)]
                print(f"{env:<20} {tau:6.2f} {tv_to_uniform(p):6.3f} {norm_entropy(p):8.3f} "
                      f"{p.max():7.3f} {po:7.3f} {27 * po:9.2f} {arg:>7}")
            print()


def main():
    OUT_DIR.mkdir(exist_ok=True)
    by_prof, stats = human_envs()
    print_table(by_prof, stats)
    stats_by_prof = {prof: stats[by_prof[prof][0]] for prof in PROFILES}
    for name, fig in [("prior_per_env", fig_per_env(by_prof, stats)),
                      ("prior_distance_vs_tau", fig_distance_vs_tau(stats_by_prof))]:
        for ext in ("png", "pdf"):
            fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=200, facecolor=SURFACE)
        print(f"Saved {OUT_DIR / name}.png/.pdf")


if __name__ == "__main__":
    main()
