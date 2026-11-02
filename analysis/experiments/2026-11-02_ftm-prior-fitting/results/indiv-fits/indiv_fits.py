"""Fig. 5: per-participant model posteriors from the saved response fits.

Main text: updaters vs all 13 models; non-updaters vs the baselines plus Prior
Sampling (Bayesian-PS on their FTM belief). Appendix: non-updaters vs all 13 plus
Prior Sampling. Per participant: softmax of their log-likelihoods over the panel's
models (uniform prior). A model whose per-participant log-likelihoods match another's
to < 1e-3 is dropped, keeping the baseline when a Bayesian model collapses onto one.
Bayesian-Threshold-BR is omitted from the main figure: at its fitted threshold it reduces to
Bayesian-BR (as in the paper), though its τ_v differs slightly, so the 1e-3 check misses it. Plot code from 2026-10-30 individual_fits.py.

Run from analysis/:
    uv run python experiments/2026-11-02_ftm-prior-fitting/results/indiv-fits/indiv_fits.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
FIT_DIR = OUT_DIR.parents[1] / "response-fitting" / "fit-results"

BAYES = ["Bayesian-Walk-BR", "Bayesian-Walk-PS", "Bayesian-PS", "Bayesian-BR",
         "Bayesian-Threshold-BR", "Bayesian-Threshold-PS"]
BASE = ["Random Walk", "Top-5", "Random-to-Optimal", "Optimal", "Copy Others",
        "Contradict Others", "Prior Sampling", "Random"]
ORDER = BAYES + BASE  # sort and stacking order
PALETTE = {
    "Bayesian-Walk-BR": "#08519c", "Bayesian-Walk-PS": "#9ecae1", "Bayesian-PS": "#41ab5d",
    "Bayesian-BR": "#1c9099", "Bayesian-Threshold-BR": "#6a51a3",
    "Bayesian-Threshold-PS": "#bcbddc", "Random Walk": "#e6550d", "Top-5": "#c51b7d",
    "Random-to-Optimal": "#8c510a", "Optimal": "#b2182b", "Copy Others": "#fff7bc",
    "Contradict Others": "#fb9a29", "Prior Sampling": "#a6d96a", "Random": "#969696",
}

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "axes.linewidth": 0.6})


def load_lls(group):
    """{model: {participant: LL}} for both families of one group's fit."""
    lls = {}
    for fam in ("bayes-models", "non-bayes-models"):
        res = json.load(open(FIT_DIR / group / f"{fam}.json"))
        lls.update({m: r["participant_ll"] for m, r in res.items()})
    return lls


def drop_collapsed(lls):
    keep = []
    for m in [m for m in BASE + BAYES if m in lls]:
        if not any(all(abs(lls[m][p] - lls[k][p]) < 1e-3 for p in lls[m]) for k in keep):
            keep.append(m)
    return [m for m in ORDER if m in keep]


def posteriors(lls, models):
    out = {}
    for p in lls[models[0]]:
        v = np.array([lls[m][p] for m in models])
        e = np.exp(v - v.max())
        out[p] = dict(zip(models, e / e.sum()))
    return out


def draw(ax, post, models, title):
    dom = {p: max(models, key=lambda m: v[m]) for p, v in post.items()}
    pids = sorted(post, key=lambda p: (models.index(dom[p]), -post[p][dom[p]]))
    x, bottoms = np.arange(len(pids)), np.zeros(len(pids))
    for m in models:
        vals = np.array([post[p][m] for p in pids])
        ax.bar(x, vals, bottom=bottoms, width=1.0, color=PALETTE[m],
               edgecolor="white", linewidth=0.3)
        bottoms += vals
    ax.set_xlim(-0.5, len(pids) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("P(model | choices)")
    ax.spines[["top", "right"]].set_visible(False)
    counts = Counter(dom.values())
    n_bayes = sum(c for m, c in counts.items() if m in BAYES)
    ax.set_title(title.format(n=len(post), k=n_bayes), fontsize=6.5, pad=3)
    return {"n": len(post), "bayesian_best": n_bayes, "counts": dict(counts.most_common()),
            "leading_post_gt_0.5": int(sum(post[p][dom[p]] > 0.5 for p in post))}


def legends(fig, used, y):
    proxy = lambda m: mpatches.Patch(facecolor=PALETTE[m], edgecolor="white", linewidth=0.3)
    kw = dict(loc="upper center", ncols=2, fontsize=5, title_fontsize=5.5, frameon=False,
              handlelength=1.2, handleheight=1.1, columnspacing=0.5, labelspacing=0.35,
              borderpad=0.2)
    for x, title, group in ((0.27, "Bayesian models", BAYES), (0.77, "Non-Bayesian baselines", BASE)):
        ms = [m for m in group if m in used]
        if ms:
            leg = fig.legend([proxy(m) for m in ms], ms, title=title, bbox_to_anchor=(x, y), **kw)
            leg._legend_box.align = "left"


def panel(ax, lls, title, summary, key):
    models = drop_collapsed(lls)
    summary[key] = draw(ax, posteriors(lls, models), models, title)
    summary[key]["dropped"] = [m for m in lls if m not in models]
    return models


def savefig(fig, name):
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(exist_ok=True)
    up, nu = load_lls("updaters"), load_lls("non-updaters")
    nu_base = {m: v for m, v in nu.items() if m not in BAYES}
    nu_base["Prior Sampling"] = nu["Bayesian-PS"]
    summary = {}

    fig, axes = plt.subplots(2, 1, figsize=(3.42, 3.9), gridspec_kw={"hspace": 0.35})
    up = {m: v for m, v in up.items() if m != "Bayesian-Threshold-BR"}
    used = panel(axes[0], up, "Updaters (n = {n}): Bayesian model best for {k}/{n}",
                 summary, "updaters")
    used += panel(axes[1], nu_base, "Non-updaters (n = {n}): baselines and Prior Sampling",
                  summary, "non_updaters")
    axes[1].set_xlabel("Participants (sorted by best-fitting model)")
    legends(fig, used, 0.04)
    savefig(fig, "indiv_fits")

    fig, ax = plt.subplots(figsize=(3.42, 1.9))
    used = panel(ax, {**nu, "Prior Sampling": nu["Bayesian-PS"]},
                 "Non-updaters (n = {n}), all models: Bayesian model best for {k}/{n}",
                 summary, "non_updaters_all_models")
    ax.set_xlabel("Participants (sorted by best-fitting model)")
    legends(fig, used, -0.02)
    savefig(fig, "indiv_fits_appendix")

    with open(OUT_DIR / "indiv_fits_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
