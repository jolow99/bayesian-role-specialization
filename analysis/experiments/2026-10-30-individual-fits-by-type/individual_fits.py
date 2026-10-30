"""2026-10-30 — Fig. 5 (individual fits) for the population, Bayesians and non-Bayesians,
with mixed beliefs from the 10-27 two-type classifier (τ_prior = 4.05, ε = 0.246, β = 0.726).

Teams mix types, so each group's models are fit per player (per-player agg-LL,
λ = 0.01 lapse on each marginal; see fit_helpers.py), each group on its
own players' choices. Top-k is fixed at k = 5 as in the paper. Per participant:
log-likelihood of their role choices under each model, softmax over models
(uniform prior). As in the paper's Fig. 5, a model whose per-participant
log-likelihoods match another's to < 1e-3 for every participant is dropped.

Run from analysis/:
    uv run python experiments/2026-10-30-individual-fits-by-type/individual_fits.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import fit_helpers as fh  # noqa: E402
from fit_helpers import pf, strategy_from_params, tps  # noqa: E402

CLASSIFIER = SCRIPT_DIR.parent / "2026-10-27-inference-soft-classifier" / "best_params.json"
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
GROUPS = ["Population", "Bayesian", "Non-Bayesian"]
TOP_K = 5


def main():
    clf = json.load(open(CLASSIFIER))
    with open(fh.TYPES_CSV) as f:
        types = {r["participant_id"]: float(r["p_prior_type"]) for r in csv.DictReader(f)}
    records = pf.load_records_with_pids(verbose=False)
    ob = fh.Obs(records, types)
    strat = strategy_from_params("full", None, clf["beta"])
    assert strat.kind == "drift_prior" and strat.param == clf["beta"]
    V = fh.Variant("mixed", records, ob, clf, strat)
    base = {m: fh.baseline_marginals(records, ob, fn) for m, fn in fh.BASE_FIXED.items()}
    base["topk"] = {TOP_K: fh.baseline_marginals(records, ob, pf.make_predict_top_k(TOP_K))}
    masks = {"Population": np.ones(ob.n, bool), "Bayesian": ~ob.nonbayes,
             "Non-Bayesian": ob.nonbayes}

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "axes.linewidth": 0.6})
    n_part = {g: len({records[ri]["participant_ids"][p]
                      for ri, p in zip(ob.ri[masks[g]], ob.pos[masks[g]])}) for g in GROUPS}
    # PNAS single column, panels stacked
    fig, axes = plt.subplots(3, 1, figsize=(3.42, 5.4), gridspec_kw={"hspace": 0.32})
    out, used = {}, set()
    for ax, g in zip(axes, GROUPS):
        mask = masks[g]
        P = fh.fit_models(ob, mask, V)
        P["Random Walk"] = {"eps": fh._fit_1d(
            lambda e: fh.metrics(fh.random_walk(ob, e), ob, mask)["agg_ll"],
            np.linspace(0, 1, 21), 0, 1)}
        P["Top-7"] = {"k": TOP_K}
        P.update({m: {} for m in fh.BASE_FIXED})
        lls = fh.panel_lls(ob, records, mask, V, base, P)
        print(f"  {g}")
        models = fh.drop_collapsed(lls)
        post = tps.posteriors_from_ll({m: lls[m] for m in models}, models)
        dom = fh.draw(ax, post, models)
        used.update(models)
        counts = Counter(dom.values())
        n_bayes = sum(c for m, c in counts.items() if m in fh.BAYES)
        ax.set_title(f"{fh.GROUP_TITLE[g]} (n = {len(post)}): "
                     f"Bayesian model best for {n_bayes}/{len(post)}", fontsize=6.5, pad=3)
        ax.set_ylabel("P(model | choices)")
        out[g] = {"n": len(post), "bayesian_dominant": n_bayes,
                  "leading_post_gt_0.5": int(sum(post[p][dom[p]] > 0.5 for p in post)),
                  "counts": {("Top-5" if m == "Top-7" else tps.dn(m)): c
                             for m, c in counts.most_common()},
                  "dropped": [tps.dn(m) for m in fh.MODEL_ORDER if m not in models],
                  "params": {tps.dn(m): fh.pstr(p) for m, p in P.items() if p}}

    lab = lambda m: "Top-5" if m == "Top-7" else tps.dn(m)
    proxy = lambda m: mpatches.Patch(facecolor=tps.PALETTE[m], edgecolor="white", linewidth=0.3)
    kw = dict(loc="upper center", ncols=2, fontsize=5, title_fontsize=5.5, frameon=False,
              handlelength=1.2, handleheight=1.1, columnspacing=0.5, labelspacing=0.35,
              borderpad=0.2)
    bayes = [m for m in tps.BAYES if m in used]
    basel = [m for m in tps.BASE if m in used]
    l1 = fig.legend([proxy(m) for m in bayes], [lab(m) for m in bayes],
                    title="Bayesian models", bbox_to_anchor=(0.27, 0.06), **kw)
    l2 = fig.legend([proxy(m) for m in basel], [lab(m) for m in basel],
                    title="Non-Bayesian baselines", bbox_to_anchor=(0.77, 0.06), **kw)
    for l in (l1, l2):
        l._legend_box.align = "left"
    axes[-1].set_xlabel("Participants (sorted by best-fitting model)")
    for ext, k in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(FIG_DIR / f"individual_fits_by_type.{ext}", bbox_inches="tight",
                    facecolor="white", **k)
    plt.close(fig)
    json.dump(out, open(SCRIPT_DIR / "individual_fits_summary.json", "w"), indent=2)
    for g, s in out.items():
        print(f"{g:13s} Bayesian {s['bayesian_dominant']}/{s['n']}  leading>0.5 "
              f"{s['leading_post_gt_0.5']}  dropped {s['dropped']}\n   {s['counts']}\n   {s['params']}")


if __name__ == "__main__":
    main()
