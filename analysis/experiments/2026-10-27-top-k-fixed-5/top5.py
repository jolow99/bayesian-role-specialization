"""2026-10-27 — Top-k with k fixed at 5 (not fit).

Fitting k by agg-LL drives it to k = 27, which predicts exactly what Random
predicts, so Top-k was dropped from Fig. 5 on 10-22. Following the 10-26
meeting, k is now fixed at 5: a round number, not tied to Fig. 3 or the fit.

1. Table 2 rows (tables.tex): Top-5 under the 10-20 scoring (λ = 0.01 lapse
   on the 27-combo joint, no floor); every other row is unchanged from the
   10-20 refit, but the bold (best per column) is recomputed.
2. Fig. 5 (R4): individual fits over 12 models (the 10-22 eleven + Top-5).
   Only Top-5 is scored here; the other eleven per-participant log-likelihoods
   are read from the 10-20 refit's results.json.

Run (about a minute) from analysis/:
    uv run python experiments/2026-10-27-top-k-fixed-5/top5.py
"""
from __future__ import annotations

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
EXP = SCRIPT_DIR.parent
REFIT = EXP / "2026-10-20-lapse-refit-all-models"
for d in ("2026-05-28-paper-figures", "2026-05-25-full-pipeline",
          "2026-10-20-tolerant-threshold-ps"):
    sys.path.insert(0, str(EXP / d))

import paper_figures as pf  # noqa: E402
import tolerant_threshold_ps as tps  # noqa: E402

K = 5
TOPK = "Top-7"  # key the refit / palette use for the Top-k model
FIG_DIR = SCRIPT_DIR / "stuff to incorporate"
FIG_DIR.mkdir(exist_ok=True)
REFIT_RES = json.load(open(REFIT / "results.json"))

DROPPED = ("Bayesian Threshold",)  # collapses onto Bayesian-BR (10-22)
ORDER = [m for m in tps.MODEL_ORDER if m not in DROPPED]
BAYES = [m for m in tps.BAYES if m not in DROPPED]
BASE = list(tps.BASE)
VALUE_BASE = ("Optimal", TOPK, "Random-to-Optimal")
dn = lambda m: f"Top-{K}" if m == TOPK else tps.dn(m)

# ── 1. score Top-5 ───────────────────────────────────────────────────

def score_top_k(records, k):
    fn = pf._wrap_baseline(pf.make_predict_top_k(k))
    metrics = tps.lapse_metrics(records, fn)
    lls = tps.individual_ll(records, pf.precompute_player_marginals(records, {TOPK: fn}))[TOPK]
    return metrics, lls

# ── 2. R4 individual fits over 12 models (10-22 plot, Top-5 added back) ──

def plot_r4(post):
    rc = {"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
          "font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 6,
          "ytick.labelsize": 6, "axes.linewidth": 0.6}
    with plt.rc_context(rc):
        dom = {p: max(ORDER, key=lambda m: v[m]) for p, v in post.items()}
        idx = {m: i for i, m in enumerate(ORDER)}
        pids = sorted(post, key=lambda p: (idx[dom[p]], -post[p][dom[p]]))
        fig, ax = plt.subplots(figsize=(3.42, 2.95))
        x = np.arange(len(pids)); bottoms = np.zeros(len(pids))
        for m in ORDER:
            vals = np.array([post[p][m] for p in pids])
            ax.bar(x, vals, bottom=bottoms, width=1.0, color=tps.PALETTE[m],
                   edgecolor="white", linewidth=0.35)
            bottoms += vals
        assert np.allclose(bottoms, 1.0, atol=1e-9)
        ax.set_xlim(-0.5, len(pids) - 0.5); ax.set_ylim(0, 1)
        ax.set_xlabel(f"Participants (n = {len(pids)}, sorted by best-fitting model)")
        ax.set_ylabel("P(model | participant's choices)")
        ax.set_xticks([]); ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.spines[["top", "right"]].set_visible(False)
        proxy = lambda m: mpatches.Patch(facecolor=tps.PALETTE[m], edgecolor="white", linewidth=0.35)
        fig.tight_layout(pad=0.3, rect=(0, 0.23, 1, 1))
        kw = dict(loc="upper center", ncols=2, fontsize=5, title_fontsize=5.5, frameon=False,
                  handlelength=1.2, handleheight=1.1, columnspacing=0.5, labelspacing=0.35,
                  borderpad=0.2)
        l1 = fig.legend([proxy(m) for m in BAYES], [dn(m) for m in BAYES],
                        title="Bayesian models", bbox_to_anchor=(0.25, 0.22), **kw)
        l2 = fig.legend([proxy(m) for m in BASE], [dn(m) for m in BASE],
                        title="Non-Bayesian baselines", bbox_to_anchor=(0.79, 0.22), **kw)
        for l in (l1, l2):
            l._legend_box.align = "left"
        for ext, k in (("png", {"dpi": 300}), ("pdf", {})):
            fig.savefig(FIG_DIR / f"R4_individual_fitting.{ext}", bbox_inches="tight",
                        facecolor="white", **k)
        plt.close(fig)
    return dom


def individual(lls):
    post = tps.posteriors_from_ll(lls, ORDER)
    dom = plot_r4(post)
    counts = Counter(dom.values())
    n = len(post)
    groups = {"Bayesian": sum(counts[m] for m in BAYES),
              "value-based baselines": sum(counts[m] for m in VALUE_BASE)}
    groups["noise / contrarian baselines"] = n - sum(groups.values())
    leading = int(sum(max(v.values()) > 0.5 for v in post.values()))
    return dict(n=n, counts={m: counts.get(m, 0) for m in ORDER}, groups=groups,
                leading_over_half=leading)

# ── 3. Table 2 rows ──────────────────────────────────────────────────

def num(x, bold=False):
    s = f"{abs(x):.3f}"
    if bold:
        return f"$\\mathbf{{-{s}}}$" if x < 0 else f"\\textbf{{{s}}}"
    return f"$-${s}" if x < 0 else s


MAIN_ROWS = [("Bayesian-BR", "Bayesian-Value"), ("Bayesian-Walk-BR", "Bayesian Walk"),
             ("Bayesian-Threshold-BR", "Bayesian Threshold"), ("Bayesian-PS", "Bayesian-Belief"),
             ("Bayesian-Walk-PS", "Bayesian Walk-PS"), ("Bayesian-Threshold-PS", "Bayesian Thresh-PS"),
             None,
             ("Optimal", "Optimal"), (f"Top-$k$ ($k={K}$)", TOPK),
             ("Random-to-Optimal", "Random-to-Optimal"), ("Copy Others", "Copy Others"),
             ("Contradict Others", "Contradict Others"), ("Random Walk", "Random Walk"),
             ("Random", "Random")]
METRICS = ("combo_r", "marg_r", "agg_ll", "mean_ll")


def write_tables(mb):
    best = {k: max(mb[m][k] for _, m in filter(None, MAIN_ROWS)) for k in METRICS}
    L = [f"% main-text Table 2 rows (agg-LL fits, lambda = 0.01 lapse, no floor; Top-k at k = {K})"]
    for row in MAIN_ROWS:
        if row is None:
            L.append("\\midrule")
            continue
        label, m = row
        cells = [num(mb[m][k], mb[m][k] == best[k]) for k in METRICS]
        L.append(f"{label:<22}& " + " & ".join(cells) + " \\\\")
    (SCRIPT_DIR / "tables.tex").write_text("\n".join(L) + "\n")


def main():
    records = pf.load_records_with_pids(verbose=False)
    lls = {m: REFIT_RES["individual_ll"][m] for m in ORDER if m != TOPK}

    # sanity: this scoring reproduces the refit's Top-27 (agg-LL fit) and its k = 5 profile entry
    m27, ll27 = score_top_k(records, 27)
    assert all(abs(ll27[p] - REFIT_RES["individual_ll"][TOPK][p]) < 1e-9 for p in ll27)
    assert all(abs(m27[k] - REFIT_RES["metrics_by_model"][TOPK][k]) < 1e-9 for k in METRICS)
    metrics, lls[TOPK] = score_top_k(records, K)
    assert all(abs(metrics[k] - REFIT_RES["top_k_profile"][str(K)][k]) < 1e-9 for k in METRICS)
    assert set(lls[TOPK]) == set(lls["Random"])

    mb = dict(REFIT_RES["metrics_by_model"])
    mb[TOPK] = metrics
    write_tables(mb)
    ind = individual(lls)
    json.dump({"k": K, "lapse": REFIT_RES["lapse"], "top_k_metrics": metrics,
               "individual_ll_top_k": lls[TOPK], **ind},
              open(SCRIPT_DIR / "results.json", "w"), indent=2)

    L = [f"# Top-k at k = {K}, and individual fits over 12 models (Fig. 5)", "",
         "## Table 2 row", "", "| Model | combo-r | marg-r | agg-LL | mean-LL |", "|---|--:|--:|--:|--:|"]
    for m, label in ((TOPK, f"Top-{K}"), (TOPK, "Top-27 (10-20 fit)"), ("Random", "Random"),
                     ("Optimal", "Optimal")):
        x = metrics if label == f"Top-{K}" else REFIT_RES["metrics_by_model"][m]
        L.append(f"| {label} | " + " | ".join(f"{x[k]:.3f}" for k in METRICS) + " |")
    L += ["", f"## Individual fits (dropped: {', '.join(tps.dn(m) for m in DROPPED)}, which "
          "collapses onto Bayesian-BR)", "",
          f"λ = {REFIT_RES['lapse']} lapse on each player's marginal, refit (agg-LL) parameters.", "",
          "| Model | Dominant for |", "|---|--:|"]
    L += [f"| {dn(m)} | {c} |" for m, c in ind["counts"].items()]
    L += ["", "| Group | n | % |", "|---|--:|--:|"]
    L += [f"| {g} | {c} | {100 * c / ind['n']:.0f}% |" for g, c in ind["groups"].items()]
    L += ["", f"Leading model posterior > 0.5: **{ind['leading_over_half']} of {ind['n']}**"]
    (SCRIPT_DIR / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
