"""2026-06-07 instrumental rationality — three paper figures + one table.

Question: are people choosing roles in a maximizing-subjective-expected-
utility way?

Model-free (recomputed here from the value-rank rows — one row per
team-stage, rank of the played joint combo among all 27 by the
precomputed value matrices, section2/06-05 conventions):

  Figure 1 (rank_by_game) — mean value-rank of the played combo by GAME
    NUMBER (round 1-8 of the session; 06-05 task2 did the by-stage
    version and found it flat). Cluster-bootstrap 95% CIs + OLS trend
    slope, chance reference at rank 14, inverted y-axis (up = better).

  Figure 2 (topk_curves) — Top-K and Bottom-K cumulative curves vs the
    uniform K/27 diagonal, restyled from
    2026-05-28-paper-figures/section2_best_response.py with
    cluster-bootstrap 95% CI bands added. Reproduction-checked against
    its topk_summary.md (mean rank 9.71, Top-1 11.7%, Top-5 42.4%,
    Bottom-5 8.3%).

Model-based (copied from 2026-05-28-paper-figures/results.json — no
model re-runs; those fits use the 05-25 pipeline's agg_ll objective):

  Figure 3 (individual_fitting) — stacked bar of each participant's
    posterior over 13 models, sorted by dominant model. Restyle of the
    05-28 figure, bar panel only (no pie). Mixture-PS is excluded: its
    agg_ll fit collapsed to w = 1.0, i.e. exactly Bayesian Walk-PS
    (identical predictions, identical per-participant posteriors
    102/102) — keeping the duplicate would double-count Walk-PS in the
    posterior normalizer and deflate every other model's share. The
    13-model posteriors are renormalized here from the stored
    per-participant log-likelihoods; dropping the duplicate changes no
    dominant-model assignment (asserted against the stored
    dominant_counts).

  Table (aggregate_table.tex) — the full 14-model aggregate comparison
    (Mixture-PS row kept here, so the table itself shows the mixture
    result), ported from 2026-06-05-paper-figures-v2/task6_latex_table.py
    and diff-checked against its output (modulo the header comment).

All CIs are percentile cluster bootstraps over team-rounds (stages
within a team-round share players, stats, and history, so resampling
individual team-stages would understate uncertainty).

MODEL_ORDER / MODEL_COLORS are read from
2026-05-28-paper-figures/paper_figures.py via AST (a plain import would
collide on the module name `pipeline`: paper_figures imports the 05-25
pipeline while common.py imports the 05-12 one).

Output: PNAS single-column figures (3.42 in wide), .png at 300 dpi and
.pdf, into "stuff to incorporate/". No in-figure titles or panel letters
(LaTeX adds those).

Run from analysis/:
    uv run python experiments/2026-06-07-instrumental-rationality/instrumental_rationality.py
"""

from __future__ import annotations

import ast
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
V2_DIR = SCRIPT_DIR.parent / "2026-06-05-paper-figures-v2"
PF_DIR = SCRIPT_DIR.parent / "2026-05-28-paper-figures"
sys.path.insert(0, str(V2_DIR))

from common import compute_rank_rows, load_human_team_records  # noqa: E402

RESULTS_PATH = PF_DIR / "results.json"
PAPER_FIGURES_PATH = PF_DIR / "paper_figures.py"
V2_TABLE_PATH = V2_DIR / "aggregate_table.tex"

OUT_DIR = SCRIPT_DIR / "stuff to incorporate"
OUT_DIR.mkdir(exist_ok=True)
OUT_TEX = OUT_DIR / "aggregate_table.tex"
OUT_MD = SCRIPT_DIR / "summary.md"

N_BOOT = 10_000
SEED = 0

# Reproduction targets from 2026-05-28-paper-figures/topk_summary.md.
EXPECTED = {"n_stages": 721, "mean_rank": 9.71,
            "top1": 11.7, "top5": 42.4, "bottom5": 8.3}  # percentages

# PNAS single column: 3.42 in wide. Same width + font sizes as the
# epistemic-rationality figures so everything stacks cleanly in LaTeX.
FIG_W = 3.42
FIG_H = 2.45
FIG_H_IND = 2.95           # individual_fitting needs room for the legend

HUMAN_COLOR = "#000000"     # human data: black solid
ACCENT_COLOR = "#8e44ad"    # second human series (Bottom-K)
REF_COLOR = "#999999"       # reference elements: grey

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
})


def savefig(fig, name: str):
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kw)
        print(f"[instrumental] wrote {path}")


def load_model_style():
    """MODEL_ORDER / MODEL_COLORS from paper_figures.py source (AST)."""
    tree = ast.parse(PAPER_FIGURES_PATH.read_text())
    ns = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("MODEL_ORDER", "MODEL_COLORS")):
            ns[node.targets[0].id] = ast.literal_eval(node.value)
    assert set(ns) == {"MODEL_ORDER", "MODEL_COLORS"}, (
        f"could not find MODEL_ORDER/MODEL_COLORS in {PAPER_FIGURES_PATH}")
    return ns["MODEL_ORDER"], ns["MODEL_COLORS"]


# ──────────────────────────────────────────────────────────────────────
# Cluster bootstrap helpers (cluster = team-round, i.e. record_idx)
# ──────────────────────────────────────────────────────────────────────

def _cluster_members(cluster):
    n_clusters = cluster.max() + 1
    return [np.flatnonzero(cluster == c) for c in range(n_clusters)]


def boot_mean_ci(cluster, y, n_boot=N_BOOT, seed=SEED):
    """95% CI of mean(y) under a cluster bootstrap over team-rounds."""
    rng = np.random.default_rng(seed)
    members = _cluster_members(cluster)
    n_clusters = len(members)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, n_clusters, n_clusters)
        idx = np.concatenate([members[c] for c in picked])
        boots[b] = y[idx].mean()
    return np.percentile(boots, [2.5, 97.5])


def boot_slope_ci(cluster, x, y, n_boot=N_BOOT, seed=SEED):
    """Slope of OLS y~x and its 95% CI under a cluster bootstrap."""
    slope = float(np.polyfit(x, y, 1)[0])
    rng = np.random.default_rng(seed)
    members = _cluster_members(cluster)
    n_clusters = len(members)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, n_clusters, n_clusters)
        idx = np.concatenate([members[c] for c in picked])
        boots[b] = np.polyfit(x[idx], y[idx], 1)[0]
    return slope, np.percentile(boots, [2.5, 97.5])


def curves_from_ranks(ranks):
    """Top-K / Bottom-K cumulative curves (length 27) from integer ranks.

    top[k-1] = mean(rank <= k); bot[k-1] = mean(rank > 27-k). Same
    definition as section2_best_response.topk_bottomk_curves, via
    bincount so the bootstrap loop stays cheap.
    """
    n = len(ranks)
    c = np.cumsum(np.bincount(ranks, minlength=28)[1:28]) / n
    top = c
    bot = np.empty(27)
    bot[:-1] = 1.0 - c[:26][::-1]
    bot[-1] = 1.0
    return top, bot


def boot_curves(cluster, ranks, n_boot=N_BOOT, seed=SEED):
    """95% CI bands for the Top-K / Bottom-K curves (cluster bootstrap)."""
    rng = np.random.default_rng(seed)
    members = _cluster_members(cluster)
    n_clusters = len(members)
    top_b = np.empty((n_boot, 27))
    bot_b = np.empty((n_boot, 27))
    for b in range(n_boot):
        picked = rng.integers(0, n_clusters, n_clusters)
        idx = np.concatenate([members[c] for c in picked])
        top_b[b], bot_b[b] = curves_from_ranks(ranks[idx])
    return (np.percentile(top_b, [2.5, 97.5], axis=0),
            np.percentile(bot_b, [2.5, 97.5], axis=0))


# ──────────────────────────────────────────────────────────────────────
# Reproduction check vs section2's topk_summary.md
# ──────────────────────────────────────────────────────────────────────

def check_reproduction(rows):
    ranks = np.array([r["rank"] for r in rows])
    top, bot = curves_from_ranks(ranks)
    got = {"n_stages": len(ranks), "mean_rank": round(float(ranks.mean()), 2),
           "top1": round(float(top[0]) * 100, 1),
           "top5": round(float(top[4]) * 100, 1),
           "bottom5": round(float(bot[4]) * 100, 1)}
    print("[instrumental] reproduction check vs 05-28 topk_summary.md:")
    for k, v in EXPECTED.items():
        status = "OK" if got[k] == v else "MISMATCH"
        print(f"  {k}: expected {v}, got {got[k]}  [{status}]")
    assert got == EXPECTED, f"rank-row reproduction failed: {got}"
    return ranks, top, bot


# ──────────────────────────────────────────────────────────────────────
# Figure 1 — mean value-rank by game number (model-free, new)
# ──────────────────────────────────────────────────────────────────────

def fig_rank_by_game(rows):
    cluster = np.array([r["record_idx"] for r in rows])
    game = np.array([r["round_number"] for r in rows])
    rank = np.array([r["rank"] for r in rows], dtype=float)

    games = sorted(np.unique(game))
    out = []  # (game, n_stages, n_team_rounds, mean, ci, top1, top5)
    for k, g in enumerate(games):
        m = game == g
        sub_cluster = np.unique(cluster[m], return_inverse=True)[1]
        ci = boot_mean_ci(sub_cluster, rank[m], seed=SEED + 10 + k)
        out.append((g, int(m.sum()), len(np.unique(cluster[m])),
                    float(rank[m].mean()), ci,
                    float((rank[m] <= 1).mean()), float((rank[m] <= 5).mean())))

    slope, slope_ci = boot_slope_ci(cluster, game.astype(float), rank,
                                    seed=SEED + 70)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    xs = [o[0] for o in out]
    means = np.array([o[3] for o in out])
    cis = np.array([o[4] for o in out])
    ax.errorbar(xs, means, yerr=np.abs(cis.T - means),
                fmt="o-", color=HUMAN_COLOR, markersize=2.6, linewidth=1.1,
                capsize=1.5, elinewidth=0.7, zorder=3, label="human teams")

    ax.axhline(14, color=REF_COLOR, linestyle=":", linewidth=0.7, zorder=1)
    ax.text(xs[-1] + 0.42, 14 - 0.15, "chance (rank 14)", fontsize=6,
            color=REF_COLOR, va="bottom", ha="right")
    for o in out:
        ax.text(o[0], 15.25, f"{o[1]}", ha="center", va="center",
                fontsize=6, color="#777")
    ax.text(xs[0] - 0.42, 15.25, "n =", ha="right", va="center", fontsize=6,
            color="#777")

    ax.set_xlabel("Game number within session")
    ax.set_ylabel("Mean value-rank of played combo\n(1 = best of 27)")
    ax.set_xticks(xs)
    ax.set_xlim(xs[0] - 0.45, xs[-1] + 0.45)
    ax.set_ylim(15.9, 5.7)          # inverted: up = better
    ax.set_yticks([6, 8, 10, 12, 14])
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=0.3)
    savefig(fig, "rank_by_game")
    plt.close(fig)
    return out, slope, slope_ci


# ──────────────────────────────────────────────────────────────────────
# Figure 2 — Top-K / Bottom-K curves with CI bands (restyle)
# ──────────────────────────────────────────────────────────────────────

def fig_topk_curves(rows, ranks, top, bot):
    cluster = np.array([r["record_idx"] for r in rows])
    top_ci, bot_ci = boot_curves(cluster, ranks, seed=SEED + 200)

    ks = np.arange(1, 28)
    uniform = ks / 27.0

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(ks, uniform, "--", color=REF_COLOR, linewidth=0.8, zorder=1,
            label="uniform random (K/27)")

    ax.fill_between(ks, bot_ci[0], bot_ci[1], color=ACCENT_COLOR, alpha=0.18,
                    linewidth=0, zorder=2)
    ax.plot(ks, bot, "s-", color=ACCENT_COLOR, markersize=1.8, linewidth=1.0,
            zorder=3, label="Bottom-K")

    ax.fill_between(ks, top_ci[0], top_ci[1], color=HUMAN_COLOR, alpha=0.15,
                    linewidth=0, zorder=2)
    ax.plot(ks, top, "o-", color=HUMAN_COLOR, markersize=1.8, linewidth=1.0,
            zorder=4, label="Top-K")

    ax.annotate(f"Top-1: {top[0]:.0%}", xy=(1, top[0]),
                xytext=(1.0, 0.42), fontsize=6, color=HUMAN_COLOR,
                ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=HUMAN_COLOR,
                                linewidth=0.5, shrinkA=1, shrinkB=2))
    ax.annotate(f"Top-5: {top[4]:.0%}", xy=(5, top[4]),
                xytext=(4.4, 0.68), fontsize=6, color=HUMAN_COLOR,
                ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=HUMAN_COLOR,
                                linewidth=0.5, shrinkA=1, shrinkB=2))

    ax.set_xlabel("K (number of best / worst combos)")
    ax.set_ylabel("Fraction of team-stages")
    ax.set_xlim(0.5, 27.5)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 27])
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(lbl) for lbl in
             ("Top-K", "Bottom-K", "uniform random (K/27)")]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="lower right", frameon=False, handlelength=1.6,
              borderpad=0.2, labelspacing=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=0.3)
    savefig(fig, "topk_curves")
    plt.close(fig)
    return top_ci, bot_ci


# ──────────────────────────────────────────────────────────────────────
# Figure 3 — individual fitting, stacked bar only (model-based, copied)
# ──────────────────────────────────────────────────────────────────────

def fig_individual_fitting(res, model_order, model_colors):
    # Renormalize the per-participant posteriors over the 13 kept models
    # from the stored log-likelihoods (lapse already folded in upstream).
    # Equivalent to dropping Mixture-PS from the stored 14-model
    # posteriors and renormalizing — asserted below.
    lls = res["individual"]["log_likelihoods"]   # {model: {pid: ll}}
    stored_posts = res["individual"]["posteriors"]
    pids_all = list(next(iter(lls.values())))
    posteriors = {}
    for pid in pids_all:
        v = np.array([lls[m][pid] for m in model_order])
        e = np.exp(v - v.max())
        posteriors[pid] = dict(zip(model_order, e / e.sum()))
    for pid, post in posteriors.items():
        keep = 1.0 - stored_posts[pid]["Mixture-PS"]
        for m in model_order:
            assert abs(post[m] - stored_posts[pid][m] / keep) < 1e-9, (
                f"13-model posterior inconsistent with stored 14-model "
                f"posterior for {pid}/{m}")

    # Dominant model per participant; ties resolve to the earlier entry
    # in MODEL_ORDER (matches the 05-28 predict_fns iteration order).
    dominant = {pid: max(model_order, key=lambda m: post[m])
                for pid, post in posteriors.items()}
    counts = Counter(dominant.values())
    stored = res["individual"]["dominant_counts"]
    assert dict(counts) == stored, (
        f"dominant-count mismatch: recomputed {dict(counts)} vs stored "
        f"{stored} — dropping Mixture-PS should not change any argmax")
    print("[instrumental] dominant-model counts match results.json "
          "(unchanged by dropping Mixture-PS):")
    for m in model_order:
        if m in counts:
            print(f"  {m}: {counts[m]}")

    order_idx = {m: i for i, m in enumerate(model_order)}
    pids = sorted(posteriors,
                  key=lambda p: (order_idx[dominant[p]],
                                 -posteriors[p][dominant[p]]))

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_IND))
    x = np.arange(len(pids))
    bottoms = np.zeros(len(pids))
    for m in model_order:
        vals = np.array([posteriors[p][m] for p in pids])
        ax.bar(x, vals, bottom=bottoms, width=1.0,
               color=model_colors.get(m, "#95a5a6"), label=m,
               edgecolor="white", linewidth=0.15)
        bottoms += vals

    ax.set_xlim(-0.5, len(pids) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"Participants (n = {len(pids)}, sorted by best-fitting model)")
    ax.set_ylabel("P(model | participant's choices)")
    ax.set_xticks([])
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    ax.spines[["top", "right"]].set_visible(False)

    # Grouped legend: Bayesian models left, non-Bayesian baselines right.
    # Figure-level legends inside a reserved bottom strip — axes-level
    # legends hung below the axes land outside the canvas and get
    # dropped from the tight bounding box.
    fig.tight_layout(pad=0.3, rect=(0, 0.21, 1, 1))
    handles_by_label = dict(zip(*ax.get_legend_handles_labels()[::-1]))
    legend_kw = dict(loc="upper center", ncols=2, fontsize=5,
                     title_fontsize=5.5, frameon=False, handlelength=1.0,
                     handleheight=0.9, columnspacing=0.8, labelspacing=0.35,
                     borderpad=0.2)
    groups = [
        ("Bayesian models", [m for m in model_order if m in BAYESIAN],
         (0.27, 0.20)),
        ("Non-Bayesian baselines", [m for m in model_order if m in BASELINES],
         (0.73, 0.20)),
    ]
    for title, members, anchor in groups:
        leg = fig.legend([handles_by_label[m] for m in members], members,
                         title=title, bbox_to_anchor=anchor, **legend_kw)
        leg.get_title().set_fontweight("bold")
        leg._legend_box.align = "left"

    savefig(fig, "individual_fitting")
    plt.close(fig)
    return counts


# ──────────────────────────────────────────────────────────────────────
# Table — aggregate model comparison (ported from 06-05 task6)
# ──────────────────────────────────────────────────────────────────────

# Mixture-PS is excluded from the INDIVIDUAL FITTING only: its agg_ll
# fit collapses to w = 1.0 (pure Bayesian Walk-PS), so as a duplicate it
# would double-count Walk-PS in the model-posterior normalizer. The
# aggregate table keeps the row — there the collapse is the result.
EXCLUDED_FROM_INDIVIDUAL = {"Mixture-PS"}

BAYESIAN = [
    "Bayesian Walk", "Bayesian Walk-PS", "Mixture-PS", "Bayesian-Belief",
    "Bayesian-Value", "Bayesian Threshold", "Bayesian Thresh-PS",
]
BASELINES = [
    "Random Walk", "Top-7", "Random-to-Optimal", "Optimal",
    "Copy Others", "Contradict Others", "Random",
]
METRICS = ["combo_r", "marg_r", "agg_ll", "mean_ll"]


def emit_aggregate_table(res):
    metrics = res["aggregate"]["metrics_by_model"]
    n_records = res["scope"]["n_records"]

    missing = [m for m in BAYESIAN + BASELINES if m not in metrics]
    assert not missing, f"models missing from results.json: {missing}"

    best = {k: max(metrics[m][k] for m in BAYESIAN + BASELINES)
            for k in METRICS}

    def cell(model, key):
        v = metrics[model][key]
        s = f"{v:.3f}"
        return rf"\textbf{{{s}}}" if abs(v - best[key]) < 5e-4 else s

    def table_rows(group):
        ordered = sorted(group, key=lambda m: -metrics[m]["combo_r"])
        return [
            f"{m} & " + " & ".join(cell(m, k) for k in METRICS) + r" \\"
            for m in ordered
        ]

    lines = [
        r"% Auto-generated by experiments/2026-06-07-instrumental-"
        r"rationality/instrumental_rationality.py",
        rf"% Source: {RESULTS_PATH.relative_to(SCRIPT_DIR.parent)}",
        rf"% Fitted on {n_records} clean human team-rounds (5 exports);"
        r" Bayesian fits use the agg_ll objective.",
        r"\begin{table}",
        r"\centering",
        r"\caption{Aggregate model comparison on " + str(n_records) +
        r" clean human team-rounds. Combo-$r$ / marg-$r$: Pearson"
        r" correlation between predicted and empirical distributions over"
        r" joint role combinations / role marginals. Aggregate-LL:"
        r" per-stage aggregate cross-entropy of the empirical stage"
        r" distribution under the model; mean-LL: mean per-environment"
        r" log-likelihood. Higher is better for all four; best per column"
        r" in bold.}",
        r"\label{tab:model-comparison}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Combo-$r$ & Marg-$r$ & Aggregate-LL & Mean-LL \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Bayesian models}} \\",
        *table_rows(BAYESIAN),
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Non-Bayesian baselines}} \\",
        *table_rows(BASELINES),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    OUT_TEX.write_text("\n".join(lines) + "\n")
    print(f"[instrumental] wrote {OUT_TEX}")

    # Diff check vs the 06-05 table (ignoring the auto-generated comments).
    strip = lambda text: [l for l in text.splitlines()
                          if not l.startswith("%")]
    ref = strip(V2_TABLE_PATH.read_text())
    new = strip(OUT_TEX.read_text())
    assert ref == new, "aggregate_table.tex differs from the 06-05 version"
    print("[instrumental] aggregate_table.tex identical to 06-05 version "
          "(modulo header comments)  [OK]")


# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────

def write_summary(n_records, rows, ranks, top, bot, top_ci, bot_ci,
                  game_out, slope, slope_ci, counts, model_order, res):
    s1 = res["scope"]["stage1"]
    lines = [
        "# Instrumental rationality — value-rank of played combos + "
        "model comparison",
        "",
        f"Scope: {n_records} clean human team-rounds (5 exports), "
        f"{len(rows)} team-stage observations. Rank = position of the "
        f"played joint combo among all 27 by the precomputed value "
        f"matrices (eap-weighted, section2 conventions; 1 = best, "
        f"chance = 14). All CIs are percentile cluster bootstraps over "
        f"team-rounds ({N_BOOT:,} resamples).",
        "",
        "## Reproduction check (vs 05-28 `topk_summary.md`)",
        "",
        f"Mean rank **{ranks.mean():.2f}** (expected 9.71), "
        f"Top-1 **{top[0]:.1%}** (11.7%), Top-5 **{top[4]:.1%}** (42.4%), "
        f"Bottom-5 **{bot[4]:.1%}** (8.3%) — all match.",
        "",
        "## Figure 1 — mean value-rank by game number (`rank_by_game`)",
        "",
        "| Game | n stages | n team-rounds | Mean rank | 95% CI | Top-1 "
        "| Top-5 |",
        "|---|--:|--:|--:|---|--:|--:|",
    ]
    for g, n, ntr, mean, ci, t1, t5 in game_out:
        lines.append(f"| {g} | {n} | {ntr} | {mean:.2f} "
                     f"| [{ci[0]:.2f}, {ci[1]:.2f}] | {t1:.1%} | {t5:.1%} |")
    lines += [
        "",
        f"Trend (OLS slope, rank per game; negative = improving): "
        f"**{slope:+.3f}** [{slope_ci[0]:+.3f}, {slope_ci[1]:+.3f}].",
        "",
        "## Figure 2 — Top-K / Bottom-K curves (`topk_curves`)",
        "",
        "| K | Top-K | 95% CI | Bottom-K | 95% CI | Uniform |",
        "|--:|--:|---|--:|---|--:|",
    ]
    for k in (1, 3, 5, 7, 10, 14):
        i = k - 1
        lines.append(
            f"| {k} | {top[i]:.1%} | [{top_ci[0][i]:.1%}, "
            f"{top_ci[1][i]:.1%}] | {bot[i]:.1%} | [{bot_ci[0][i]:.1%}, "
            f"{bot_ci[1][i]:.1%}] | {k / 27:.1%} |")
    lines += [
        "",
        "## Figure 3 — individual model fitting (`individual_fitting`)",
        "",
        f"Posteriors over 13 models for "
        f"{res['scope']['n_participants']} participants, renormalized "
        f"from the per-participant log-likelihoods in "
        f"`2026-05-28-paper-figures/results.json` (agg_ll-objective "
        f"fits; lapse = {res['scope']['lapse_rate']}; Stage-1: "
        f"τ_prior = {s1['tau_prior']:.4f}, ε = {s1['epsilon']:.4f}, "
        f"memory = `{s1['memory_strategy']}`). **Mixture-PS is "
        f"excluded**: its fitted mixture weight under the agg_ll "
        f"objective is w = 1.0, i.e. it collapses exactly onto Bayesian "
        f"Walk-PS (identical posteriors for all 102 participants), so "
        f"keeping it would double-count Walk-PS in the model-posterior "
        f"normalizer. Dropping it changes no dominant-model assignment. "
        f"Dominant-model counts (recomputed, match the stored "
        f"`dominant_counts`):",
        "",
        "| Model | n participants |",
        "|-------|--:|",
    ]
    for m in model_order:
        if m in counts:
            lines.append(f"| {m} | {counts[m]} |")
    lines += [
        "",
        "## Table — `aggregate_table.tex`",
        "",
        "Regenerated from `2026-05-28-paper-figures/results.json`; "
        "diff-identical to `2026-06-05-paper-figures-v2/"
        "aggregate_table.tex` (modulo the auto-generated header comment).",
        "",
    ]
    OUT_MD.write_text("\n".join(lines))
    print(f"[instrumental] wrote {OUT_MD}")


def main():
    model_order, model_colors = load_model_style()
    model_order = [m for m in model_order
                   if m not in EXCLUDED_FROM_INDIVIDUAL]
    with open(RESULTS_PATH) as f:
        res = json.load(f)

    records = load_human_team_records()
    rows = compute_rank_rows(records)
    print(f"[instrumental] {len(rows)} team-stage observations from "
          f"{len(records)} team-rounds")
    ranks, top, bot = check_reproduction(rows)

    game_out, slope, slope_ci = fig_rank_by_game(rows)
    print(f"\n  overall mean rank: {ranks.mean():.2f}")
    for g, n, ntr, mean, ci, t1, t5 in game_out:
        print(f"  game {g}: n={n:3d} stages ({ntr} team-rounds)  "
              f"mean rank {mean:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]  "
              f"top-1 {t1:.1%}  top-5 {t5:.1%}")
    print(f"  trend: {slope:+.3f} rank/game  "
          f"95% CI [{slope_ci[0]:+.3f}, {slope_ci[1]:+.3f}]")

    top_ci, bot_ci = fig_topk_curves(rows, ranks, top, bot)
    counts = fig_individual_fitting(res, model_order, model_colors)
    emit_aggregate_table(res)

    write_summary(len(records), rows, ranks, top, bot, top_ci, bot_ci,
                  game_out, slope, slope_ci, counts, model_order, res)


if __name__ == "__main__":
    main()
