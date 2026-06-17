"""2026-06-17 PNAS figure revision — individual model fitting (R4).

Successor to 2026-06-16-pnas-figure-revisions/individual_fitting_revision.py
(was R2). The FITTING LOGIC is kept VERBATIM: the 13-model renormalization
from the stored per-participant log-likelihoods, the Mixture-PS exclusion,
and the dominant_counts assertion against results.json. None of the fitting
numbers change. Output is **R4_individual_fitting**.

All 13 models are shown individually (6 Bayesian + 7 non-Bayesian
baselines) under two titled sub-legends. The 06-16 version distinguished
the baselines with HATCHING, which read badly at single-column width
(the diagonal/dot/cross lines turned to mud in the thin bars). This
revision drops hatching entirely and instead leans on a purpose-built
palette + white separators:

  1. NO HATCHING. Every segment is a solid fill.
  2. TWO-FAMILY PALETTE. Bayesian models are a COOL family (blues, green,
     teal, purples); non-Bayesian baselines are a WARM/NEUTRAL family
     (orange, magenta, brown, red, gold, amber, grey). Cool-vs-warm is a
     redundant cue that matches the two sub-legend headers, so any
     Bayesian-vs-baseline confusion is a hue-family confusion the eye
     resolves even in thin bands. Within each family the hues are
     hand-tuned for maximal separation, a model and its "-PS" variant
     share a hue at different lightness (Walk/Walk-PS = blue,
     Threshold/Thresh-PS = purple) so the legend reads as pairs, and
     stack-adjacent segments are checked for luminance contrast.
  3. WHITE SEPARATORS. Each segment gets a slightly heavier white edge so
     the bands stay legible without any fill texture.

  The honest normalizer is unchanged: each bar sums to exactly 1
  (P(model | participant)); nothing is dropped or collapsed.

Run from analysis/:
    uv run python experiments/2026-06-17-pnas-figure-revisions/individual_fitting_revision.py
"""

from __future__ import annotations

import ast
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
PF_DIR = SCRIPT_DIR.parent / "2026-05-28-paper-figures"

RESULTS_PATH = PF_DIR / "results.json"
PAPER_FIGURES_PATH = PF_DIR / "paper_figures.py"

OUT_DIR = SCRIPT_DIR / "stuff to incorporate"
OUT_DIR.mkdir(exist_ok=True)
OUT_MD = SCRIPT_DIR / "summary.md"

FIG_W = 3.42         # single-column (PNAS)
FIG_H_IND = 2.95

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
})

# Same model groupings as the source.
EXCLUDED_FROM_INDIVIDUAL = {"Mixture-PS"}
BAYESIAN = [
    "Bayesian Walk", "Bayesian Walk-PS", "Mixture-PS", "Bayesian-Belief",
    "Bayesian-Value", "Bayesian Threshold", "Bayesian Thresh-PS",
]
BASELINES = [
    "Random Walk", "Top-7", "Random-to-Optimal", "Optimal",
    "Copy Others", "Contradict Others", "Random",
]

# Two-family palette (overrides paper_figures MODEL_COLORS), no hatching.
# Bayesian = COOL (blues / green / teal / purples); baselines = WARM/NEUTRAL
# (orange / magenta / brown / red / gold / amber / grey). Cool-vs-warm is a
# redundant cue matching the two sub-legend headers. Within each family the
# hues are hand-tuned for separation; a model and its "-PS" variant share a
# hue at different lightness (Walk/Walk-PS = blue, Threshold/Thresh-PS =
# purple); stack-adjacent segments are checked for luminance contrast.
PALETTE = {
    # Bayesian (cool)
    "Bayesian Walk":       "#08519c",   # dark blue
    "Bayesian Walk-PS":    "#9ecae1",   # light blue
    "Bayesian-Belief":     "#41ab5d",   # green
    "Bayesian-Value":      "#1c9099",   # teal
    "Bayesian Threshold":  "#6a51a3",   # purple
    "Bayesian Thresh-PS":  "#bcbddc",   # light purple
    # Baselines (warm / neutral)
    "Random Walk":         "#e6550d",   # orange
    "Top-7":               "#c51b7d",   # magenta
    "Random-to-Optimal":   "#8c510a",   # brown
    "Optimal":             "#b2182b",   # red
    "Copy Others":         "#fff7bc",   # pale gold (often negligible)
    "Contradict Others":   "#fb9a29",   # amber
    "Random":              "#969696",   # grey
}


def savefig(fig, name: str):
    paths = []
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kw)
        print(f"[individual] wrote {path.resolve()}")
        paths.append(path)
    return paths


def load_model_order():
    """MODEL_ORDER from paper_figures.py source (AST) — drives the sort."""
    tree = ast.parse(PAPER_FIGURES_PATH.read_text())
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "MODEL_ORDER"):
            return ast.literal_eval(node.value)
    raise AssertionError(f"could not find MODEL_ORDER in {PAPER_FIGURES_PATH}")


def fig_individual_fitting(res, model_order, model_colors):
    # ---- data logic VERBATIM from the source ------------------------------
    lls = res["individual"]["log_likelihoods"]
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

    dominant = {pid: max(model_order, key=lambda m: post[m])
                for pid, post in posteriors.items()}
    counts = Counter(dominant.values())
    stored = res["individual"]["dominant_counts"]
    assert dict(counts) == stored, (
        f"dominant-count mismatch: recomputed {dict(counts)} vs stored "
        f"{stored} — dropping Mixture-PS should not change any argmax")
    print("[individual] dominant-model counts match results.json "
          "(unchanged by dropping Mixture-PS):")
    for m in model_order:
        if m in counts:
            print(f"  {m}: {counts[m]}")
    # ---- end verbatim data logic ------------------------------------------

    pids_sorted_idx = {m: i for i, m in enumerate(model_order)}
    pids = sorted(posteriors,
                  key=lambda p: (pids_sorted_idx[dominant[p]],
                                 -posteriors[p][dominant[p]]))

    # Max per-participant posterior for every model (reported for context).
    max_post = {m: max(posteriors[p][m] for p in pids) for m in model_order}

    # All 13 models shown individually under two titled sub-legends.
    kept_bayesian = [m for m in model_order if m in BAYESIAN]
    all_baselines = [m for m in model_order if m in BASELINES]

    print("[individual] non-Bayesian baselines (all shown individually):")
    for m in all_baselines:
        print(f"  {m}: max per-participant posterior = {max_post[m]:.4f}, "
              f"dominant for {counts.get(m, 0)}")

    # Plot order: Bayesian (cool), then baselines (warm/neutral). Each bar
    # sums to exactly 1 (honest normalizer; nothing dropped or collapsed).
    plot_models = kept_bayesian + all_baselines

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_IND))
    x = np.arange(len(pids))
    bottoms = np.zeros(len(pids))
    for m in plot_models:
        vals = np.array([posteriors[p][m] for p in pids])
        # Heavier white separator (no hatching) keeps the bands legible.
        ax.bar(x, vals, bottom=bottoms, width=1.0,
               color=model_colors.get(m, "#999999"),
               edgecolor="white", linewidth=0.35)
        bottoms += vals

    # Sanity: bars sum to 1 (normalizer intact).
    assert np.allclose(bottoms, 1.0, atol=1e-9), \
        f"bars do not sum to 1 (max dev {np.abs(bottoms - 1).max():.2e})"

    ax.set_xlim(-0.5, len(pids) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel(
        f"Participants (n = {len(pids)}, sorted by best-fitting model)")
    ax.set_ylabel("P(model | participant's choices)")
    ax.set_xticks([])
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.spines[["top", "right"]].set_visible(False)

    def proxy(m):
        return mpatches.Patch(facecolor=model_colors.get(m, "#999999"),
                              edgecolor="white", linewidth=0.35)

    # Two titled sub-legends: Bayesian models (cool) | baselines (warm).
    fig.tight_layout(pad=0.3, rect=(0, 0.23, 1, 1))
    legend_kw = dict(loc="upper center", ncols=2, fontsize=5,
                     title_fontsize=5.5, frameon=False, handlelength=1.2,
                     handleheight=1.1, columnspacing=0.8, labelspacing=0.35,
                     borderpad=0.2)
    leg1 = fig.legend([proxy(m) for m in kept_bayesian], kept_bayesian,
                      title="Bayesian models",
                      bbox_to_anchor=(0.27, 0.22), **legend_kw)
    leg1.get_title().set_fontweight("bold")
    leg1._legend_box.align = "left"
    leg2 = fig.legend([proxy(m) for m in all_baselines], all_baselines,
                      title="Non-Bayesian baselines",
                      bbox_to_anchor=(0.74, 0.22), **legend_kw)
    leg2.get_title().set_fontweight("bold")
    leg2._legend_box.align = "left"

    savefig(fig, "R4_individual_fitting")
    plt.close(fig)

    return {
        "counts": counts, "n_participants": len(pids),
        "baselines": [(m, max_post[m], counts.get(m, 0)) for m in all_baselines],
        "model_order": model_order,
    }


def write_summary(info, res):
    s1 = res["scope"]["stage1"]
    counts = info["counts"]
    lines = [
        "# R4 — Individual model fitting figure (revised palette)",
        "",
        f"Posteriors over 13 models for {info['n_participants']} "
        f"participants, renormalized from the per-participant log-"
        f"likelihoods in `2026-05-28-paper-figures/results.json` "
        f"(agg_ll-objective fits; lapse = {res['scope']['lapse_rate']}; "
        f"Stage-1: tau_prior = {s1['tau_prior']:.4f}, "
        f"epsilon = {s1['epsilon']:.4f}, memory = `{s1['memory_strategy']}`). "
        f"Mixture-PS excluded (collapses onto Bayesian Walk-PS; would double-"
        f"count Walk-PS in the normalizer). The fitting numbers are unchanged "
        f"from `2026-06-07-instrumental-rationality`; only the colour encoding "
        f"changed from the 06-16 version (R2 -> renamed R4).",
        "",
        "## This revision",
        "",
        "All 13 models are shown individually (6 Bayesian + 7 non-Bayesian "
        "baselines) under two titled sub-legends. The 06-16 version "
        "distinguished the baselines with **hatching**, which read badly at "
        "single-column width — the diagonal/dot/cross lines turned to mud in "
        "the thin bars. Fixes:",
        "",
        "1. **No hatching.** Every segment is a solid fill.",
        "2. **Two-family palette.** Bayesian models are a COOL family (dark/"
        "light blue, green, teal, dark/light purple — a model and its `-PS` "
        "variant share a hue at different lightness); non-Bayesian baselines "
        "are a WARM/NEUTRAL family (orange, magenta, brown, red, pale gold, "
        "amber, grey). Cool-vs-warm is a redundant cue matching the two "
        "sub-legend headers; within each family the hues are hand-tuned for "
        "separation and stack-adjacent segments are checked for luminance "
        "contrast.",
        "3. **White separators.** A slightly heavier white edge keeps the "
        "bands legible without any fill texture. Single-column (3.42 in).",
        "",
        "The honest normalizer (each bar sums to 1) is unchanged — nothing is "
        "dropped or collapsed.",
        "",
        "## All seven non-Bayesian baselines (shown individually)",
        "",
        "| Model | colour | max per-participant posterior | dominant for |",
        "|-------|--------|--:|--:|",
    ]
    for m, mx, dom in info["baselines"]:
        lines.append(f"| {m} | `{PALETTE[m]}` | {mx:.4f} | {dom} |")
    lines += [
        "",
        "## Dominant-model counts (recomputed, match stored `dominant_counts`)",
        "",
        "| Model | n participants |",
        "|-------|--:|",
    ]
    for m in info["model_order"]:
        if m in counts:
            lines.append(f"| {m} | {counts[m]} |")
    lines.append("")
    section = "\n".join(lines)

    marker = "# R4 — Individual model fitting figure (revised palette)"
    existing = OUT_MD.read_text() if OUT_MD.exists() else ""
    if marker in existing:
        head = existing.split(marker)[0].rstrip()
        tail = existing.split(marker, 1)[1]
        rest = ""
        nxt = tail.find("\n# ")
        if nxt != -1:
            rest = tail[nxt + 1:]
        new = (head + "\n\n" + section + ("\n" + rest if rest else "")).strip()
    else:
        new = (existing.rstrip() + "\n\n" + section).strip() if existing \
            else section
    OUT_MD.write_text(new + "\n")
    print(f"[individual] wrote {OUT_MD.resolve()}")


def main():
    model_order = load_model_order()
    model_order = [m for m in model_order
                   if m not in EXCLUDED_FROM_INDIVIDUAL]
    with open(RESULTS_PATH) as f:
        res = json.load(f)
    info = fig_individual_fitting(res, model_order, PALETTE)
    write_summary(info, res)


if __name__ == "__main__":
    main()
