"""2026-06-16 PNAS figure revision — individual model fitting (R2).

Revision of 2026-06-07-instrumental-rationality/instrumental_rationality.py
`fig_individual_fitting()`. The data logic is kept VERBATIM: the 13-model
renormalization from the stored per-participant log-likelihoods, the
Mixture-PS exclusion, the dominant-model sort, and the dominant_counts
assertion against results.json. None of the fitting changes.

The problem the advisor flagged: in the stacked bars the non-Bayesian
baselines are hard to tell apart from the Bayesian models and from each
other. Two fixes:

  1. HATCHING. Non-Bayesian baselines get distinct hatch patterns
     ('///', '...', 'xxx', '\\\\', '++', 'oo', '--'); Bayesian models stay
     solid fills. The same hatch appears on the legend swatches. This
     separates baselines from Bayesian models and from each other — in
     particular Random Walk vs Bayesian Walk-PS and Copy Others vs Random.

  2. NEGLIGIBLE-BASELINE COLLAPSE. A baseline is "negligible" iff its
     per-participant posterior mass is < 0.01 for EVERY participant AND it
     is the dominant model for 0 participants. Negligible baselines are
     lumped into a single thin grey "other baselines" segment so the bars
     stay an honest P(model | participant) (the normalizer is unchanged).
     Bayesian models are never dropped or collapsed. Exactly which models
     were collapsed (and their max per-participant posterior) is printed
     and written to summary.md.

Output: PNAS single-column figure (3.42 in wide, FIG_H_IND = 2.95), .png
at 300 dpi and .pdf, into "stuff to incorporate/" as R2_individual_fitting.
Legend grouped Bayesian vs baselines. No in-figure title.

Run from analysis/:
    uv run python experiments/2026-06-16-pnas-figure-revisions/individual_fitting_revision.py
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
V2_DIR = SCRIPT_DIR.parent / "2026-06-05-paper-figures-v2"
PF_DIR = SCRIPT_DIR.parent / "2026-05-28-paper-figures"
sys.path.insert(0, str(V2_DIR))

RESULTS_PATH = PF_DIR / "results.json"
PAPER_FIGURES_PATH = PF_DIR / "paper_figures.py"

OUT_DIR = SCRIPT_DIR / "stuff to incorporate"
OUT_DIR.mkdir(exist_ok=True)
OUT_MD = SCRIPT_DIR / "summary.md"

FIG_W = 3.42
FIG_H_IND = 2.95

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

# Distinct hatch per baseline (Bayesian models stay solid / no hatch).
BASELINE_HATCH = {
    "Random Walk":       "///",
    "Top-7":             "...",
    "Random-to-Optimal": "xxx",
    "Optimal":           "\\\\\\",
    "Copy Others":       "++",
    "Contradict Others": "oo",
    "Random":            "--",
}
OTHER_COLOR = "#bdc3c7"     # thin grey "other baselines" lump
OTHER_HATCH = ""
OTHER_LABEL = "other baselines (negligible)"

NEGLIGIBLE_THRESH = 0.01    # < this for every participant => negligible


def savefig(fig, name: str):
    paths = []
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kw)
        print(f"[individual] wrote {path.resolve()}")
        paths.append(path)
    return paths


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

    # Max per-participant posterior for every model (used for collapse).
    max_post = {m: max(posteriors[p][m] for p in pids) for m in model_order}

    # Collapse rule: negligible baseline = baseline with max per-participant
    # posterior < NEGLIGIBLE_THRESH AND dominant for 0 participants.
    # Bayesian models are never collapsed.
    baselines_present = [m for m in model_order if m in BASELINES]
    collapsed = [m for m in baselines_present
                 if max_post[m] < NEGLIGIBLE_THRESH and m not in counts]
    kept_baselines = [m for m in baselines_present if m not in collapsed]
    kept_bayesian = [m for m in model_order if m in BAYESIAN]

    print(f"[individual] negligible-baseline collapse "
          f"(< {NEGLIGIBLE_THRESH} for every participant AND dominant for 0):")
    if collapsed:
        for m in collapsed:
            print(f"  COLLAPSED {m}: max per-participant posterior "
                  f"= {max_post[m]:.4f}, dominant for {counts.get(m, 0)}")
    else:
        print("  (none collapsed)")
    for m in baselines_present:
        if m not in collapsed:
            print(f"  kept      {m}: max per-participant posterior "
                  f"= {max_post[m]:.4f}, dominant for {counts.get(m, 0)}")

    # Plot order: Bayesian (solid), then kept baselines (hatched), then the
    # lumped "other baselines" segment last. Normalizer is unchanged: the
    # "other" segment carries the summed mass of the collapsed baselines, so
    # each bar still sums to exactly P(model | participant) = 1.
    plot_models = kept_bayesian + kept_baselines
    other_vals = (np.array([sum(posteriors[p][m] for m in collapsed)
                            for p in pids]) if collapsed
                  else np.zeros(len(pids)))

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_IND))
    x = np.arange(len(pids))
    bottoms = np.zeros(len(pids))
    seg_handles = {}
    for m in plot_models:
        vals = np.array([posteriors[p][m] for p in pids])
        hatch = BASELINE_HATCH.get(m, "") if m in BASELINES else ""
        bars = ax.bar(x, vals, bottom=bottoms, width=1.0,
                      color=model_colors.get(m, "#95a5a6"),
                      hatch=hatch if hatch else None,
                      edgecolor="white", linewidth=0.15)
        seg_handles[m] = bars[0]
        bottoms += vals
    if collapsed:
        ax.bar(x, other_vals, bottom=bottoms, width=1.0, color=OTHER_COLOR,
               hatch=OTHER_HATCH if OTHER_HATCH else None,
               edgecolor="white", linewidth=0.15)
        bottoms += other_vals

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

    # Legend swatches: match fill colour + hatch. Build proxies so the
    # hatch shows even where a segment is tiny in the bars.
    def proxy(m):
        return mpatches.Patch(facecolor=model_colors.get(m, "#95a5a6"),
                              hatch=(BASELINE_HATCH.get(m, "")
                                     if m in BASELINES else None),
                              edgecolor="white", linewidth=0.15)

    bayes_labels = kept_bayesian
    base_labels = list(kept_baselines)
    base_proxies = [proxy(m) for m in base_labels]
    if collapsed:
        base_labels = base_labels + [OTHER_LABEL]
        base_proxies = base_proxies + [
            mpatches.Patch(facecolor=OTHER_COLOR, edgecolor="white",
                           linewidth=0.15)]

    fig.tight_layout(pad=0.3, rect=(0, 0.23, 1, 1))
    legend_kw = dict(loc="upper center", ncols=2, fontsize=5,
                     title_fontsize=5.5, frameon=False, handlelength=1.2,
                     handleheight=1.1, columnspacing=0.8, labelspacing=0.35,
                     borderpad=0.2)
    leg1 = fig.legend([proxy(m) for m in bayes_labels], bayes_labels,
                      title="Bayesian models (solid)",
                      bbox_to_anchor=(0.27, 0.22), **legend_kw)
    leg1.get_title().set_fontweight("bold")
    leg1._legend_box.align = "left"
    leg2 = fig.legend(base_proxies, base_labels,
                      title="Non-Bayesian baselines (hatched)",
                      bbox_to_anchor=(0.74, 0.22), **legend_kw)
    leg2.get_title().set_fontweight("bold")
    leg2._legend_box.align = "left"

    savefig(fig, "R2_individual_fitting")
    plt.close(fig)

    return {
        "counts": counts, "n_participants": len(pids),
        "collapsed": [(m, max_post[m], counts.get(m, 0)) for m in collapsed],
        "kept_baselines": [(m, max_post[m], counts.get(m, 0))
                           for m in kept_baselines],
        "model_order": model_order,
    }


def write_summary(info, res):
    s1 = res["scope"]["stage1"]
    counts = info["counts"]
    lines = [
        "# R2 — Individual model fitting figure (revised)",
        "",
        f"Posteriors over 13 models for {info['n_participants']} "
        f"participants, renormalized from the per-participant log-"
        f"likelihoods in `2026-05-28-paper-figures/results.json` "
        f"(agg_ll-objective fits; lapse = {res['scope']['lapse_rate']}; "
        f"Stage-1: tau_prior = {s1['tau_prior']:.4f}, "
        f"epsilon = {s1['epsilon']:.4f}, memory = `{s1['memory_strategy']}`). "
        f"Mixture-PS excluded (collapses onto Bayesian Walk-PS; would double-"
        f"count Walk-PS in the normalizer). The fitting logic is unchanged "
        f"from `2026-06-07-instrumental-rationality`.",
        "",
        "## Visual fixes",
        "",
        "1. **Hatching.** Non-Bayesian baselines now carry distinct hatch "
        "patterns (Random Walk `///`, Top-7 `...`, Random-to-Optimal `xxx`, "
        "Optimal `\\\\\\`, Copy Others `++`, Contradict Others `oo`, Random "
        "`--`); Bayesian models stay solid fills. The same hatch is shown on "
        "the legend swatches. Random Walk vs Bayesian Walk-PS and Copy Others "
        "vs Random are now separable.",
        "2. **Negligible-baseline collapse.** A baseline is negligible iff its "
        f"per-participant posterior is < {NEGLIGIBLE_THRESH} for **every** "
        "participant **and** it is dominant for 0 participants. Negligible "
        "baselines are lumped into a single thin grey \"other baselines\" "
        "segment. The normalizer is left intact (the bars still sum to 1 and "
        "remain an honest P(model | participant)); only the colour/hatch "
        "encoding of the lumped mass changes.",
        "",
        "## Models collapsed into \"other baselines\"",
        "",
    ]
    if info["collapsed"]:
        lines += ["| Model | max per-participant posterior | dominant for |",
                  "|-------|--:|--:|"]
        for m, mx, dom in info["collapsed"]:
            lines.append(f"| {m} | {mx:.4f} | {dom} |")
    else:
        lines.append("None — no baseline met the negligibility criterion.")
    lines += [
        "",
        "## Baselines kept as their own hatched segment",
        "",
        "| Model | max per-participant posterior | dominant for |",
        "|-------|--:|--:|",
    ]
    for m, mx, dom in info["kept_baselines"]:
        lines.append(f"| {m} | {mx:.4f} | {dom} |")
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

    marker = "# R2 — Individual model fitting figure (revised)"
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
    model_order, model_colors = load_model_style()
    model_order = [m for m in model_order
                   if m not in EXCLUDED_FROM_INDIVIDUAL]
    with open(RESULTS_PATH) as f:
        res = json.load(f)
    info = fig_individual_fitting(res, model_order, model_colors)
    write_summary(info, res)


if __name__ == "__main__":
    main()
