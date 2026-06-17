"""2026-06-16 PNAS figure revision — calibration (R1).

Revision of 2026-06-07-epistemic-rationality/epistemic_rationality.py
`fig_calibration()`. The binned-points-on-identity plot with cluster-
bootstrap CIs and the bin-occupancy histogram is CORRECT and kept verbatim.

The only change is the HEADLINE statistic. The source displayed
``r = 0.46`` — the Pearson r over the RAW (report, role) pairs with
y in {0, 1}. With a 0/1 indicator the per-pair variance caps Pearson r
well below 1 even under perfect probability matching, so the raw r
understates the agreement the figure actually shows. The advisor wants
the headline to be the r over the BINNED (decile) points (``r_binned``,
already computed in the source), with the raw-pair r kept as smaller
clearly-labeled secondary text.

We also report robustness to bin count: binned r at N_BINS in {5, 10, 20}.
The main plotted binning stays at 10 (decile).

Output: PNAS single-column figure (3.42 in wide), .png at 300 dpi and
.pdf, into "stuff to incorporate/" as R1_calibration. No in-figure title.

Run from analysis/:
    uv run python experiments/2026-06-16-pnas-figure-revisions/calibration_revision.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
V2_DIR = SCRIPT_DIR.parent / "2026-06-05-paper-figures-v2"
EPI_DIR = SCRIPT_DIR.parent / "2026-06-07-epistemic-rationality"
sys.path.insert(0, str(V2_DIR))

from common import (  # noqa: E402
    compute_posteriors, load_clean_human_teams, load_stage1, prepare_team,
    target_marginal,
)
from shared.constants import ROLE_NAMES  # noqa: E402

# Stage-1 source of truth: the 05-25 full pipeline (same convention as the
# 06-07 epistemic-rationality script). common.py loads the 05-12 fit — the
# two are byte-identical; we assert agreement so any re-fit fails loudly.
FULL_PIPELINE_STAGE1 = (SCRIPT_DIR.parent / "2026-05-25-full-pipeline"
                        / "stage1_inference" / "best_inference_params.json")

OUT_DIR = SCRIPT_DIR / "stuff to incorporate"
OUT_DIR.mkdir(exist_ok=True)
OUT_MD = SCRIPT_DIR / "summary.md"      # appended to (see write_summary)

N_BOOT = 10_000
N_BINS = 10                       # main plotted binning (decile)
BIN_COUNTS = (5, 10, 20)          # robustness check
SEED = 0

# PNAS single column: 3.42 in wide. Same font sizes as the 06-07 figures.
FIG_W = 3.42
FIG_H = 2.45

HUMAN_COLOR = "#000000"
REF_COLOR = "#999999"

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
    paths = []
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kw)
        print(f"[calibration] wrote {path.resolve()}")
        paths.append(path)
    return paths


def load_stage1_canonical():
    with open(FULL_PIPELINE_STAGE1) as f:
        s1_canon = json.load(f)
    s1, strat = load_stage1()
    for k in ("tau_prior", "epsilon", "memory_strategy"):
        assert s1_canon[k] == s1[k], (
            f"Stage-1 param mismatch on '{k}': 05-25 full pipeline has "
            f"{s1_canon[k]!r}, common.py (05-12) has {s1[k]!r} — "
            f"re-point common.py or update this experiment.")
    return s1_canon, strat


# ──────────────────────────────────────────────────────────────────────
# Data: one row per inference report (verbatim from epistemic_rationality)
# ──────────────────────────────────────────────────────────────────────

def collect_reports():
    s1, strat = load_stage1_canonical()
    teams = load_clean_human_teams()

    key_to_idx: dict = {}
    rows = []
    for key, team_prs in teams.items():
        data = prepare_team(team_prs)
        posteriors = compute_posteriors(data, s1["tau_prior"], s1["epsilon"],
                                        strat)
        for obs, si, target_pos, guessed, true_prev in data["queries"]:
            if si >= len(posteriors):
                continue
            ci = key_to_idx.setdefault(key, len(key_to_idx))
            rows.append({
                "cluster": ci,
                "round_number": int(key[2]),
                "observer": obs,
                "stage": si,
                "target": target_pos,
                "marginal": target_marginal(posteriors[si], target_pos),
                "guessed": guessed,
                "true_prev": true_prev,
            })
    meta = {
        "n_team_rounds": len(key_to_idx),
        "n_reports": len(rows),
        "stage1": {"tau_prior": s1["tau_prior"], "epsilon": s1["epsilon"],
                   "memory_strategy": strat.name},
    }
    print(f"[calibration] {len(rows)} reports from "
          f"{len(key_to_idx)} team-rounds")
    return rows, meta


# ──────────────────────────────────────────────────────────────────────
# Cluster bootstrap helpers (verbatim from epistemic_rationality)
# ──────────────────────────────────────────────────────────────────────

def _cluster_members(cluster):
    n_clusters = cluster.max() + 1
    return [np.flatnonzero(cluster == c) for c in range(n_clusters)]


def bin_stats(x, y, edges):
    """(mean x, mean y, n) per bin; NaN for empty bins."""
    mx = np.full(len(edges) - 1, np.nan)
    my = np.full(len(edges) - 1, np.nan)
    ns = np.zeros(len(edges) - 1, dtype=int)
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (x >= lo) & ((x < hi) if hi < edges[-1] else (x <= hi))
        ns[b] = m.sum()
        if ns[b]:
            mx[b] = x[m].mean()
            my[b] = y[m].mean()
    return mx, my, ns


def boot_calibration(cluster, x, y, edges, n_boot=N_BOOT, seed=SEED):
    """Cluster bootstrap: r over raw pairs + CI of mean(y) per bin."""
    rng = np.random.default_rng(seed)
    members = _cluster_members(cluster)
    n_clusters = len(members)
    boot_r = np.empty(n_boot)
    boot_bins = np.empty((n_boot, len(edges) - 1))
    for b in range(n_boot):
        picked = rng.integers(0, n_clusters, n_clusters)
        idx = np.concatenate([members[c] for c in picked])
        xb, yb = x[idx], y[idx]
        boot_r[b] = np.corrcoef(xb, yb)[0, 1]
        boot_bins[b] = bin_stats(xb, yb, edges)[1]
    r_ci = np.percentile(boot_r, [2.5, 97.5])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bin_ci = np.nanpercentile(boot_bins, [2.5, 97.5], axis=0)
    return r_ci, bin_ci


def binned_r(x, y, n_bins):
    """Pearson r over the (mean x, mean y) of `n_bins` equal-width bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mx, my, ns = bin_stats(x, y, edges)
    ok = ~np.isnan(my)
    r = float(np.corrcoef(mx[ok], my[ok])[0, 1])
    return r, int(ok.sum())


# ──────────────────────────────────────────────────────────────────────
# Figure — calibration against the posterior
# ──────────────────────────────────────────────────────────────────────

def fig_calibration(rows):
    cluster = np.repeat([r["cluster"] for r in rows], 3)
    x = np.concatenate([r["marginal"] for r in rows])
    y = np.array([float(r["guessed"] == ri) for r in rows for ri in range(3)])

    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    r_raw = float(np.corrcoef(x, y)[0, 1])
    r_ci, bin_ci = boot_calibration(cluster, x, y, edges)
    mx, my, ns = bin_stats(x, y, edges)
    ok = ~np.isnan(my)
    r_binned = float(np.corrcoef(mx[ok], my[ok])[0, 1])

    # Robustness: binned r at 5 / 10 / 20 bins.
    r_by_bins = {nb: binned_r(x, y, nb) for nb in BIN_COUNTS}

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    ax.plot([0, 1], [0, 1], "--", color=REF_COLOR, linewidth=0.8, zorder=1,
            label="identity (probability matching)")

    yerr = np.abs(bin_ci[:, ok] - my[ok])
    ax.errorbar(mx[ok], my[ok], yerr=yerr, fmt="o-", color=HUMAN_COLOR,
                markersize=2.6, linewidth=1.1, capsize=1.5, elinewidth=0.7,
                zorder=4, label="binned mean (95% CI)")

    # Bin-occupancy histogram along the bottom, counts labeled per bar.
    counts, _ = np.histogram(x, bins=edges)
    h = counts / counts.max() * 0.10
    centers = (edges[:-1] + edges[1:]) / 2
    ax.bar(centers, h, width=0.094, bottom=-0.145, color="#c2cdd6",
           alpha=0.8, zorder=2)
    ax.axhline(-0.145, color="#c2cdd6", linewidth=0.6)
    for c, cnt, hh in zip(centers, counts, h):
        if cnt:
            ax.text(c, -0.14 + hh + 0.012, f"{cnt}", ha="center",
                    va="bottom", fontsize=6, color="#777")

    # HEADLINE: r over the binned (decile) points — the statistic the
    # figure actually displays. Raw-pair r kept as smaller secondary text.
    # Placed in the open lower-right whitespace, above the histogram strip.
    ax.text(0.97, 0.42, f"$r$ = {r_binned:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            fontweight="bold")
    ax.text(0.97, 0.355, "binned (decile) points",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5,
            color="#555")
    ax.text(0.97, 0.285,
            f"robustness: $r$ = {r_by_bins[5][0]:.2f}/"
            f"{r_by_bins[10][0]:.2f}/{r_by_bins[20][0]:.2f} "
            f"at 5/10/20 bins",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.0,
            color="#777")
    ax.text(0.97, 0.225,
            f"raw pairs: $r$ = {r_raw:.2f} "
            f"[{r_ci[0]:.2f}, {r_ci[1]:.2f}], {len(y):,} pairs",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.0,
            color="#777")

    ax.set_xlabel("Bayesian model posterior probability of role")
    ax.set_ylabel("Fraction of reports naming role")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.155, 1.04)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.legend(loc="upper left", frameon=False, handlelength=1.6,
              borderpad=0.2, labelspacing=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=0.3)
    savefig(fig, "R1_calibration")
    plt.close(fig)

    # Per-role raw-pair correlations (kept from the source for the summary).
    role = np.tile(np.arange(3), len(rows))
    per_role = []
    for ri in range(3):
        m = role == ri
        rr = float(np.corrcoef(x[m], y[m])[0, 1])
        sub_cluster = np.unique(cluster[m], return_inverse=True)[1]
        rci, _ = boot_calibration(sub_cluster, x[m], y[m], edges,
                                  seed=SEED + 1 + ri)
        per_role.append((ROLE_NAMES[ri], rr, rci))

    return {
        "r_raw": r_raw, "r_ci": r_ci, "r_binned": r_binned,
        "r_by_bins": r_by_bins,
        "mx": mx, "my": my, "ns": ns, "bin_ci": bin_ci, "ok": ok,
        "per_role": per_role, "n_pairs": len(y),
    }


# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────

def write_summary(meta, cal):
    s1 = meta["stage1"]
    # Confirm the headline claim: binned points lie on identity (small mean
    # |my - mx| over occupied bins) and binned r >> raw r.
    ok = cal["ok"]
    dev = float(np.mean(np.abs(cal["my"][ok] - cal["mx"][ok])))
    lines = [
        "# R1 — Calibration figure (revised headline)",
        "",
        f"Scope: {meta['n_team_rounds']} clean human team-rounds (5 exports), "
        f"{meta['n_reports']:,} inference reports "
        f"({cal['n_pairs']:,} (report, role) pairs). Stage-1 params: "
        f"tau_prior = {s1['tau_prior']:.4f}, epsilon = {s1['epsilon']:.4f}, "
        f"memory = `{s1['memory_strategy']}`. All CIs are percentile cluster "
        f"bootstraps over team-rounds ({N_BOOT:,} resamples).",
        "",
        "## Headline statistic (the fix)",
        "",
        "The figure now headlines the Pearson r over the **binned (decile) "
        "points** (mean x vs mean y), not the raw (report, role) pairs.",
        "",
        "| Statistic | r |",
        "|-----------|---|",
        f"| **Binned (decile) points** — headline | **{cal['r_binned']:.3f}** |",
        f"| Raw (report, role) pairs | {cal['r_raw']:.3f} "
        f"[{cal['r_ci'][0]:.3f}, {cal['r_ci'][1]:.3f}] |",
        "",
        "## Robustness to bin count",
        "",
        "| N_BINS | binned r | occupied bins |",
        "|--:|--:|--:|",
    ]
    for nb in BIN_COUNTS:
        r, nocc = cal["r_by_bins"][nb]
        lines.append(f"| {nb} | {r:.3f} | {nocc} |")
    lines += [
        "",
        f"**Confirmation.** Over the {int(ok.sum())} occupied decile bins the "
        f"mean absolute deviation of report frequency from the posterior "
        f"probability (|mean y - mean x|) is **{dev:.3f}** — the binned points "
        f"lie essentially on the identity line. The binned r is "
        f"**{cal['r_binned']:.2f}** (very high) while the raw-pair r is only "
        f"**{cal['r_raw']:.2f}**. The discrepancy is expected: a 0/1 report "
        f"indicator has within-bin variance p(1-p) that caps the achievable "
        f"per-pair Pearson r far below 1 even under perfect probability "
        f"matching, so the raw-pair r understates the agreement the binned "
        f"points display.",
        "",
        "## Per-role raw-pair correlations (unchanged)",
        "",
        "| Role | raw-pair r | 95% CI |",
        "|------|--:|--|",
    ]
    for name, rr, rci in cal["per_role"]:
        lines.append(f"| {name} | {rr:.3f} | [{rci[0]:.3f}, {rci[1]:.3f}] |")
    lines += [
        "",
        "## Per-bin detail (decile, main plotted binning)",
        "",
        "| Posterior bin | n | mean x | Report frequency | 95% CI |",
        "|---------------|--:|-------:|-----------------:|--------|",
    ]
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    for b in range(N_BINS):
        if not cal["ok"][b]:
            continue
        lines.append(
            f"| {edges[b]:.1f}-{edges[b + 1]:.1f} | {cal['ns'][b]} "
            f"| {cal['mx'][b]:.3f} | {cal['my'][b]:.3f} "
            f"| [{cal['bin_ci'][0][b]:.3f}, {cal['bin_ci'][1][b]:.3f}] |")
    lines.append("")
    section = "\n".join(lines)

    # Append (or create) summary.md, replacing any prior R1-calibration block.
    marker = "# R1 — Calibration figure (revised headline)"
    existing = OUT_MD.read_text() if OUT_MD.exists() else ""
    if marker in existing:
        head = existing.split(marker)[0].rstrip()
        # Drop everything from this section to the next top-level "# " header.
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
    print(f"[calibration] wrote {OUT_MD.resolve()}")


def main():
    rows, meta = collect_reports()
    cal = fig_calibration(rows)
    print(f"\n  binned-means r (headline) = {cal['r_binned']:.3f}")
    print(f"  raw-pair r = {cal['r_raw']:.3f}  "
          f"95% CI [{cal['r_ci'][0]:.3f}, {cal['r_ci'][1]:.3f}]")
    for nb in BIN_COUNTS:
        r, nocc = cal["r_by_bins"][nb]
        print(f"  binned r @ {nb:2d} bins = {r:.3f}  ({nocc} occupied)")
    for name, rr, rci in cal["per_role"]:
        print(f"  {name:8s} raw r = {rr:.3f}  "
              f"95% CI [{rci[0]:.3f}, {rci[1]:.3f}]")
    write_summary(meta, cal)


if __name__ == "__main__":
    main()
