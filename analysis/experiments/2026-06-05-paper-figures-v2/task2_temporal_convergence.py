"""Task 2 — temporal convergence figure (figures/temporal_convergence.png).

For each team-stage, rank the played joint combo among all 27 by the
precomputed value matrix (eap-weighted, section2 conventions).

Panel A: mean rank vs stage number (1-5) with bootstrap 95% CIs and the
chance line at rank 14.
Panel B: fraction of team-stages whose combo is top-1 / top-5 vs stage,
with bootstrap 95% CIs and the chance lines (1/27, 5/27).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    FIGURES_DIR, SCRIPT_DIR, compute_rank_rows, load_human_team_records,
)

OUT_PATH = FIGURES_DIR / "temporal_convergence.png"
N_BOOT = 10_000
RNG = np.random.default_rng(0)


def boot_ci(vals, stat=np.mean):
    """Bootstrap 95% CI for a statistic over a 1-D sample."""
    vals = np.asarray(vals, dtype=float)
    idx = RNG.integers(0, len(vals), size=(N_BOOT, len(vals)))
    boots = stat(vals[idx], axis=1)
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def main():
    records = load_human_team_records()
    rows = compute_rank_rows(records)
    print(f"[task2] {len(rows)} team-stage observations "
          f"from {len(records)} team-rounds")

    stages = sorted({r["stage"] for r in rows})
    per_stage = {s: np.array([r["rank"] for r in rows if r["stage"] == s])
                 for s in stages}

    # Fixed cohort: rounds observed for >= 4 stages. The all-rounds series is
    # survivorship-confounded — teams that lock onto a top combo kill the
    # boss sooner and exit the sample, so later stages are enriched in
    # struggling teams. Holding the cohort fixed isolates within-team change.
    last_stage = {}
    for r in rows:
        last_stage[r["record_idx"]] = max(last_stage.get(r["record_idx"], 0),
                                          r["stage"])
    cohort_ids = {ri for ri, ls in last_stage.items() if ls >= 3}
    cohort_stages = [s for s in stages if s <= 3]
    cohort_per_stage = {
        s: np.array([r["rank"] for r in rows
                     if r["stage"] == s and r["record_idx"] in cohort_ids])
        for s in cohort_stages}

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # ── Panel A: mean rank vs stage ──
    xs = [s + 1 for s in stages]
    means = [per_stage[s].mean() for s in stages]
    cis = [boot_ci(per_stage[s]) for s in stages]
    yerr = np.array([[m - lo, hi - m] for m, (lo, hi) in zip(means, cis)]).T
    ax_a.errorbar(xs, means, yerr=yerr, fmt="o-", color="#2c3e50",
                  markersize=6, linewidth=2, capsize=4, zorder=3,
                  label="all rounds")

    c_xs = [s + 1 for s in cohort_stages]
    c_means = [cohort_per_stage[s].mean() for s in cohort_stages]
    c_cis = [boot_ci(cohort_per_stage[s]) for s in cohort_stages]
    c_yerr = np.array([[m - lo, hi - m]
                       for m, (lo, hi) in zip(c_means, c_cis)]).T
    ax_a.errorbar(c_xs, c_means, yerr=c_yerr, fmt="s--", color="#e67e22",
                  markersize=5, linewidth=1.6, capsize=3, zorder=2,
                  alpha=0.9,
                  label=f"fixed cohort, ≥4 stages (n={len(cohort_ids)})")
    ax_a.axhline(14, color="#888", linestyle="--", linewidth=1.2, zorder=1)
    ax_a.text(4.95, 14.3, "chance (rank 14)", fontsize=9, color="#888",
              ha="right")
    for x, s in zip(xs, stages):
        ax_a.text(x, 15.6, f"n={len(per_stage[s])}",
                  ha="center", va="bottom", fontsize=8, color="#555")
    ax_a.set_xlabel("Stage", fontsize=11)
    ax_a.set_ylabel("Mean value-rank of played combo\n(1 = optimal, 27 = worst)",
                    fontsize=10)
    ax_a.set_xticks(xs)
    ax_a.set_ylim(4, 16)
    ax_a.invert_yaxis()           # up = better
    ax_a.set_title("A    Value-rank of the played combo, by stage",
                   fontsize=11, loc="left")
    ax_a.legend(fontsize=9, loc="upper right", frameon=False)
    ax_a.spines[["top", "right"]].set_visible(False)

    # ── Panel B: top-1 / top-5 fraction vs stage ──
    for k, color, marker in [(1, "#8e44ad", "o"), (5, "#1abc9c", "s")]:
        fracs, los, his = [], [], []
        for s in stages:
            hits = (per_stage[s] <= k).astype(float)
            fracs.append(hits.mean())
            lo, hi = boot_ci(hits)
            los.append(fracs[-1] - lo)
            his.append(hi - fracs[-1])
        ax_b.errorbar(xs, fracs, yerr=[los, his], fmt=f"{marker}-",
                      color=color, markersize=6, linewidth=2, capsize=4,
                      zorder=3, label=f"top-{k}")
        ax_b.axhline(k / 27, color=color, linestyle=":", linewidth=1.1,
                     alpha=0.6, zorder=1)
        ax_b.text(5.05, k / 27, f"chance {k}/27", fontsize=8, color=color,
                  va="center")
    ax_b.set_xlabel("Stage", fontsize=11)
    ax_b.set_ylabel("Fraction of team-stages", fontsize=10)
    ax_b.set_xticks(xs)
    ax_b.set_xlim(0.7, 5.9)
    ax_b.set_ylim(0, 0.72)
    ax_b.set_title("B    Fraction of team-stages in the top-K, by stage",
                   fontsize=11, loc="left")
    ax_b.legend(fontsize=9, loc="upper left", frameon=False)
    ax_b.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[task2] wrote {OUT_PATH}")

    # Console + markdown summary
    lines = ["# Task 2 — temporal convergence (value-rank of played combo)",
             "",
             f"{len(rows)} team-stage observations, {len(records)} clean "
             "human team-rounds. Rank 1 = value-optimal of 27; chance = 14.",
             "",
             "| Stage | N | Mean rank [95% CI] | Top-1 | Top-5 |",
             "|------:|--:|--------------------|------:|------:|"]
    for s in stages:
        v = per_stage[s]
        lo, hi = boot_ci(v)
        lines.append(f"| {s+1} | {len(v)} | {v.mean():.2f} "
                     f"[{lo:.2f}, {hi:.2f}] | {(v <= 1).mean():.1%} "
                     f"| {(v <= 5).mean():.1%} |")
    lines += [
        "",
        f"## Fixed cohort (rounds lasting ≥4 stages, n={len(cohort_ids)})",
        "",
        "The all-rounds series is survivorship-confounded: teams that lock "
        "onto a top combo kill the boss sooner and leave the sample, so "
        "later stages over-represent struggling teams.",
        "",
        "| Stage | N | Mean rank [95% CI] | Top-1 | Top-5 |",
        "|------:|--:|--------------------|------:|------:|"]
    for s in cohort_stages:
        v = cohort_per_stage[s]
        lo, hi = boot_ci(v)
        lines.append(f"| {s+1} | {len(v)} | {v.mean():.2f} "
                     f"[{lo:.2f}, {hi:.2f}] | {(v <= 1).mean():.1%} "
                     f"| {(v <= 5).mean():.1%} |")
    lines += [
        "",
        "## Reading",
        "",
        "Teams sit **well above chance from stage 1** (mean rank ≈ 9.3 vs "
        "14; top-5 ≈ 47% vs 18.5%) but the per-stage trend is flat-to-"
        "slightly-worse, in both the all-rounds series and the fixed "
        "cohort. There is no evidence of a not-top-K → top-K shift across "
        "stages on this metric: good coordination is mostly present from "
        "the start (stat-driven priors), and rounds that lock onto top "
        "combos end early by winning. Note normalized optimality (05-28 "
        "topk summary) *does* rise across stages (0.60 → 0.78), so 'value "
        "of what teams play, relative to what's attainable' improves even "
        "though the rank of the chosen combo does not."]
    summary = "\n".join(lines)
    (SCRIPT_DIR / "temporal_convergence.md").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
