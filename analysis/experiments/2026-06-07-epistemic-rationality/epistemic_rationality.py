"""2026-06-07 epistemic rationality — two paper figures (PNAS style).

Question: are people forming/updating their beliefs in a Bayesian way?

For every human inference report (observer i at stage s >= 2 guessing
teammate j's stage s-1 role), the fitted Stage-1 Bayesian observer
(tau_prior, epsilon, memory drift from the 05-25 full pipeline — the
Stage-1 source of truth that 05-28-paper-figures also consumes; the
05-12 fit used by 06-05's common.py is byte-identical, asserted at load)
produces a posterior marginal over j's role from the same observed
actions.

Figure 1 (accuracy_by_game) — correctness, model-free + learning:
  Inference accuracy vs the teammate's TRUE previous-stage role, by the
  GAME NUMBER (round 1-8 of the session). Three lines:
    * human reports (black) — the data;
    * Bayesian observer, sampling readout (accent) — the model's mean
      posterior mass on the true role, i.e. the expected accuracy of an
      agent that samples its report from the posterior. This is the
      comparison that matters given Figure 2 shows probability matching;
    * Bayesian observer, MAP readout (grey, thin) — posterior-mode hit
      rate, a ceiling reference.
  Cluster-bootstrap 95% CIs (resampling team-rounds); chance at 1/3.

Figure 2 (calibration) — correlation, model-based:
  Every report crossed with each of the 3 roles (9,312 pairs):
  x = model posterior probability of that role, y = 1 if the human
  reported it. Decile-binned mean(y) at the bin's mean x with cluster-
  bootstrap 95% CIs, identity line, labeled bin-occupancy histogram.
  Pearson r on the raw pairs (with bootstrap 95% CI) on the figure.

All CIs are percentile cluster bootstraps over team-rounds — reports
within a team-round share evidence and observers, so resampling
individual reports would understate uncertainty.

Output: PNAS single-column figures (3.42 in wide), .png at 300 dpi and
.pdf, into "stuff to incorporate/". No in-figure titles or panel letters
(LaTeX adds those).

Run from analysis/:
    uv run python experiments/2026-06-07-epistemic-rationality/epistemic_rationality.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
V2_DIR = SCRIPT_DIR.parent / "2026-06-05-paper-figures-v2"
sys.path.insert(0, str(V2_DIR))

from common import (  # noqa: E402
    compute_posteriors, load_clean_human_teams, load_stage1, prepare_team,
    target_marginal,
)
from shared.constants import ROLE_NAMES  # noqa: E402

# Stage-1 source of truth: the 05-25 full pipeline (re-fit from scratch on
# the 5-export scope; what 05-28-paper-figures consumes). common.py loads
# the 05-12 fit — the two are byte-identical, but we read the 05-25 file
# and assert agreement so any future re-fit fails loudly here.
FULL_PIPELINE_STAGE1 = (SCRIPT_DIR.parent / "2026-05-25-full-pipeline"
                        / "stage1_inference" / "best_inference_params.json")


def load_stage1_canonical():
    import json
    with open(FULL_PIPELINE_STAGE1) as f:
        s1_canon = json.load(f)
    s1, strat = load_stage1()
    for k in ("tau_prior", "epsilon", "memory_strategy"):
        assert s1_canon[k] == s1[k], (
            f"Stage-1 param mismatch on '{k}': 05-25 full pipeline has "
            f"{s1_canon[k]!r}, common.py (05-12) has {s1[k]!r} — "
            f"re-point common.py or update this experiment.")
    return s1_canon, strat

OUT_DIR = SCRIPT_DIR / "stuff to incorporate"
OUT_DIR.mkdir(exist_ok=True)
OUT_MD = SCRIPT_DIR / "summary.md"

N_BOOT = 10_000
N_BINS = 10
SEED = 0

# PNAS single column: 3.42 in wide. Same width + font sizes for both
# figures so they stack cleanly as LaTeX subfigures.
FIG_W = 3.42
FIG_H = 2.45

HUMAN_COLOR = "#000000"     # human data: black solid
ACCENT_COLOR = "#8e44ad"    # Bayesian sampling readout (project purple)
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
        print(f"[epistemic] wrote {path}")


# ──────────────────────────────────────────────────────────────────────
# Data: one row per inference report
# ──────────────────────────────────────────────────────────────────────

def collect_reports():
    """One row per report: cluster id, game, model marginal, guess, truth."""
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
                "round_number": int(key[2]),   # game 1-8 within the session
                "observer": obs,
                "stage": si,             # 0-indexed; report made AT stage si
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
    print(f"[epistemic] {len(rows)} reports from "
          f"{len(key_to_idx)} team-rounds")
    return rows, meta


# ──────────────────────────────────────────────────────────────────────
# Cluster bootstrap helpers
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
        # Bins empty in the full data are all-NaN across replicates too.
        warnings.simplefilter("ignore", RuntimeWarning)
        bin_ci = np.nanpercentile(boot_bins, [2.5, 97.5], axis=0)
    return r_ci, bin_ci


# ──────────────────────────────────────────────────────────────────────
# Figure 1 — accuracy by game number (model-free + learning)
# ──────────────────────────────────────────────────────────────────────

def fig_accuracy_by_game(rows):
    cluster = np.array([r["cluster"] for r in rows])
    game = np.array([r["round_number"] for r in rows])    # 1-8
    human_ok = np.array([r["guessed"] == r["true_prev"] for r in rows],
                        dtype=float)
    model_ok = np.array([int(np.argmax(r["marginal"])) == r["true_prev"]
                         for r in rows], dtype=float)
    # Sampling readout: posterior mass on the TRUE previous role = the
    # expected accuracy of an agent sampling its report from the posterior.
    samp_acc = np.array([r["marginal"][r["true_prev"]] for r in rows],
                        dtype=float)

    games = sorted(np.unique(game))
    out = []  # (game, n, h_acc, h_ci, m_acc, m_ci, s_acc, s_ci)
    for k, g in enumerate(games):
        m = game == g
        sub_cluster = np.unique(cluster[m], return_inverse=True)[1]
        h_ci = boot_mean_ci(sub_cluster, human_ok[m], seed=SEED + 10 + k)
        m_ci = boot_mean_ci(sub_cluster, model_ok[m], seed=SEED + 40 + k)
        s_ci = boot_mean_ci(sub_cluster, samp_acc[m], seed=SEED + 100 + k)
        out.append((g, int(m.sum()), human_ok[m].mean(), h_ci,
                    model_ok[m].mean(), m_ci, samp_acc[m].mean(), s_ci))

    # Learning trend: cluster-bootstrapped slope of accuracy on game number.
    h_slope, h_slope_ci = boot_slope_ci(cluster, game.astype(float), human_ok,
                                        seed=SEED + 70)
    m_slope, m_slope_ci = boot_slope_ci(cluster, game.astype(float), model_ok,
                                        seed=SEED + 71)

    # Overall sampling readout + paired human-minus-sampling difference.
    samp_overall_ci = boot_mean_ci(cluster, samp_acc, seed=SEED + 120)
    diff_ci = boot_mean_ci(cluster, human_ok - samp_acc, seed=SEED + 121)
    diff_mean = float(np.mean(human_ok - samp_acc))

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    xs = [o[0] for o in out]

    # MAP readout: ceiling reference, thin grey dashed.
    map_vals = np.array([o[4] for o in out])
    map_cis = np.array([o[5] for o in out])
    ax.errorbar(xs, map_vals, yerr=np.abs(map_cis.T - map_vals),
                fmt="s--", color=REF_COLOR, markersize=2, linewidth=0.8,
                capsize=1.2, elinewidth=0.6,
                label="Bayesian observer (MAP readout)", zorder=2)

    # Sampling readout: the comparison that matters.
    s_vals = np.array([o[6] for o in out])
    s_cis = np.array([o[7] for o in out])
    ax.errorbar(xs, s_vals, yerr=np.abs(s_cis.T - s_vals),
                fmt="D-", color=ACCENT_COLOR, markersize=2.6, linewidth=1.1,
                capsize=1.5, elinewidth=0.7,
                label="Bayesian observer (sampling readout)", zorder=3)

    # Human reports.
    h_vals = np.array([o[2] for o in out])
    h_cis = np.array([o[3] for o in out])
    ax.errorbar(xs, h_vals, yerr=np.abs(h_cis.T - h_vals),
                fmt="o-", color=HUMAN_COLOR, markersize=2.6, linewidth=1.1,
                capsize=1.5, elinewidth=0.7,
                label="human reports", zorder=4)

    ax.axhline(1 / 3, color=REF_COLOR, linestyle=":", linewidth=0.7, zorder=1)
    ax.text(xs[-1] + 0.42, 1 / 3 + 0.015, "chance", fontsize=6,
            color=REF_COLOR, va="bottom", ha="right")
    for o in out:
        ax.text(o[0], 0.015, f"{o[1]}", ha="center", va="bottom",
                fontsize=6, color="#777")
    ax.text(xs[0] - 0.42, 0.015, "n =", ha="right", va="bottom", fontsize=6,
            color="#777")

    ax.set_xlabel("Game number within session")
    ax.set_ylabel("Inference accuracy")
    ax.set_xticks(xs)
    ax.set_xlim(xs[0] - 0.45, xs[-1] + 0.45)
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(lbl) for lbl in (
        "human reports", "Bayesian observer (sampling readout)",
        "Bayesian observer (MAP readout)")]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="lower left", bbox_to_anchor=(0.01, 0.06), frameon=False,
              handlelength=1.6, borderpad=0.2, labelspacing=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=0.3)
    savefig(fig, "accuracy_by_game")
    plt.close(fig)

    trends = {"human": (h_slope, h_slope_ci), "model": (m_slope, m_slope_ci)}
    sampling = {
        "overall": float(samp_acc.mean()), "overall_ci": samp_overall_ci,
        "diff_mean": diff_mean, "diff_ci": diff_ci,
    }
    return (out, float(human_ok.mean()), float(model_ok.mean()), trends,
            sampling)


# ──────────────────────────────────────────────────────────────────────
# Figure 2 — calibration against the posterior (model-based)
# ──────────────────────────────────────────────────────────────────────

def fig_calibration(rows):
    # Every report crossed with each role: 3,104 x 3 = 9,312 pairs.
    cluster = np.repeat([r["cluster"] for r in rows], 3)
    x = np.concatenate([r["marginal"] for r in rows])
    y = np.array([float(r["guessed"] == ri) for r in rows for ri in range(3)])

    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    r_raw = float(np.corrcoef(x, y)[0, 1])
    r_ci, bin_ci = boot_calibration(cluster, x, y, edges)
    mx, my, ns = bin_stats(x, y, edges)
    ok = ~np.isnan(my)
    r_binned = float(np.corrcoef(mx[ok], my[ok])[0, 1])

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

    ax.text(0.97, 0.10,
            f"r = {r_raw:.2f}, 95% CI [{r_ci[0]:.2f}, {r_ci[1]:.2f}]\n"
            f"{len(y):,} (report, role) pairs",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6)

    ax.set_xlabel("Bayesian model posterior probability of role")
    ax.set_ylabel("Fraction of reports naming role")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.155, 1.04)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.legend(loc="upper left", frameon=False, handlelength=1.6,
              borderpad=0.2, labelspacing=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=0.3)
    savefig(fig, "calibration")
    plt.close(fig)

    # Per-role raw-pair correlations.
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
        "mx": mx, "my": my, "ns": ns, "bin_ci": bin_ci, "ok": ok,
        "per_role": per_role, "n_pairs": len(y),
    }


# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────

def write_summary(meta, acc_rows, h_acc, m_acc, trends, sampling, cal):
    s1 = meta["stage1"]
    lines = [
        "# Epistemic rationality — human vs Bayesian-model inferences",
        "",
        f"Scope: {meta['n_team_rounds']} clean human team-rounds (5 exports), "
        f"{meta['n_reports']:,} inference reports. Stage-1 params: "
        f"τ_prior = {s1['tau_prior']:.4f}, ε = {s1['epsilon']:.4f}, "
        f"memory = `{s1['memory_strategy']}`. All CIs are percentile cluster "
        f"bootstraps over team-rounds ({N_BOOT:,} resamples).",
        "",
        "## Figure 1 — inference accuracy by game number (vs true previous role)",
        "",
        f"Overall: human **{h_acc:.3f}**, Bayesian sampling readout "
        f"**{sampling['overall']:.3f}** "
        f"[{sampling['overall_ci'][0]:.3f}, {sampling['overall_ci'][1]:.3f}], "
        f"MAP readout **{m_acc:.3f}**, chance 1/3. "
        f"Human − sampling paired difference: {sampling['diff_mean']:+.3f} "
        f"[{sampling['diff_ci'][0]:+.3f}, {sampling['diff_ci'][1]:+.3f}]. "
        f"Game number = round 1-8 of the session; each participant has a "
        f"unique human/bot round ordering, so each game number samples a "
        f"different subset of teams.",
        "",
        "| Game | n reports | Human acc | 95% CI | Sampling acc | 95% CI "
        "| MAP acc | 95% CI |",
        "|---|--:|--:|---|--:|---|--:|---|",
    ]
    for g, n, ha, hci, ma, mci, sa, sci in acc_rows:
        lines.append(f"| {g} | {n} | {ha:.3f} | [{hci[0]:.3f}, {hci[1]:.3f}] "
                     f"| {sa:.3f} | [{sci[0]:.3f}, {sci[1]:.3f}] "
                     f"| {ma:.3f} | [{mci[0]:.3f}, {mci[1]:.3f}] |")
    hs, hci = trends["human"]
    ms, mci = trends["model"]
    lines += [
        "",
        f"Learning trend (OLS slope, accuracy per game): human "
        f"**{hs:+.4f}** [{hci[0]:+.4f}, {hci[1]:+.4f}], MAP readout "
        f"{ms:+.4f} [{mci[0]:+.4f}, {mci[1]:+.4f}].",
    ]
    lines += [
        "",
        "## Figure 2 — calibration of human reports against the posterior",
        "",
        "Every report × each role: x = posterior probability of the role, "
        "y = 1 if the human reported it.",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Pearson r (raw pairs) | **{cal['r_raw']:.3f}** "
        f"[{cal['r_ci'][0]:.3f}, {cal['r_ci'][1]:.3f}] "
        f"({cal['n_pairs']:,} pairs) |",
        f"| Pearson r (binned means) | **{cal['r_binned']:.3f}** |",
    ]
    for name, rr, rci in cal["per_role"]:
        lines.append(f"| Pearson r — {name} only (raw pairs) | {rr:.3f} "
                     f"[{rci[0]:.3f}, {rci[1]:.3f}] |")
    lines += [
        "",
        "| Posterior bin | n | mean x | Report frequency | 95% CI |",
        "|---------------|--:|-------:|-----------------:|--------|",
    ]
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    for b in range(N_BINS):
        if not cal["ok"][b]:
            continue
        lines.append(
            f"| {edges[b]:.1f}–{edges[b + 1]:.1f} | {cal['ns'][b]} "
            f"| {cal['mx'][b]:.3f} | {cal['my'][b]:.3f} "
            f"| [{cal['bin_ci'][0][b]:.3f}, {cal['bin_ci'][1][b]:.3f}] |")
    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print(f"[epistemic] wrote {OUT_MD}")


def main():
    rows, meta = collect_reports()

    acc_rows, h_acc, m_acc, trends, sampling = fig_accuracy_by_game(rows)
    print(f"\n  overall accuracy: human = {h_acc:.3f}, "
          f"sampling readout = {sampling['overall']:.3f} "
          f"[{sampling['overall_ci'][0]:.3f}, "
          f"{sampling['overall_ci'][1]:.3f}], "
          f"MAP readout = {m_acc:.3f}")
    print(f"  human - sampling (paired): {sampling['diff_mean']:+.3f} "
          f"[{sampling['diff_ci'][0]:+.3f}, {sampling['diff_ci'][1]:+.3f}]")
    for g, n, ha, hci, ma, mci, sa, sci in acc_rows:
        print(f"  game {g}: n={n:4d}  human {ha:.3f} "
              f"[{hci[0]:.3f}, {hci[1]:.3f}]  sampling {sa:.3f} "
              f"[{sci[0]:.3f}, {sci[1]:.3f}]  MAP {ma:.3f} "
              f"[{mci[0]:.3f}, {mci[1]:.3f}]")
    for who in ("human", "model"):
        s, ci = trends[who]
        print(f"  {who} trend: {s:+.4f}/game  "
              f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")

    cal = fig_calibration(rows)
    print(f"\n  raw-pair r = {cal['r_raw']:.3f}  "
          f"95% CI [{cal['r_ci'][0]:.3f}, {cal['r_ci'][1]:.3f}]")
    print(f"  binned-means r = {cal['r_binned']:.3f}")
    for name, rr, rci in cal["per_role"]:
        print(f"  {name:8s} r = {rr:.3f}  95% CI [{rci[0]:.3f}, {rci[1]:.3f}]")

    write_summary(meta, acc_rows, h_acc, m_acc, trends, sampling, cal)


if __name__ == "__main__":
    main()
