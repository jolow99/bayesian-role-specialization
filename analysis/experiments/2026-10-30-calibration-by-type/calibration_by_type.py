"""2026-10-30 — Fig. 2 (calibration) for the population, Bayesians and non-Bayesians,
against the 10-27 two-type classifier's posterior.

x = the posterior the classifier scores reports against: P_N, the Bayesian
posterior at the end of the previous stage before that stage's drift toward
the prior (drift β carried between stages), at the classifier's parameters
(τ_prior = 4.05, ε = 0.246, β = 0.726); the paper's Fig. 2 uses the 05-25
Stage-1 posterior. Binning, cluster bootstrap and layout are the paper's
(2026-06-16 calibration_revision.py).

Run from analysis/:
    uv run python experiments/2026-10-30-calibration-by-type/calibration_by_type.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXP = SCRIPT_DIR.parent
CLF_DIR = EXP / "2026-10-27-inference-soft-classifier"
TYPES_CSV = CLF_DIR / "participant_types.csv"
THRESHOLD = 0.5
sys.path.insert(0, str(EXP / "2026-06-16-pnas-figure-revisions"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses look the module up here
    spec.loader.exec_module(mod)
    return mod


import calibration_revision as cr  # noqa: E402  (puts 06-05 common.py on sys.path)

clf_pipeline = _load("clf_pipeline", CLF_DIR / "pipeline.py")
from common import load_clean_human_teams, prepare_team, target_marginal  # noqa: E402

cr.OUT_DIR = SCRIPT_DIR / "figures"
cr.OUT_DIR.mkdir(exist_ok=True)
OUT_MD = SCRIPT_DIR / "calibration_summary.md"


def load_types():
    with open(TYPES_CSV) as f:
        return {r["participant_id"]: float(r["p_prior_type"]) for r in csv.DictReader(f)}


def relabel_clusters(rows):
    """Clusters must be 0..n-1 for the bootstrap after subsetting."""
    remap = {c: i for i, c in enumerate(sorted({r["cluster"] for r in rows}))}
    return [{**r, "cluster": remap[r["cluster"]]} for r in rows]


def run(rows, name):
    """Paper calibration figure on `rows`, saved as `name`; returns its statistics."""
    orig = cr.savefig
    cr.savefig = lambda fig, _name: orig(fig, name)
    try:
        cal = cr.fig_calibration(relabel_clusters(rows))
    finally:
        cr.savefig = orig
    ok = cal["ok"]
    return {
        "n_participants": len({r["participant"] for r in rows}),
        "n_reports": len(rows),
        "r_binned": cal["r_binned"],
        "r_by_bins": {nb: cal["r_by_bins"][nb][0] for nb in cr.BIN_COUNTS},
        "r_raw": cal["r_raw"], "r_ci": cal["r_ci"],
        "mad": float(np.mean(np.abs(cal["my"][ok] - cal["mx"][ok]))),
        "acc": float(np.mean([r["guessed"] == r["true_prev"] for r in rows])),
        "sampler_acc": float(np.mean([r["marginal"][r["true_prev"]] for r in rows])),
        "bins": [(b, cal["ns"][b], cal["mx"][b], cal["my"][b],
                  cal["bin_ci"][0][b], cal["bin_ci"][1][b])
                 for b in range(cr.N_BINS) if ok[b]],
        "cal": cal,
    }


PANEL_TITLE = {"All": "Population", "Bayesian": "Bayesians", "Non-Bayesian": "Non-Bayesians"}


def combined_figure(res, name="R1_calibration_by_type"):
    """The three groups side by side, same marks as the single figures (from their statistics)."""
    import matplotlib.pyplot as plt

    edges = np.linspace(0.0, 1.0, cr.N_BINS + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.45), sharey=True)
    for i, (ax, (g, s)) in enumerate(zip(axes, res.items())):
        cal = s["cal"]
        ok = cal["ok"]
        ident, = ax.plot([0, 1], [0, 1], "--", color=cr.REF_COLOR, linewidth=0.8, zorder=1)
        yerr = np.abs(cal["bin_ci"][:, ok] - cal["my"][ok])
        pts = ax.errorbar(cal["mx"][ok], cal["my"][ok], yerr=yerr, fmt="o-", color=cr.HUMAN_COLOR,
                          markersize=2.6, linewidth=1.1, capsize=1.5, elinewidth=0.7, zorder=4)
        h = cal["ns"] / cal["ns"].max() * 0.10
        ax.bar(centers, h, width=0.094, bottom=-0.145, color="#c2cdd6", alpha=0.8, zorder=2)
        ax.axhline(-0.145, color="#c2cdd6", linewidth=0.6)
        for c, cnt, hh in zip(centers, cal["ns"], h):
            if cnt:
                ax.text(c, -0.14 + hh + 0.012, f"{cnt}", ha="center", va="bottom",
                        fontsize=4.5, color="#777")
        rb = cal["r_by_bins"]
        ax.text(0.03, 0.97, f"$r$ = {cal['r_binned']:.2f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=8, fontweight="bold")
        ax.text(0.03, 0.86,
                f"binned (decile) points\n"
                f"{rb[5][0]:.2f}/{rb[10][0]:.2f}/{rb[20][0]:.2f} at 5/10/20 bins\n"
                f"raw pairs: $r$ = {cal['r_raw']:.2f} [{cal['r_ci'][0]:.2f}, {cal['r_ci'][1]:.2f}]",
                transform=ax.transAxes, ha="left", va="top", fontsize=5, color="#555",
                linespacing=1.4)
        ax.set_title(f"{PANEL_TITLE[g]} (n = {s['n_participants']}, "
                     f"{s['n_reports']:,} reports)", fontsize=7, pad=4)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.155, 1.04)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.spines[["top", "right"]].set_visible(False)
        if i == 0:
            ax.set_ylabel("Fraction of reports naming role")
    fig.supxlabel("Bayesian model posterior probability of role", fontsize=7, y=0.04)
    fig.legend([ident, pts], ["identity (probability matching)", "binned mean (95% CI)"],
               loc="lower center", ncols=2, frameon=False, fontsize=6,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(pad=0.3, w_pad=1.0, rect=(0, 0.04, 1, 1))
    cr.savefig(fig, name)
    plt.close(fig)


def collect_reports(clf):
    strat = clf_pipeline.MemoryStrategy(f"drift_prior_{clf['beta']:.3f}", "drift_prior", clf["beta"])
    key_to_idx, rows = {}, []
    for key, team_prs in load_clean_human_teams().items():
        data = prepare_team(team_prs)
        part = {pr.player_id: pr.participant_id for pr in team_prs}
        posteriors = clf_pipeline.compute_posteriors(data, clf["tau_prior"], clf["epsilon"], strat)
        for obs, si, target_pos, guessed, true_prev in data["queries"]:
            if si >= len(posteriors):
                continue
            rows.append({"cluster": key_to_idx.setdefault(key, len(key_to_idx)),
                         "participant": part[obs],
                         "marginal": target_marginal(posteriors[si], target_pos),
                         "guessed": guessed, "true_prev": true_prev})
    return rows


def main():
    clf = json.load(open(CLF_DIR / "best_params.json"))
    types = load_types()
    rows = collect_reports(clf)
    assert len(rows) == 3104, len(rows)

    # sanity: these posteriors reproduce the classifier's Bayesian-type log-likelihoods
    log_b = {}
    for r in rows:
        p = (1 - clf_pipeline.INFERENCE_LAPSE) * r["marginal"][r["guessed"]] + clf_pipeline.INFERENCE_LAPSE / 3
        log_b[r["participant"]] = log_b.get(r["participant"], 0.0) + float(np.log(p))
    with open(TYPES_CSV) as f:
        ref = {r["participant_id"]: float(r["log_B"]) for r in csv.DictReader(f)}
    worst = max(abs(log_b[p] - ref[p]) for p in ref)
    assert worst < 1e-3, worst
    print(f"posterior reproduces the classifier's log B_i (max |diff| = {worst:.1e})")

    groups = {
        "All": (rows, "R1_calibration_all"),
        "Bayesian": ([r for r in rows if types[r["participant"]] < THRESHOLD],
                     "R1_calibration_bayesian"),
        "Non-Bayesian": ([r for r in rows if types[r["participant"]] >= THRESHOLD],
                         "R1_calibration_nonbayesian"),
    }
    res = {g: run(rs, name) for g, (rs, name) in groups.items()}
    combined_figure(res)
    L = ["# Fig. 2 (calibration) by inference type, classifier posterior", "",
         f"x = classifier P_N (τ_prior = {clf['tau_prior']:.3f}, ε = {clf['epsilon']:.3f}, "
         f"β = {clf['beta']:.3f}). Bayesian = p_prior_type < {THRESHOLD}.", "",
         "| Group | participants | reports | binned r (5/10/20 bins) | raw-pair r [95% CI] "
         "| mean abs. bin deviation | human acc. | sampler acc. |",
         "|---|--:|--:|---|---|--:|--:|--:|"]
    for g, s in res.items():
        rb = "/".join(f"{s['r_by_bins'][nb]:.2f}" for nb in cr.BIN_COUNTS)
        L.append(f"| {g} | {s['n_participants']} | {s['n_reports']:,} | **{s['r_binned']:.2f}** ({rb}) "
                 f"| {s['r_raw']:.2f} [{s['r_ci'][0]:.2f}, {s['r_ci'][1]:.2f}] | {s['mad']:.3f} "
                 f"| {s['acc']:.1%} | {s['sampler_acc']:.1%} |")
    for g, s in res.items():
        L += ["", f"## {g}: per-bin detail", "",
              "| Posterior bin | n | mean x | Report frequency | 95% CI |", "|---|--:|--:|--:|---|"]
        for b, n, mx, my, lo, hi in s["bins"]:
            L.append(f"| {b / cr.N_BINS:.1f}-{(b + 1) / cr.N_BINS:.1f} | {n} | {mx:.3f} "
                     f"| {my:.3f} | [{lo:.3f}, {hi:.3f}] |")
    OUT_MD.write_text("\n".join(L) + "\n")
    print("\n".join(L[:8]))


if __name__ == "__main__":
    main()
