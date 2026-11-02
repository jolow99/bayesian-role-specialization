"""Fig. 2: calibration of inference reports against the classifier's P_N, by type.

x = model probability of a role for the target, y = 1 if the participant
reported that role. Non-updater = P(non-updater) > 0.5. Binning, team-cluster
bootstrap and plot marks lifted from 2026-06-16 calibration_revision.py.

Panels: updaters vs P_N (main text); population vs P_N, non-updaters vs P_N,
non-updaters vs their learned FTM prior, one point per role (appendix).

Run from analysis/:
    uv run python experiments/2026-11-02_ftm-prior-fitting/results/calibration/calibration.py
"""

from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
FITTING_DIR = OUT_DIR.parents[1] / "inference-fitting"
sys.path.insert(0, str(FITTING_DIR))

from pipeline import answer_marginals, load_answers, participant_logprob  # noqa: E402

N_BOOT = 10_000
N_BINS = 10
BIN_COUNTS = (5, 10, 20)
SEED = 0
EDGES = np.linspace(0.0, 1.0, N_BINS + 1)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})


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


def boot_calibration(cluster, x, y, edges):
    """Cluster bootstrap: CI of r over raw pairs and of mean(y) per bin."""
    rng = np.random.default_rng(SEED)
    members = [np.flatnonzero(cluster == c) for c in np.unique(cluster)]
    boot_r = np.empty(N_BOOT)
    boot_bins = np.empty((N_BOOT, len(edges) - 1))
    for b in range(N_BOOT):
        idx = np.concatenate([members[c] for c in rng.integers(0, len(members), len(members))])
        boot_r[b] = np.corrcoef(x[idx], y[idx])[0, 1]
        boot_bins[b] = bin_stats(x[idx], y[idx], edges)[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bin_ci = np.nanpercentile(boot_bins, [2.5, 97.5], axis=0)
    return np.percentile(boot_r, [2.5, 97.5]), bin_ci


def binned_r(x, y, n_bins):
    """Pearson r over the (mean x, mean y) of n_bins equal-width bins."""
    mx, my, _ = bin_stats(x, y, np.linspace(0.0, 1.0, n_bins + 1))
    ok = ~np.isnan(my)
    return float(np.corrcoef(mx[ok], my[ok])[0, 1])


def calibrate(marg, inferred, team, edges=None):
    """Decile bins with binned r by default; custom `edges` (one bin per role) without."""
    x = marg.ravel()
    y = (inferred[:, None] == np.arange(3)).ravel().astype(float)
    r_ci, bin_ci = boot_calibration(np.repeat(team, 3), x, y, EDGES if edges is None else edges)
    mx, my, ns = bin_stats(x, y, EDGES if edges is None else edges)
    ok = ns > 0
    out = {"r_raw": float(np.corrcoef(x, y)[0, 1]), "r_ci": r_ci.tolist(),
           "mad": float(np.mean(np.abs(my[ok] - mx[ok]))),
           "mx": mx, "my": my, "ns": ns, "bin_ci": bin_ci}
    if edges is None:
        out.update(r_binned=binned_r(x, y, N_BINS),
                   r_by_bins={nb: binned_r(x, y, nb) for nb in BIN_COUNTS})
    return out


def panel(ax, cal, title, roles=None, top_left=False):
    """Decile panel; with `roles`, one labelled point per role and no histogram / binned r.
    top_left puts the r text in the upper left, clear of a curve that stays low at high x."""
    ok = cal["ns"] > 0
    centers = (EDGES[:-1] + EDGES[1:]) / 2
    ax.plot([0, 1], [0, 1], "--", color="#999999", linewidth=0.8, zorder=1,
            label="identity (probability matching)")
    ax.errorbar(cal["mx"][ok], cal["my"][ok], yerr=np.abs(cal["bin_ci"][:, ok] - cal["my"][ok]),
                fmt="o" if roles else "o-", color="#000000", markersize=2.6, linewidth=1.1,
                capsize=1.5, elinewidth=0.7, zorder=4, label="binned mean (95% CI)")
    ax.set_title(title, fontsize=7, pad=4)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.155, 1.04)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.spines[["top", "right"]].set_visible(False)
    if roles:
        for r, x, y, dy in zip(roles, cal["mx"], cal["my"], (-9, 5, 5)):
            ax.annotate(r, (x, y), xytext=(5, dy), textcoords="offset points", fontsize=6)
        ax.text(0.97, 0.22,
                f"raw pairs: $r$ = {cal['r_raw']:.2f} [{cal['r_ci'][0]:.2f}, {cal['r_ci'][1]:.2f}]\n"
                f"mean abs. deviation = {cal['mad']:.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=5, color="#555",
                linespacing=1.4)
        return
    h = cal["ns"] / cal["ns"].max() * 0.10
    ax.bar(centers, h, width=0.094, bottom=-0.145, color="#c2cdd6", alpha=0.8, zorder=2)
    ax.axhline(-0.145, color="#c2cdd6", linewidth=0.6)
    for c, n, hh in zip(centers, cal["ns"], h):
        if n:
            ax.text(c, -0.14 + hh + 0.012, f"{n}", ha="center", va="bottom", fontsize=4.5,
                    color="#777")
    rb = cal["r_by_bins"]
    x, ha, (y_r, y_s, va) = (0.03, "left", (0.97, 0.86, "top")) if top_left else \
        (0.97, "right", (0.42, 0.22, "bottom"))
    ax.text(x, y_r, f"$r$ = {cal['r_binned']:.2f}", transform=ax.transAxes,
            ha=ha, va=va, fontsize=9, fontweight="bold")
    ax.text(x, y_s,
            f"binned (decile) points\n{rb[5]:.2f}/{rb[10]:.2f}/{rb[20]:.2f} at 5/10/20 bins\n"
            f"raw pairs: $r$ = {cal['r_raw']:.2f} [{cal['r_ci'][0]:.2f}, {cal['r_ci'][1]:.2f}]\n"
            f"mean abs. bin deviation = {cal['mad']:.3f}",
            transform=ax.transAxes, ha=ha, va=va, fontsize=5, color="#555",
            linespacing=1.4)


def savefig(fig, name):
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)


def main():
    params = json.load(open(FITTING_DIR / "params.json"))
    with open(FITTING_DIR / "participant_types.csv") as f:
        types = {r["participant_id"]: r for r in csv.DictReader(f)}

    teams, answers = load_answers()
    fit_args = (params["tau_prior"], params["epsilon"], params["beta"])
    p_N = answer_marginals(teams, answers, *fit_args)
    p_ftm = np.array([params["p_F"], params["p_T"], params["p_M"]])
    order = np.argsort(p_ftm)  # one bin per role, in increasing prior probability
    ftm_edges = np.r_[0.0, (p_ftm[order][:-1] + p_ftm[order][1:]) / 2, 1.0]
    ftm_roles = ["FTM"[i] for i in order]

    # sanity: P_N reproduces the classifier's per-participant log B_i
    log_B = participant_logprob(p_N[np.arange(len(p_N)), answers["inferred"]], answers)
    ref = np.array([float(types[pid]["log_B"]) for pid in answers["participant_ids"]])
    assert np.abs(log_B - ref).max() < 1e-3, np.abs(log_B - ref).max()

    is_nu = np.array([float(types[pid]["p_non_updater"]) > 0.5
                      for pid in answers["participant_ids"]])[answers["participant"]]
    groups = {"updater_PN": ("Updaters", ~is_nu, p_N),
              "population_PN": ("Population", np.ones_like(is_nu), p_N),
              "non_updater_PN": ("Non-updaters", is_nu, p_N),
              "non_updater_FTM": ("Non-updaters vs. learned FTM prior", is_nu,
                                  np.tile(p_ftm, (len(p_N), 1)))}

    FIG_DIR.mkdir(exist_ok=True)
    res, titles = {}, {}
    for key, (label, m, marg) in groups.items():
        res[key] = calibrate(marg[m], answers["inferred"][m], answers["team"][m],
                             ftm_edges if key == "non_updater_FTM" else None)
        n_part = len(np.unique(answers["participant"][m]))
        titles[key] = f"{label} (n = {n_part}, {m.sum():,} reports)"
        res[key].update(n_participants=n_part, n_reports=int(m.sum()))

    fig, ax = plt.subplots(figsize=(3.42, 2.45))
    panel(ax, res["updater_PN"], titles["updater_PN"])
    ax.set_xlabel("Bayesian model posterior probability of role")
    ax.set_ylabel("Fraction of reports naming role")
    ax.legend(loc="upper left", frameon=False, handlelength=1.6, borderpad=0.2, labelspacing=0.3)
    fig.tight_layout(pad=0.3)
    savefig(fig, "calibration_updater")

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharey=True)
    for ax, key in zip(axes.flat, ["population_PN", "updater_PN", "non_updater_PN",
                                   "non_updater_FTM"]):
        panel(ax, res[key], titles[key], ftm_roles if key == "non_updater_FTM" else None,
              top_left=key == "non_updater_PN")
    for ax in axes[:, 0]:
        ax.set_ylabel("Fraction of reports naming role")
    for ax in axes[0]:
        ax.set_xlabel("Bayesian model posterior probability of role ($P_N$)")
    axes[1, 0].set_xlabel("Bayesian model posterior probability of role ($P_N$)")
    axes[1, 1].set_xlabel("Learned FTM prior probability of role")
    axes[0, 0].legend(loc="upper left", frameon=False, handlelength=1.6, borderpad=0.2,
                      labelspacing=0.3)
    fig.tight_layout(pad=0.3, w_pad=1.5, h_pad=1.5)
    savefig(fig, "calibration_2x2")

    out = {k: {s: v for s, v in r.items() if s not in ("mx", "my", "ns", "bin_ci")}
           for k, r in res.items()}
    with open(OUT_DIR / "calibration.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
