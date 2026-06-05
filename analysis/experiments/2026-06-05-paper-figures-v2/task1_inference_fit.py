"""Task 1 — inference-fit figure (figures/inference_fit.png).

Panel A: bin every human inference report by the role the human guessed
(Fighter / Tank / Medic). Within each bin, the mean model posterior
probability (fitted Stage-1 params) on each of the three roles at the same
(game, round, stage, target) point. 3×3 grouped bars, paper role colors,
n annotated per bin. A good model puts the diagonal on top.

Panel B: where ≥2 observers reported on the same (game, round, stage,
target) evidence, scatter of human selection frequency for a role vs the
model posterior on that role, with binned means and the identity line.
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    FIGURES_DIR, ROLE_COLORS_IDX, compute_posteriors, load_clean_human_teams,
    load_stage1, prepare_team, target_marginal,
)
from shared.constants import ROLE_NAMES

OUT_PATH = FIGURES_DIR / "inference_fit.png"


def collect_reports():
    """One row per human inference report, with the model marginal attached."""
    s1, strat = load_stage1()
    teams = load_clean_human_teams()

    rows = []
    for key, team_prs in teams.items():
        data = prepare_team(team_prs)
        posteriors = compute_posteriors(data, s1["tau_prior"], s1["epsilon"],
                                        strat)
        for obs, si, target_pos, guessed, true_prev in data["queries"]:
            if si >= len(posteriors):
                continue
            marg = target_marginal(posteriors[si], target_pos)
            rows.append({
                "key": key,             # (export, game_id, round_number)
                "observer": obs,
                "stage": si,
                "target": target_pos,
                "guessed": guessed,
                "true_prev": true_prev,
                "marginal": marg,
            })
    print(f"[task1] {len(rows)} inference reports")
    return rows


def plot(rows):
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(12.5, 5), gridspec_kw={"width_ratios": [1.15, 1]})

    # ── Panel A: grouped bars ──
    width = 0.26
    for bin_role in range(3):
        sub = [r["marginal"] for r in rows if r["guessed"] == bin_role]
        n = len(sub)
        mean_marg = np.mean(sub, axis=0) if n else np.zeros(3)
        sem = (np.std(sub, axis=0, ddof=1) / np.sqrt(n)) if n > 1 else np.zeros(3)
        for model_role in range(3):
            x = bin_role + (model_role - 1) * width
            ax_a.bar(x, mean_marg[model_role], width * 0.92,
                     color=ROLE_COLORS_IDX[model_role],
                     yerr=sem[model_role], capsize=2,
                     error_kw={"linewidth": 0.9, "alpha": 0.7},
                     edgecolor=("black" if model_role == bin_role else "none"),
                     linewidth=1.4,
                     zorder=3)
            ax_a.text(x, mean_marg[model_role] + 0.025,
                      f"{mean_marg[model_role]:.2f}", ha="center",
                      va="bottom", fontsize=8,
                      fontweight="bold" if model_role == bin_role else "normal")
        ax_a.text(bin_role, -0.115, f"n = {n}", ha="center", va="top",
                  fontsize=9, color="#444", transform=ax_a.get_xaxis_transform())

    ax_a.axhline(1 / 3, color="#888", linestyle="--", linewidth=1, zorder=1)
    ax_a.text(2.42, 1 / 3 + 0.008, "1/3", fontsize=8, color="#888")
    ax_a.set_xticks(range(3))
    ax_a.set_xticklabels([f"guessed {ROLE_NAMES[r]}" for r in range(3)],
                         fontsize=10)
    ax_a.set_ylabel("Mean model posterior probability", fontsize=10)
    ax_a.set_ylim(0, 0.85)
    ax_a.set_title("A    Model posterior, binned by the role the\n"
                   "human reported for that teammate", fontsize=11, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=ROLE_COLORS_IDX[r])
               for r in range(3)]
    ax_a.legend(handles, [f"P({ROLE_NAMES[r]})" for r in range(3)],
                fontsize=9, loc="upper left", frameon=False,
                title="Model posterior on", title_fontsize=9)
    ax_a.spines[["top", "right"]].set_visible(False)

    # ── Panel B: selection frequency vs posterior (shared evidence) ──
    groups = defaultdict(list)
    for r in rows:
        groups[(r["key"], r["stage"], r["target"])].append(r)
    xs, ys, cs = [], [], []
    n_groups = 0
    for grp in groups.values():
        if len(grp) < 2:
            continue
        n_groups += 1
        marg = grp[0]["marginal"]          # same evidence → same posterior
        guesses = [g["guessed"] for g in grp]
        for role in range(3):
            xs.append(marg[role])
            ys.append(np.mean([g == role for g in guesses]))
            cs.append(ROLE_COLORS_IDX[role])
    xs, ys = np.array(xs), np.array(ys)

    rng = np.random.default_rng(0)
    ax_b.scatter(xs, ys + rng.uniform(-0.012, 0.012, len(ys)), s=14, c=cs,
                 alpha=0.25, linewidths=0, zorder=2)
    ax_b.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, zorder=1)

    # Binned means (equal-width posterior bins)
    edges = np.linspace(0, 1, 11)
    centers, means, errs = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (xs >= lo) & (xs < hi) if hi < 1 else (xs >= lo) & (xs <= hi)
        if m.sum() < 5:
            continue
        centers.append((lo + hi) / 2)
        means.append(ys[m].mean())
        errs.append(ys[m].std(ddof=1) / np.sqrt(m.sum()))
    ax_b.errorbar(centers, means, yerr=errs, fmt="o-", color="black",
                  markersize=5, linewidth=1.6, capsize=2.5, zorder=4,
                  label="binned mean ± SEM")

    r_corr = np.corrcoef(xs, ys)[0, 1]
    ax_b.set_xlabel("Model posterior probability of role", fontsize=10)
    ax_b.set_ylabel("Fraction of observers selecting role", fontsize=10)
    ax_b.set_title(f"B    Human selection frequency vs model posterior\n"
                   f"({n_groups} shared-evidence points, r = {r_corr:.2f})",
                   fontsize=11, loc="left")
    ax_b.set_xlim(-0.03, 1.03)
    ax_b.set_ylim(-0.06, 1.06)
    ax_b.legend(fontsize=9, loc="lower right", frameon=False)
    ax_b.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[task1] wrote {OUT_PATH}")

    # Console summary
    for bin_role in range(3):
        sub = [r["marginal"] for r in rows if r["guessed"] == bin_role]
        mm = np.mean(sub, axis=0)
        print(f"  guessed {ROLE_NAMES[bin_role]:8s} (n={len(sub):4d}): "
              f"P(F)={mm[0]:.3f} P(T)={mm[1]:.3f} P(M)={mm[2]:.3f}")


if __name__ == "__main__":
    plot(collect_reports())
