"""Fig. 6: bot-round adaptation by stage, for updaters and for non-updaters (P(non-updater) > 0.5).

Per (bot round, stage): does the human play their deviate-optimal role (the true
optimal), their stat-optimal role, or another? Fractions by stage with 95% CIs from
a cluster bootstrap over participants. Ported from 2026-06-07-bot-adaptation/
adaptation_by_stage.py; checks it reproduces that script's numbers on all participants.

Run from analysis/:
    uv run python experiments/2026-11-02_ftm-prior-fitting/results/bot-adaptation/bot_adaptation.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shared import EXPORTS_DIR
from shared.data_loading import load_all_exports

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
TYPES_CSV = OUT_DIR.parents[1] / "inference-fitting" / "participant_types.csv"
N_BOOT = 10_000
SEED = 0
KEYS = ("is_dev", "is_stat", "is_other")

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})


def load_rows():
    """One row per (clean bot round, stage); index 0 of optimal/deviate roles is the human."""
    prs = load_all_exports(data_dirs=sorted(EXPORTS_DIR.glob("bayesian-role-specialization-*")))
    dropout_games = {pr.game_id for pr in prs if pr.is_dropout}
    rows, n_rounds = [], 0
    for pr in prs:
        rnd = pr.round
        if (pr.game_id in dropout_games or rnd.round_type != "bot"
                or not rnd.config.get("optimalDeviateRolesId") or not rnd.stages):
            continue
        n_rounds += 1
        stat, dev = int(rnd.optimal_roles[0]), int(rnd.deviate_roles[0])
        for s, stage in enumerate(rnd.stages):
            r = int(stage.role_idx)
            rows.append({"pid": pr.participant_id, "stage": s + 1, "is_dev": float(r == dev),
                         "is_stat": float(r == stat), "is_other": float(r not in (dev, stat))})
    return rows, n_rounds


def boot_mean_ci(cluster, y, seed):
    """95% CI of mean(y) under a cluster bootstrap over participants."""
    rng = np.random.default_rng(seed)
    members = [np.flatnonzero(cluster == c) for c in np.unique(cluster)]
    boots = [y[np.concatenate([members[c] for c in rng.integers(0, len(members), len(members))])].mean()
             for _ in range(N_BOOT)]
    return np.percentile(boots, [2.5, 97.5])


def per_stage(rows):
    """[(stage, n, {key: (mean, ci)})]"""
    stage = np.array([r["stage"] for r in rows])
    cluster = np.unique([r["pid"] for r in rows], return_inverse=True)[1]
    out = []
    for k, st in enumerate(np.unique(stage)):
        m = stage == st
        vals = {}
        for j, key in enumerate(KEYS):
            y = np.array([r[key] for r in rows])[m]
            vals[key] = (float(y.mean()), boot_mean_ci(cluster[m], y, SEED + 10 + 30 * j + k))
        out.append((int(st), int(m.sum()), vals))
    return out


def participant_types(rows):
    by_pid = defaultdict(list)
    for r in rows:
        by_pid[r["pid"]].append(r)
    stat = np.array([np.mean([r["is_stat"] for r in rs]) for rs in by_pid.values()])
    dev = np.array([np.mean([r["is_dev"] for r in rs]) for rs in by_pid.values()])
    return {"stat_rate": float(stat.mean()), "dev_rate": float(dev.mean()),
            "stat_adherent": int((stat >= 0.7).sum()),
            "deviator": int(((stat < 0.7) & (dev >= 0.5)).sum()),
            "mixed": int(((stat < 0.7) & (dev < 0.5)).sum()), "n": len(by_pid)}


def plot(stages, name):
    fig, ax = plt.subplots(figsize=(3.42, 2.45))
    xs = [s for s, _, _ in stages]
    styles = {"is_dev": ("o-", "#27ae60", "deviate-optimal role (true optimal)", 4),
              "is_stat": ("s-", "#c0392b", "stat-optimal role", 3),
              "is_other": ("^--", "#999999", "other role", 2)}
    for key, (fmt, color, label, z) in styles.items():
        vals = np.array([v[key][0] for _, _, v in stages])
        cis = np.array([v[key][1] for _, _, v in stages])
        ax.errorbar(xs, vals, yerr=np.abs(cis.T - vals), fmt=fmt, color=color, markersize=2.6,
                    linewidth=1.1, capsize=1.5, elinewidth=0.7, label=label, zorder=z)
    ax.axhline(1 / 3, color="#999999", linestyle=":", linewidth=0.7, zorder=1)
    ax.text(xs[0] - 0.32, 1 / 3 + 0.015, "chance", fontsize=6, color="#999999", va="bottom")
    for st, n, _ in stages:
        ax.text(st, 0.015, f"{n}", ha="center", va="bottom", fontsize=6, color="#777")
    ax.text(xs[0] - 0.32, 0.015, "n =", ha="right", va="bottom", fontsize=6, color="#777")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Fraction of bot rounds")
    ax.set_xticks(xs)
    ax.set_xlim(xs[0] - 0.35, xs[-1] + 0.35)
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.legend(loc="upper right", frameon=False, handlelength=1.6, borderpad=0.2, labelspacing=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    FIG_DIR.mkdir(exist_ok=True)
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(FIG_DIR / f"adaptation_by_stage_{name}.{ext}", bbox_inches="tight",
                    facecolor="white", **kw)
    plt.close(fig)


def main():
    rows, n_rounds = load_rows()

    # check: all participants reproduce 2026-06-07 adaptation_by_stage.py
    t = participant_types(rows)
    s1 = [r for r in rows if r["stage"] == 1]
    assert n_rounds == 204 and t["n"] == 102, (n_rounds, t["n"])
    assert abs(t["stat_rate"] - 0.60) <= 0.01 and abs(t["dev_rate"] - 0.26) <= 0.01, t
    assert (t["stat_adherent"], t["mixed"], t["deviator"]) == (46, 36, 20), t
    assert abs(np.mean([r["is_dev"] for r in s1]) - 0.23) <= 0.02
    assert abs(np.mean([r["is_stat"] for r in s1]) - 0.64) <= 0.02

    with open(TYPES_CSV) as f:
        p_nu = {r["participant_id"]: float(r["p_non_updater"]) for r in csv.DictReader(f)}
    assert {r["pid"] for r in rows} <= set(p_nu), "bot-round participant without a type"
    for name, keep in (("updaters", lambda p: p <= 0.5), ("non-updaters", lambda p: p > 0.5)):
        sub = [r for r in rows if keep(p_nu[r["pid"]])]
        stages = per_stage(sub)
        plot(stages, name)
        out = {"n_bot_rounds": sum(r["stage"] == 1 for r in sub), **participant_types(sub),
               "by_stage": {st: {"n": n, **{k: {"mean": v[k][0], "ci": v[k][1].tolist()} for k in KEYS}}
                            for st, n, v in stages}}
        with open(OUT_DIR / f"bot_adaptation_{name}.json", "w") as f:
            json.dump(out, f, indent=2)
        print(name, json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
