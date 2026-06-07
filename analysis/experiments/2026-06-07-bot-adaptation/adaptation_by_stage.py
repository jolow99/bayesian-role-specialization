"""Figure 1 — bot-round adaptation by stage (PNAS single column) + summary.

One row per (clean bot round, stage): does the human play the
deviate-optimal role (the TRUE optimal — adapting to the stubborn bots),
their stat-optimal role (what their stat profile suggests), or another
role? Per stage 1-5: three fractions with cluster-bootstrap 95% CIs,
cluster = participant (participants contribute ~2 bot rounds each).

Also writes summary.md: overall stat/dev rates, per-participant behavior
types (05-28 thresholds), the per-stage table behind the figure, and a
team-symmetry breakdown (SYMMETRIC_PROFILES).

Reproduction checks vs prior art (fail loudly on drift):
  * 204 rounds / 102 participants (05-28 bot_adaptation_summary.md)
  * overall stat-rate ≈ 60%, dev-rate ≈ 26% (participant means, ±1pp)
  * behavior types 46 / 36 / 20 (stat ≥ 0.7 / mixed / dev ≥ 0.5)
  * stage 1: P(dev) ≈ 23%, P(stat) ≈ 64% (05-25 fig_bot_role_choice, ±2pp)

Run from analysis/:
    uv run python experiments/2026-06-07-bot-adaptation/adaptation_by_stage.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from common_bot import OUT_DIR, load_bot_records  # noqa: E402
from shared.constants import ROLE_SHORT  # noqa: E402

OUT_MD = SCRIPT_DIR / "summary.md"

N_BOOT = 10_000
SEED = 0

FIG_W = 3.42
FIG_H = 2.45

DEV_COLOR = "#27ae60"     # deviate-optimal (the true optimal): green
STAT_COLOR = "#c0392b"    # stat-optimal (the stubborn default): red
OTHER_COLOR = "#999999"   # neither: grey

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
        print(f"[bot-adaptation] wrote {path}")


# ──────────────────────────────────────────────────────────────────────
# Cluster bootstrap (copied from 2026-06-07-epistemic-rationality)
# ──────────────────────────────────────────────────────────────────────

def _cluster_members(cluster):
    n_clusters = cluster.max() + 1
    return [np.flatnonzero(cluster == c) for c in range(n_clusters)]


def boot_mean_ci(cluster, y, n_boot=N_BOOT, seed=SEED):
    """95% CI of mean(y) under a cluster bootstrap over participants."""
    rng = np.random.default_rng(seed)
    members = _cluster_members(cluster)
    n_clusters = len(members)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, n_clusters, n_clusters)
        idx = np.concatenate([members[c] for c in picked])
        boots[b] = y[idx].mean()
    return np.percentile(boots, [2.5, 97.5])


# ──────────────────────────────────────────────────────────────────────
# Rows: one per (bot round, stage)
# ──────────────────────────────────────────────────────────────────────

def collect_rows(records):
    pid_to_idx: dict = {}
    rows = []
    for rec in records:
        ci = pid_to_idx.setdefault(rec["participant_id"], len(pid_to_idx))
        for s, role in enumerate(rec["human_roles"]):
            is_dev = role == rec["human_dev_opt"]
            is_stat = role == rec["human_stat_opt"]
            assert not (is_dev and is_stat)   # stat-opt != dev-opt by design
            rows.append({
                "cluster": ci,
                "participant_id": rec["participant_id"],
                "stage": s + 1,
                "is_dev": float(is_dev),
                "is_stat": float(is_stat),
                "is_other": float(not is_dev and not is_stat),
                "symmetry": rec["symmetry"],
            })
    return rows, len(pid_to_idx)


# ──────────────────────────────────────────────────────────────────────
# Figure 1 — adaptation by stage
# ──────────────────────────────────────────────────────────────────────

def fig_adaptation_by_stage(rows):
    cluster = np.array([r["cluster"] for r in rows])
    stage = np.array([r["stage"] for r in rows])
    series = {k: np.array([r[k] for r in rows])
              for k in ("is_dev", "is_stat", "is_other")}

    stages = sorted(np.unique(stage))
    per_stage = []  # (stage, n, {key: (mean, ci)})
    for k, st in enumerate(stages):
        m = stage == st
        sub_cluster = np.unique(cluster[m], return_inverse=True)[1]
        vals = {}
        for j, key in enumerate(("is_dev", "is_stat", "is_other")):
            y = series[key][m]
            ci = boot_mean_ci(sub_cluster, y, seed=SEED + 10 + 30 * j + k)
            vals[key] = (float(y.mean()), ci)
        per_stage.append((int(st), int(m.sum()), vals))

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    xs = [p[0] for p in per_stage]
    styles = {
        "is_dev": ("o-", DEV_COLOR, "deviate-optimal role (true optimal)", 4),
        "is_stat": ("s-", STAT_COLOR, "stat-optimal role", 3),
        "is_other": ("^--", OTHER_COLOR, "other role", 2),
    }
    for key, (fmt, color, label, z) in styles.items():
        vals = np.array([p[2][key][0] for p in per_stage])
        cis = np.array([p[2][key][1] for p in per_stage])
        ax.errorbar(xs, vals, yerr=np.abs(cis.T - vals), fmt=fmt,
                    color=color, markersize=2.6, linewidth=1.1,
                    capsize=1.5, elinewidth=0.7, label=label, zorder=z)

    ax.axhline(1 / 3, color=OTHER_COLOR, linestyle=":", linewidth=0.7,
               zorder=1)
    ax.text(xs[0] - 0.32, 1 / 3 + 0.015, "chance", fontsize=6,
            color=OTHER_COLOR, va="bottom", ha="left")
    for st, n, _ in per_stage:
        ax.text(st, 0.015, f"{n}", ha="center", va="bottom", fontsize=6,
                color="#777")
    ax.text(xs[0] - 0.32, 0.015, "n =", ha="right", va="bottom", fontsize=6,
            color="#777")

    ax.set_xlabel("Stage")
    ax.set_ylabel("Fraction of bot rounds")
    ax.set_xticks(xs)
    ax.set_xlim(xs[0] - 0.35, xs[-1] + 0.35)
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.legend(loc="upper right", frameon=False, handlelength=1.6,
              borderpad=0.2, labelspacing=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=0.3)
    savefig(fig, "adaptation_by_stage")
    plt.close(fig)
    return per_stage


# ──────────────────────────────────────────────────────────────────────
# Per-participant rates + behavior types (05-28 conventions)
# ──────────────────────────────────────────────────────────────────────

def participant_rates(rows):
    by_pid = defaultdict(list)
    for r in rows:
        by_pid[r["participant_id"]].append(r)
    out = []
    for pid, rs in by_pid.items():
        stat_rate = float(np.mean([r["is_stat"] for r in rs]))
        dev_rate = float(np.mean([r["is_dev"] for r in rs]))
        if stat_rate >= 0.7:
            cat = "Stat-adherent"
        elif dev_rate >= 0.5:
            cat = "Deviator"
        else:
            cat = "Mixed/Explorer"
        out.append({"participant_id": pid, "n_stages": len(rs),
                    "stat_rate": stat_rate, "dev_rate": dev_rate,
                    "type": cat})
    return out


def symmetry_breakdown(rows):
    """Stat/dev fractions split by team stat-profile symmetry class.

    SYMMETRIC_PROFILES: 'all' = all three players share 222 stats (the
    human included — fully symmetrical); 'last_two' = the two bots are
    symmetric (222) while the human has a distinct profile.
    """
    by_sym = defaultdict(list)
    for r in rows:
        by_sym[r["symmetry"] or "asymmetric"].append(r)
    out = {}
    for sym, rs in sorted(by_sym.items()):
        cluster = np.unique([r["cluster"] for r in rs],
                            return_inverse=True)[1]
        entry = {"n_stage_rows": len(rs),
                 "n_participants": int(cluster.max()) + 1}
        for j, key in enumerate(("is_dev", "is_stat")):
            y = np.array([r[key] for r in rs])
            entry[key] = (float(y.mean()),
                          boot_mean_ci(cluster, y, seed=SEED + 200 + j))
        out[sym] = entry
    return out


# ──────────────────────────────────────────────────────────────────────
# Summary + reproduction checks
# ──────────────────────────────────────────────────────────────────────

def write_summary(n_records, n_participants, per_stage, prates, sym):
    types = {t: sum(1 for p in prates if p["type"] == t)
             for t in ("Stat-adherent", "Mixed/Explorer", "Deviator")}
    stat_overall = float(np.mean([p["stat_rate"] for p in prates]))
    dev_overall = float(np.mean([p["dev_rate"] for p in prates]))

    lines = [
        "# Bot-round adaptation — can humans adapt to stubborn teammates?",
        "",
        f"Scope: {n_records} clean bot rounds (5 exports), "
        f"{n_participants} participants. In each bot round, 2 fixed-strategy "
        "bots play their deviate-optimal roles; the human's stat-optimal "
        "role always differs from their deviate-optimal role, so adapting "
        "means abandoning the stat-suggested default. All CIs are "
        f"percentile cluster bootstraps over participants ({N_BOOT:,} "
        "resamples).",
        "",
        "## Headline numbers (participant means, 05-28 conventions)",
        "",
        f"- Stat-optimal play: **{stat_overall:.0%}** of stages.",
        f"- Deviate-optimal play: **{dev_overall:.0%}** of stages.",
        "",
        "## Behavior types per participant",
        "",
        "| Type | Criterion | N | % |",
        "|------|-----------|--:|--:|",
    ]
    crits = {"Stat-adherent": "stat_rate ≥ 0.70",
             "Mixed/Explorer": "neither threshold met",
             "Deviator": "dev_rate ≥ 0.50"}
    for t, c in crits.items():
        n = types[t]
        lines.append(f"| **{t}** | {c} | {n} | {n / len(prates):.0%} |")

    lines += [
        "",
        "## Adaptation by stage (Figure 1)",
        "",
        "Row unit = (bot round, stage). Cluster-bootstrap 95% CIs over "
        "participants.",
        "",
        "| Stage | n | P(deviate-opt) | 95% CI | P(stat-opt) | 95% CI "
        "| P(other) | 95% CI |",
        "|---|--:|--:|---|--:|---|--:|---|",
    ]
    for st, n, vals in per_stage:
        cells = []
        for key in ("is_dev", "is_stat", "is_other"):
            mean, ci = vals[key]
            cells.append(f"{mean:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}]")
        lines.append(f"| {st} | {n} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Symmetry breakdown",
        "",
        "Team stat-profile symmetry class (`SYMMETRIC_PROFILES`): "
        "`last_two` = the two bots share symmetric 222 stats while the "
        "human has a distinct profile; `all` would mean the human is "
        "222 as well (fully symmetrical).",
        "",
        "| Symmetry class | n stage-rows | n participants "
        "| P(deviate-opt) | 95% CI | P(stat-opt) | 95% CI |",
        "|---|--:|--:|--:|---|--:|---|",
    ]
    for s, e in sym.items():
        d_mean, d_ci = e["is_dev"]
        st_mean, st_ci = e["is_stat"]
        lines.append(
            f"| `{s}` | {e['n_stage_rows']} | {e['n_participants']} "
            f"| {d_mean:.3f} | [{d_ci[0]:.3f}, {d_ci[1]:.3f}] "
            f"| {st_mean:.3f} | [{st_ci[0]:.3f}, {st_ci[1]:.3f}] |")

    lines += [
        "",
        "## Interpretation",
        "",
        "Stat-optimal play starts dominant and falls across stages while "
        "deviate-optimal play roughly doubles — evidence that a meaningful "
        "fraction of humans integrate their teammates' observed behavior "
        "and adapt away from their stat-suggested default. The "
        "per-participant split shows this is driven by individual "
        "differences (Deviators adapt, Stat-adherents never do) rather "
        "than uniform partial adaptation.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines))
    print(f"[bot-adaptation] wrote {OUT_MD}")
    return types, stat_overall, dev_overall


def main():
    records = load_bot_records()
    rows, n_participants = collect_rows(records)

    # Reproduction checks vs 05-28 bot_adaptation_summary.md
    assert len(records) == 204, len(records)
    assert n_participants == 102, n_participants

    per_stage = fig_adaptation_by_stage(rows)
    prates = participant_rates(rows)
    sym = symmetry_breakdown(rows)
    types, stat_overall, dev_overall = write_summary(
        len(records), n_participants, per_stage, prates, sym)

    print(f"\n  overall (participant means): stat {stat_overall:.3f}, "
          f"dev {dev_overall:.3f}")
    for st, n, vals in per_stage:
        print(f"  stage {st}: n={n:3d}  "
              + "  ".join(f"{k[3:]} {vals[k][0]:.3f} "
                          f"[{vals[k][1][0]:.3f}, {vals[k][1][1]:.3f}]"
                          for k in ("is_dev", "is_stat", "is_other")))
    print(f"  behavior types: {types}")

    assert abs(stat_overall - 0.60) <= 0.01, stat_overall
    assert abs(dev_overall - 0.26) <= 0.01, dev_overall
    assert (types["Stat-adherent"], types["Mixed/Explorer"],
            types["Deviator"]) == (46, 36, 20), types
    s1_vals = per_stage[0][2]
    assert abs(s1_vals["is_dev"][0] - 0.23) <= 0.02, s1_vals["is_dev"][0]
    assert abs(s1_vals["is_stat"][0] - 0.64) <= 0.02, s1_vals["is_stat"][0]
    print("  reproduction checks vs 05-28/05-25 prior art: OK")


if __name__ == "__main__":
    main()
