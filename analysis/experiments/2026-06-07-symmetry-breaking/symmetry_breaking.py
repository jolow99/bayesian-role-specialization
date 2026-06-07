"""Symmetry-breaking in human rounds — one PNAS single-column figure.

Question (human-round counterpart of the bot-round adaptation figure):
can humans resolve SYMMETRICAL cases? When two teammates have identical
stat profiles, their stat-based priors suggest the same role, so the
pair starts in a coordination problem the stats cannot solve — someone
has to deviate.

Row unit: (team-round, player pair, stage); clash = both players of the
pair chose the same role at that stage. Three series by stage:

  * identical-stat pairs — the symmetric case (122 rounds with one
    222/222 pair + 40 fully-symmetric 222_222_222 rounds with 3 pairs);
  * stat-distinct pairs — internal control: same games, same stages,
    but the stats already break the tie;
  * chance = 1/3 (two independent uniform choices collide w.p. 1/3).

Cluster-bootstrap 95% CIs, cluster = GAME (export, game_id): the same
3 participants play all rounds of a game, so team-round-level
resampling would understate uncertainty.

summary.md adds: stage-1 vs final clash rates, end-of-round split
fraction, time-to-first-split for initially-clashing identical pairs,
mirror switches (simultaneous same-role switches), and the
fully-symmetric (222_222_222) vs one-pair (H_222_222) breakdown.

Scope: 5 exports, clean games, human rounds with 3 players — the same
204 team-rounds as 06-05/06-07-instrumental/epistemic.

Run from analysis/:
    uv run python experiments/2026-06-07-symmetry-breaking/symmetry_breaking.py
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
V2_DIR = SCRIPT_DIR.parent / "2026-06-05-paper-figures-v2"
sys.path.insert(0, str(V2_DIR))

from common import load_clean_human_teams  # noqa: E402

OUT_DIR = SCRIPT_DIR / "stuff to incorporate"
OUT_DIR.mkdir(exist_ok=True)
OUT_MD = SCRIPT_DIR / "summary.md"

N_BOOT = 10_000
SEED = 0

FIG_W = 3.42
FIG_H = 2.45

IDENT_COLOR = "#000000"     # identical-stat pairs (the symmetric case)
DISTINCT_COLOR = "#8e44ad"  # stat-distinct pairs (internal control)
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
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kw)
        print(f"[symmetry] wrote {path}")


# ──────────────────────────────────────────────────────────────────────
# Cluster bootstrap (epistemic-rationality conventions, game clusters)
# ──────────────────────────────────────────────────────────────────────

def _cluster_members(cluster):
    n_clusters = cluster.max() + 1
    return [np.flatnonzero(cluster == c) for c in range(n_clusters)]


def boot_mean_ci(cluster, y, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    members = _cluster_members(cluster)
    n_clusters = len(members)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, n_clusters, n_clusters)
        idx = np.concatenate([members[c] for c in picked])
        boots[b] = y[idx].mean()
    return np.percentile(boots, [2.5, 97.5])


def boot_group_diff_ci(cluster, y, group, n_boot=N_BOOT, seed=SEED):
    """95% CI of mean(y|group) - mean(y|~group), resampling whole games
    (most games contribute pairs of BOTH types, so the difference is
    largely within-cluster)."""
    rng = np.random.default_rng(seed)
    members = _cluster_members(cluster)
    n_clusters = len(members)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, n_clusters, n_clusters)
        idx = np.concatenate([members[c] for c in picked])
        gb, yb = group[idx], y[idx]
        boots[b] = yb[gb].mean() - yb[~gb].mean()
    diff = float(y[group].mean() - y[~group].mean())
    return diff, np.percentile(boots, [2.5, 97.5])


# ──────────────────────────────────────────────────────────────────────
# Rows: one per (team-round, pair, stage)
# ──────────────────────────────────────────────────────────────────────

def collect_rows():
    teams = load_clean_human_teams()
    game_to_idx: dict = {}
    rows = []        # stage-level clash rows
    pair_rounds = []  # one per (team-round, pair): trajectory-level facts
    for (export, gid, rn), team_prs in teams.items():
        rnd = team_prs[0].round
        parts = rnd.stat_profile_id.split("_")
        roles = {pr.player_id: [s.role_idx for s in pr.round.stages]
                 for pr in team_prs}
        ci = game_to_idx.setdefault((export, gid), len(game_to_idx))
        for a in range(3):
            for b in range(a + 1, 3):
                identical = parts[a] == parts[b]
                n = min(len(roles[a]), len(roles[b]))
                traj = [(roles[a][s], roles[b][s]) for s in range(n)]
                if not traj:
                    continue
                for s, (ra, rb) in enumerate(traj):
                    rows.append({
                        "cluster": ci,
                        "stage": s + 1,
                        "identical": identical,
                        "clash": float(ra == rb),
                    })
                # simultaneous switches to the same role ("mirroring")
                mirror = sum(
                    1 for s in range(1, n)
                    if traj[s][0] == traj[s][1]
                    and traj[s][0] != traj[s - 1][0]
                    and traj[s][1] != traj[s - 1][1])
                clash0 = traj[0][0] == traj[0][1]
                first_split = next((s for s, (ra, rb) in enumerate(traj)
                                    if ra != rb), None)
                pair_rounds.append({
                    "cluster": ci,
                    "identical": identical,
                    "fully_symmetric": rnd.stat_profile_id == "222_222_222",
                    "n_stages": n,
                    "clash_first": clash0,
                    "split_last": traj[-1][0] != traj[-1][1],
                    "first_split": first_split,
                    "mirror_switches": mirror,
                })
    print(f"[symmetry] {len(rows)} pair-stage rows, "
          f"{len(pair_rounds)} pair-rounds, {len(game_to_idx)} games, "
          f"{len(teams)} team-rounds")
    return rows, pair_rounds, len(teams)


# ──────────────────────────────────────────────────────────────────────
# Figure — clash rate by stage, identical vs distinct pairs
# ──────────────────────────────────────────────────────────────────────

def fig_symmetry_breaking(rows):
    cluster = np.array([r["cluster"] for r in rows])
    stage = np.array([r["stage"] for r in rows])
    identical = np.array([r["identical"] for r in rows])
    clash = np.array([r["clash"] for r in rows])

    stages = sorted(np.unique(stage))
    per_stage = []  # (stage, {True/False: (n, mean, ci)})
    for k, st in enumerate(stages):
        vals = {}
        for j, ident in enumerate((True, False)):
            m = (stage == st) & (identical == ident)
            sub_cluster = np.unique(cluster[m], return_inverse=True)[1]
            ci = boot_mean_ci(sub_cluster, clash[m],
                              seed=SEED + 10 + 30 * j + k)
            vals[ident] = (int(m.sum()), float(clash[m].mean()), ci)
        per_stage.append((int(st), vals))

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    xs = [p[0] for p in per_stage]
    styles = {
        True: ("o-", IDENT_COLOR, "identical-stat pairs", 4),
        False: ("D-", DISTINCT_COLOR, "stat-distinct pairs", 3),
    }
    for ident, (fmt, color, label, z) in styles.items():
        vals = np.array([p[1][ident][1] for p in per_stage])
        cis = np.array([p[1][ident][2] for p in per_stage])
        ax.errorbar(xs, vals, yerr=np.abs(cis.T - vals), fmt=fmt,
                    color=color, markersize=2.6, linewidth=1.1,
                    capsize=1.5, elinewidth=0.7, label=label, zorder=z)

    ax.axhline(1 / 3, color=REF_COLOR, linestyle=":", linewidth=0.7,
               zorder=1)
    ax.text(xs[-1] + 0.32, 1 / 3 + 0.015, "chance", fontsize=6,
            color=REF_COLOR, va="bottom", ha="right")
    for st, vals in per_stage:
        ax.text(st, 0.015, f"{vals[True][0]}", ha="center", va="bottom",
                fontsize=6, color="#777")
    ax.text(xs[0] - 0.32, 0.015, "n =", ha="right", va="bottom",
            fontsize=6, color="#777")

    ax.set_xlabel("Stage")
    ax.set_ylabel("Fraction of pairs playing the same role")
    ax.set_xticks(xs)
    ax.set_xlim(xs[0] - 0.35, xs[-1] + 0.35)
    ax.set_ylim(0, 0.82)
    ax.set_yticks(np.arange(0, 0.81, 0.2))
    ax.legend(loc="upper right", frameon=False, handlelength=1.6,
              borderpad=0.2, labelspacing=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=0.3)
    savefig(fig, "symmetry_breaking")
    plt.close(fig)

    # pooled identical-minus-distinct clash difference, all stages
    diff, diff_ci = boot_group_diff_ci(cluster, clash, identical,
                                       seed=SEED + 500)
    return per_stage, (diff, diff_ci)


# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────

def trajectory_stats(pair_rounds, ident_filter):
    prs = [p for p in pair_rounds if ident_filter(p)]
    if not prs:
        return None
    cluster = np.unique([p["cluster"] for p in prs], return_inverse=True)[1]
    clash_first = np.array([p["clash_first"] for p in prs], dtype=float)
    split_last = np.array([p["split_last"] for p in prs], dtype=float)
    out = {
        "n": len(prs),
        "clash_first": (float(clash_first.mean()),
                        boot_mean_ci(cluster, clash_first, seed=SEED + 300)),
        "split_last": (float(split_last.mean()),
                       boot_mean_ci(cluster, split_last, seed=SEED + 301)),
        "mirror_total": int(sum(p["mirror_switches"] for p in prs)),
    }
    # among initially-clashing pairs that ever split: stages until split
    clashers = [p for p in prs if p["clash_first"]]
    if clashers:
        resolved = [p for p in clashers if p["first_split"] is not None]
        out["clashers"] = len(clashers)
        out["resolved"] = len(resolved)
        out["mean_stages_to_split"] = (
            float(np.mean([p["first_split"] for p in resolved]))
            if resolved else float("nan"))
    return out


def write_summary(per_stage, pooled_diff, pair_rounds, n_team_rounds):
    ident = trajectory_stats(pair_rounds, lambda p: p["identical"])
    ident_full = trajectory_stats(
        pair_rounds, lambda p: p["identical"] and p["fully_symmetric"])
    ident_one = trajectory_stats(
        pair_rounds, lambda p: p["identical"] and not p["fully_symmetric"])
    distinct = trajectory_stats(pair_rounds, lambda p: not p["identical"])

    lines = [
        "# Symmetry-breaking in human rounds",
        "",
        f"Scope: {n_team_rounds} clean human team-rounds (5 exports). "
        "Row unit = (team-round, player pair, stage); clash = both "
        "players of the pair chose the same role. Identical-stat pairs "
        "come from the 114/141/411_222_222 profiles (one 222/222 pair "
        "each) and 222_222_222 (fully symmetric, 3 pairs); 411_141_114 "
        "rounds contribute stat-distinct pairs only. All CIs are "
        f"percentile cluster bootstraps over games ({N_BOOT:,} "
        "resamples) — the same 3 participants play every round of a "
        "game.",
        "",
        "## Clash rate by stage (the figure)",
        "",
        "| Stage | n identical | P(clash) identical | 95% CI "
        "| n distinct | P(clash) distinct | 95% CI |",
        "|---|--:|--:|---|--:|--:|---|",
    ]
    for st, vals in per_stage:
        ni, mi, cii = vals[True]
        nd, md, cid = vals[False]
        lines.append(
            f"| {st} | {ni} | {mi:.3f} | [{cii[0]:.3f}, {cii[1]:.3f}] "
            f"| {nd} | {md:.3f} | [{cid[0]:.3f}, {cid[1]:.3f}] |")
    diff, diff_ci = pooled_diff
    lines += [
        "",
        f"Pooled over all stages, identical-stat pairs clash "
        f"**{diff:+.3f}** [{diff_ci[0]:+.3f}, {diff_ci[1]:+.3f}] more "
        "often than stat-distinct pairs (cluster-bootstrap difference, "
        "game clusters).",
    ]

    def block(name, s):
        if s is None:
            return [f"### {name}", "", "_no pairs_", ""]
        out = [
            f"### {name} (n = {s['n']} pair-rounds)",
            "",
            f"- P(clash at stage 1): **{s['clash_first'][0]:.3f}** "
            f"[{s['clash_first'][1][0]:.3f}, {s['clash_first'][1][1]:.3f}]",
            f"- P(split at final stage): **{s['split_last'][0]:.3f}** "
            f"[{s['split_last'][1][0]:.3f}, {s['split_last'][1][1]:.3f}]",
            f"- mirror switches (simultaneous same-role switches): "
            f"{s['mirror_total']} total",
        ]
        if "clashers" in s:
            out.append(
                f"- of {s['clashers']} pairs clashing at stage 1, "
                f"{s['resolved']} split at some point "
                f"(mean stages to first split "
                f"{s['mean_stages_to_split']:.2f})")
        out.append("")
        return out

    lines += ["", "## Trajectory-level stats", ""]
    lines += block("Identical-stat pairs (all)", ident)
    lines += block("— of which fully-symmetric rounds (222_222_222)",
                   ident_full)
    lines += block("— of which one-pair rounds (H_222_222)", ident_one)
    lines += block("Stat-distinct pairs (control)", distinct)

    lines += [
        "## Interpretation",
        "",
        "Symmetry has a real but modest cost, and humans resolve it "
        "behaviorally. Identical-stat pairs clash consistently more "
        "than the stat-distinct control at every stage (pooled "
        f"difference {diff:+.3f}), most at stage 1 of fully-symmetric "
        "222_222_222 rounds (0.44), where stats give no tie-breaking "
        "signal at all. The stage-1 clash rate sits near chance rather "
        "than far above it — players already randomize/diversify "
        "somewhat from the start. Crucially, clashes do not persist: "
        "~3/4 of identical pairs that start clashing split within "
        "~1.4 stages on average, and both pair types' clash rates "
        "decline across the round — consistent with players using "
        "teammates' observed behavior (not stats) to settle who takes "
        "the contested role. Mirror switches (both players switching "
        "into the same role simultaneously) do occur (70 among "
        "identical pairs) — the symmetry-breaking failure mode the "
        "qualitative flip-flop figure illustrates.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines))
    print(f"[symmetry] wrote {OUT_MD}")


def main():
    rows, pair_rounds, n_team_rounds = collect_rows()
    assert n_team_rounds == 204, n_team_rounds

    per_stage, pooled_diff = fig_symmetry_breaking(rows)
    for st, vals in per_stage:
        ni, mi, cii = vals[True]
        nd, md, cid = vals[False]
        print(f"  stage {st}: identical {mi:.3f} [{cii[0]:.3f}, "
              f"{cii[1]:.3f}] (n={ni})   distinct {md:.3f} "
              f"[{cid[0]:.3f}, {cid[1]:.3f}] (n={nd})")
    print(f"  pooled identical-distinct diff: {pooled_diff[0]:+.3f} "
          f"[{pooled_diff[1][0]:+.3f}, {pooled_diff[1][1]:+.3f}]")
    write_summary(per_stage, pooled_diff, pair_rounds, n_team_rounds)


if __name__ == "__main__":
    main()
