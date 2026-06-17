"""2026-06-16 PNAS figure revision — accuracy numbers for prose (R1).

The accuracy_by_game FIGURE is being removed; these numbers go in prose.
No figure is produced here. Everything reuses the
2026-06-07-epistemic-rationality machinery (collect_reports + cluster-
bootstrap helpers) for human rounds, and the 2026-06-07-bot-adaptation
common_bot machinery (load_bot_records, bot_posteriors) for the NEW
bot-round comparison.

Numbers written (all with cluster-bootstrap 95% CIs, N_BOOT = 10000):
  1. Overall HUMAN inference accuracy (vs true previous-stage role).
  2. Bayesian observer SAMPLING-readout accuracy (mean posterior mass on
     the true role) + the paired human-minus-sampling difference.
  3. Posterior-MODE (MAP) accuracy — ceiling reference.
  4. Learning slope of human accuracy across GAME NUMBER (1-8).
  5. NEW: human inference accuracy on BOT rounds, and the human-round vs
     bot-round difference (confirming bot ~= human).

Cluster bootstrap: human rounds cluster by team-round; bot rounds cluster
by (game_id, participant_id) — (game_id, round_number) is not unique for
bot rounds (memory: bot-round-key-ambiguity).

Output: R1_accuracy_numbers.md in "stuff to incorporate/", plus a section
appended to summary.md.

Run from analysis/:
    uv run python experiments/2026-06-16-pnas-figure-revisions/accuracy_numbers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
V2_DIR = SCRIPT_DIR.parent / "2026-06-05-paper-figures-v2"
BOT_DIR = SCRIPT_DIR.parent / "2026-06-07-bot-adaptation"
sys.path.insert(0, str(V2_DIR))
sys.path.insert(0, str(BOT_DIR))

from common import (  # noqa: E402
    compute_posteriors, load_clean_human_teams, load_stage1, prepare_team,
    target_marginal,
)
from common_bot import (  # noqa: E402
    bot_posteriors, load_bot_records, load_stage1_canonical,
)

FULL_PIPELINE_STAGE1 = (SCRIPT_DIR.parent / "2026-05-25-full-pipeline"
                        / "stage1_inference" / "best_inference_params.json")

OUT_DIR = SCRIPT_DIR / "stuff to incorporate"
OUT_DIR.mkdir(exist_ok=True)
OUT_MD = SCRIPT_DIR / "summary.md"
OUT_NUMBERS = OUT_DIR / "R1_accuracy_numbers.md"

N_BOOT = 10_000
SEED = 0
CHANCE = 1.0 / 3.0


# ──────────────────────────────────────────────────────────────────────
# Cluster bootstrap helpers (verbatim from epistemic_rationality)
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


def boot_slope_ci(cluster, x, y, n_boot=N_BOOT, seed=SEED):
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


def boot_two_group_diff_ci(cluster_a, y_a, cluster_b, y_b,
                           n_boot=N_BOOT, seed=SEED):
    """95% CI of mean(y_a) - mean(y_b) under independent cluster bootstraps
    of the two (unpaired) groups."""
    rng = np.random.default_rng(seed)
    mem_a = _cluster_members(cluster_a)
    mem_b = _cluster_members(cluster_b)
    na, nb = len(mem_a), len(mem_b)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        ia = np.concatenate([mem_a[c] for c in rng.integers(0, na, na)])
        ib = np.concatenate([mem_b[c] for c in rng.integers(0, nb, nb)])
        boots[b] = y_a[ia].mean() - y_b[ib].mean()
    diff = float(y_a.mean() - y_b.mean())
    return diff, np.percentile(boots, [2.5, 97.5])


# ──────────────────────────────────────────────────────────────────────
# Human-round reports (verbatim collect_reports from epistemic_rationality)
# ──────────────────────────────────────────────────────────────────────

def collect_human_reports():
    with open(FULL_PIPELINE_STAGE1) as f:
        s1_canon = json.load(f)
    s1, strat = load_stage1()
    for k in ("tau_prior", "epsilon", "memory_strategy"):
        assert s1_canon[k] == s1[k], (
            f"Stage-1 param mismatch on '{k}': 05-25 {s1_canon[k]!r} vs "
            f"common.py {s1[k]!r}")
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
                "marginal": target_marginal(posteriors[si], target_pos),
                "guessed": guessed,
                "true_prev": true_prev,
            })
    print(f"[accuracy] human: {len(rows)} reports from "
          f"{len(key_to_idx)} team-rounds")
    return rows, {"tau_prior": s1["tau_prior"], "epsilon": s1["epsilon"],
                  "memory_strategy": strat.name}


# ──────────────────────────────────────────────────────────────────────
# Bot-round reports (NEW) — CLAUDE.md bot-round ground truth via common_bot
# ──────────────────────────────────────────────────────────────────────

def collect_bot_reports():
    """One row per human report in a bot round.

    rec["inferred"]: {stage_idx: {bot_pos: guessed_role_idx}}; reports made
    at stage s pair with bot_posteriors[s]. Bots never switch, so a report
    about bot position `pos` is correct iff guessed == bot_role_map[pos].
    Cluster by (game_id, participant_id) — (game_id, round_number) is not
    unique for bot rounds.
    """
    s1, strat = load_stage1_canonical()
    records = load_bot_records()

    key_to_idx: dict = {}
    rows = []
    for rec in records:
        if not rec["inferred"]:
            continue
        posteriors = bot_posteriors(rec, s1, strat)
        key = (rec["game_id"], rec["participant_id"])
        for si, guesses in rec["inferred"].items():
            if si >= len(posteriors):
                continue
            for pos, guessed in guesses.items():
                if pos not in rec["bot_role_map"]:
                    continue
                true_role = rec["bot_role_map"][pos]
                ci = key_to_idx.setdefault(key, len(key_to_idx))
                rows.append({
                    "cluster": ci,
                    "marginal": target_marginal(posteriors[si], pos),
                    "guessed": guessed,
                    "true_prev": true_role,
                })
    print(f"[accuracy] bot: {len(rows)} reports from "
          f"{len(key_to_idx)} (game, participant) clusters")
    return rows


# ──────────────────────────────────────────────────────────────────────
# Compute + write
# ──────────────────────────────────────────────────────────────────────

def fmt(v):
    return f"{v:.3f}"


def fmt_ci(ci):
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def main():
    h_rows, s1info = collect_human_reports()
    b_rows = collect_bot_reports()

    # ---- human round arrays ----------------------------------------------
    h_cluster = np.array([r["cluster"] for r in h_rows])
    h_game = np.array([r["round_number"] for r in h_rows], dtype=float)
    h_ok = np.array([r["guessed"] == r["true_prev"] for r in h_rows],
                    dtype=float)
    h_map = np.array([int(np.argmax(r["marginal"])) == r["true_prev"]
                      for r in h_rows], dtype=float)
    h_samp = np.array([r["marginal"][r["true_prev"]] for r in h_rows],
                      dtype=float)

    # 1. overall human accuracy
    h_acc = float(h_ok.mean())
    h_acc_ci = boot_mean_ci(h_cluster, h_ok, seed=SEED + 1)

    # 2. sampling readout + paired human - sampling
    samp = float(h_samp.mean())
    samp_ci = boot_mean_ci(h_cluster, h_samp, seed=SEED + 2)
    diff_hs = float(np.mean(h_ok - h_samp))
    diff_hs_ci = boot_mean_ci(h_cluster, h_ok - h_samp, seed=SEED + 3)

    # 3. MAP / mode accuracy (ceiling)
    mapacc = float(h_map.mean())
    map_ci = boot_mean_ci(h_cluster, h_map, seed=SEED + 4)

    # 4. learning slope across game number
    h_slope, h_slope_ci = boot_slope_ci(h_cluster, h_game, h_ok, seed=SEED + 5)

    # 5. bot-round human accuracy + human-vs-bot difference
    b_cluster = np.array([r["cluster"] for r in b_rows])
    b_ok = np.array([r["guessed"] == r["true_prev"] for r in b_rows],
                    dtype=float)
    b_acc = float(b_ok.mean())
    b_acc_ci = boot_mean_ci(b_cluster, b_ok, seed=SEED + 6)
    hb_diff, hb_diff_ci = boot_two_group_diff_ci(
        h_cluster, h_ok, b_cluster, b_ok, seed=SEED + 7)

    # ---- print -----------------------------------------------------------
    print()
    print(f"  1. human accuracy            {fmt(h_acc)} {fmt_ci(h_acc_ci)}")
    print(f"  2. sampling readout          {fmt(samp)} {fmt_ci(samp_ci)}")
    print(f"     human - sampling (paired) {diff_hs:+.3f} "
          f"[{diff_hs_ci[0]:+.3f}, {diff_hs_ci[1]:+.3f}]")
    print(f"  3. MAP/mode accuracy         {fmt(mapacc)} {fmt_ci(map_ci)}")
    print(f"  4. human learning slope/game {h_slope:+.4f} "
          f"[{h_slope_ci[0]:+.4f}, {h_slope_ci[1]:+.4f}]")
    print(f"  5. bot-round human accuracy  {fmt(b_acc)} {fmt_ci(b_acc_ci)}")
    print(f"     human - bot (difference)  {hb_diff:+.3f} "
          f"[{hb_diff_ci[0]:+.3f}, {hb_diff_ci[1]:+.3f}]")

    # ---- markdown --------------------------------------------------------
    n_h_clusters = int(h_cluster.max()) + 1
    n_b_clusters = int(b_cluster.max()) + 1
    lines = [
        "# R1 — Inference-accuracy numbers for paper prose",
        "",
        f"No figure (the accuracy_by_game panel is removed; report in prose). "
        f"Human rounds: {len(h_rows):,} reports from {n_h_clusters} clean "
        f"team-rounds (5 exports). Bot rounds: {len(b_rows):,} human reports "
        f"from {n_b_clusters} (game, participant) clusters. Stage-1 params: "
        f"tau_prior = {s1info['tau_prior']:.4f}, "
        f"epsilon = {s1info['epsilon']:.4f}, "
        f"memory = `{s1info['memory_strategy']}`. "
        f"All CIs are percentile cluster bootstraps with {N_BOOT:,} "
        f"resamples; human rounds clustered by team-round, bot rounds by "
        f"(game_id, participant_id) — (game_id, round_number) is not unique "
        f"for bot rounds. Chance accuracy = 1/3 = {CHANCE:.3f}.",
        "",
        "## Numbers to quote",
        "",
        f"- **Overall human inference accuracy** (vs true previous-stage "
        f"role): **{h_acc:.3f}** (95% CI {fmt_ci(h_acc_ci)}); chance = "
        f"{CHANCE:.3f}.",
        f"- **Bayesian observer, sampling-readout accuracy** (mean posterior "
        f"mass on the true role): **{samp:.3f}** (95% CI {fmt_ci(samp_ci)}). "
        f"**Paired human - sampling difference**: **{diff_hs:+.3f}** "
        f"(95% CI [{diff_hs_ci[0]:+.3f}, {diff_hs_ci[1]:+.3f}]).",
        f"- **Posterior-mode (MAP) accuracy** (ceiling reference): "
        f"**{mapacc:.3f}** (95% CI {fmt_ci(map_ci)}).",
        f"- **Learning slope of human accuracy across game number (1-8)**: "
        f"**{h_slope:+.4f}** per game (95% CI [{h_slope_ci[0]:+.4f}, "
        f"{h_slope_ci[1]:+.4f}]).",
        f"- **Bot-round human inference accuracy**: **{b_acc:.3f}** "
        f"(95% CI {fmt_ci(b_acc_ci)}). **Human-round minus bot-round "
        f"difference**: **{hb_diff:+.3f}** (95% CI [{hb_diff_ci[0]:+.3f}, "
        f"{hb_diff_ci[1]:+.3f}]) — the CI straddles 0"
        + (", confirming bot-round inference accuracy is statistically "
           "indistinguishable from human-round accuracy."
           if hb_diff_ci[0] <= 0 <= hb_diff_ci[1] else
           ". NOTE: the CI excludes 0.")
        + "",
        "",
        "## Summary table",
        "",
        "| Quantity | Value | 95% CI |",
        "|----------|------:|--------|",
        f"| Human accuracy (human rounds) | {h_acc:.3f} | {fmt_ci(h_acc_ci)} |",
        f"| Bayesian sampling readout | {samp:.3f} | {fmt_ci(samp_ci)} |",
        f"| Human - sampling (paired) | {diff_hs:+.3f} | "
        f"[{diff_hs_ci[0]:+.3f}, {diff_hs_ci[1]:+.3f}] |",
        f"| MAP (mode) accuracy / ceiling | {mapacc:.3f} | {fmt_ci(map_ci)} |",
        f"| Human learning slope (per game) | {h_slope:+.4f} | "
        f"[{h_slope_ci[0]:+.4f}, {h_slope_ci[1]:+.4f}] |",
        f"| Human accuracy (bot rounds) | {b_acc:.3f} | {fmt_ci(b_acc_ci)} |",
        f"| Human-round - bot-round difference | {hb_diff:+.3f} | "
        f"[{hb_diff_ci[0]:+.3f}, {hb_diff_ci[1]:+.3f}] |",
        "",
    ]
    OUT_NUMBERS.write_text("\n".join(lines) + "\n")
    print(f"\n[accuracy] wrote {OUT_NUMBERS.resolve()}")

    # ---- append to summary.md --------------------------------------------
    section = "\n".join(lines)
    marker = "# R1 — Inference-accuracy numbers for paper prose"
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
    print(f"[accuracy] wrote {OUT_MD.resolve()}")


if __name__ == "__main__":
    main()
