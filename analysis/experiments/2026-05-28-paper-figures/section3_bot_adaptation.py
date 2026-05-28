"""Section 3 — Can the model explain human adaptation to stubborn teammates?

Quantitative: bot-round adaptation analysis on 5-export clean bot rounds.
    Each human plays with 2 fixed-role bots. The human's stat-suggested role
    differs from the true deviate-optimal role. Per-participant, we measure:
        stat_rate = fraction of stages where human played stat-optimal role
        dev_rate  = fraction of stages where human played deviate-optimal role
    Categorize each participant as:
        Stat-adherent (stat_rate >= 0.7)  — refuses to deviate
        Deviator      (dev_rate  >= 0.5)  — successfully adapts
        Mixed         (else)              — flips between roles
    Finding: a meaningful fraction of humans CAN adapt.

Qualitative: a single HUMAN team-round where one or more players flip-flop
    between roles before the team converges to a coordinated combo. (Bot-round
    examples would just repeat what the quantitative overview already shows
    — the qualitative role is to illustrate the same adaptation dynamics in
    the all-human setting, where uncertainty about teammates is mutual.)

Outputs:
    figures/bot_adaptation_overview.png      — per-config bars + behavior-type pie
    figures/qualitative_flip_flop.png        — human-round flip-flop → convergence
    bot_adaptation_summary.md                — markdown summary + interpretation

Bot-round ground truth follows CLAUDE.md (build_bot_round_layout).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent / "2026-05-25-full-pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

sys.path.insert(0, str(SCRIPT_DIR))

from pipeline import (
    discover_dropout_games, filter_clean_prs, EXPORT_DIRS,
    load_human_team_records,
)
from shared.constants import ROLE_SHORT, TURNS_PER_STAGE, ROLE_CHAR_TO_IDX
from shared.data_loading import load_all_exports, build_bot_round_layout
from shared.parsing import parse_stat_optimal_roles, parse_deviate_roles
from shared.inference import preferred_action, game_step
from _qualitative import render_qualitative_human

FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
SUMMARY_PATH = SCRIPT_DIR / "bot_adaptation_summary.md"

ROLE_NAMES_FULL = {0: "Fighter", 1: "Tank", 2: "Medic"}


# ──────────────────────────────────────────────────────────────────────
# Data: per-player bot rounds
# ──────────────────────────────────────────────────────────────────────

def load_bot_player_rounds():
    """Return one record per clean human in a bot round.

    Each record uses logical-position-0 = the human, so stat-optimal/deviate-
    optimal role indices line up with the human's choices.
    """
    all_prs = load_all_exports(data_dirs=EXPORT_DIRS)
    dropout_games = discover_dropout_games(all_prs)
    clean = filter_clean_prs(all_prs, dropout_games)

    records = []
    for pr in clean:
        if pr.round.round_type != "bot":
            continue
        cfg = pr.round.config
        dev_id = cfg.get("optimalDeviateRolesId")
        if not dev_id:
            continue
        layout = build_bot_round_layout(pr)
        stat_opt = parse_stat_optimal_roles(dev_id)
        dev_opt = parse_deviate_roles(dev_id)
        human_stat_opt = int(stat_opt[0])
        human_dev_opt = int(dev_opt[0])

        # Human role sequence (logical "role index" per stage, already keyed by
        # the human's own position by Empirica)
        human_role_seq = [int(s.role_idx) for s in pr.round.stages]
        if not human_role_seq:
            continue

        records.append({
            "participant_id": pr.participant_id,
            "game_id": pr.game_id,
            "round_number": int(pr.round.round_number),
            "treatment_id": f"{pr.round.stat_profile_id}__{dev_id}",
            "stat_profile": pr.round.stat_profile_id,
            "deviate_roles_id": dev_id,
            "human_stat_optimal": human_stat_opt,
            "human_deviate_optimal": human_dev_opt,
            "human_role_seq": human_role_seq,
            "human_pid": layout.pid,
            "bot_role_map": dict(layout.bot_role_map),
            "bot_positions": list(layout.others),
            "player_stats": layout.player_stats,
            "lds": [int(c) for c in pr.round.enemy_intent_sequence],
            "config": cfg,
            "stages": pr.round.stages,
        })
    return records


# ──────────────────────────────────────────────────────────────────────
# Aggregate per-participant rates
# ──────────────────────────────────────────────────────────────────────

def per_participant_rates(records):
    """For each participant, compute aggregated stat_rate and dev_rate across
    all their bot rounds.
    """
    by_pid = defaultdict(list)
    for r in records:
        by_pid[r["participant_id"]].append(r)

    rows = []
    for pid, recs in by_pid.items():
        n_total = n_stat = n_dev = 0
        for r in recs:
            for role_idx in r["human_role_seq"]:
                n_total += 1
                if role_idx == r["human_stat_optimal"]:
                    n_stat += 1
                if role_idx == r["human_deviate_optimal"]:
                    n_dev += 1
        if n_total == 0:
            continue
        rows.append({
            "participant_id": pid,
            "n_stages": n_total,
            "n_rounds": len(recs),
            "stat_rate": n_stat / n_total,
            "dev_rate": n_dev / n_total,
        })
    return pd.DataFrame(rows)


def categorize(row, stat_threshold=0.7, dev_threshold=0.5):
    if row["stat_rate"] >= stat_threshold:
        return "Stat-adherent"
    if row["dev_rate"] >= dev_threshold:
        return "Deviator"
    return "Mixed/Explorer"


# ──────────────────────────────────────────────────────────────────────
# Per-treatment role-choice distribution
# ──────────────────────────────────────────────────────────────────────

def per_treatment_role_dist(records):
    """Returns dict[treatment_id] = {role_char → fraction}, plus stat-optimal /
    deviate-optimal annotation."""
    by_tx = defaultdict(list)
    for r in records:
        by_tx[r["treatment_id"]].append(r)

    out = {}
    for tx, recs in by_tx.items():
        roles = []
        for r in recs:
            roles.extend(r["human_role_seq"])
        n = len(roles)
        if n == 0:
            continue
        stat_opt = recs[0]["human_stat_optimal"]
        dev_opt = recs[0]["human_deviate_optimal"]
        out[tx] = {
            "n_stages": n,
            "n_rounds": len(recs),
            "stat_optimal_role": stat_opt,
            "deviate_optimal_role": dev_opt,
            "frac": {role_idx: roles.count(role_idx) / n for role_idx in (0, 1, 2)},
        }
    return out


# ──────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────

CAT_COLORS = {
    "Stat-adherent":  "#E53935",
    "Mixed/Explorer": "#FFA726",
    "Deviator":       "#43A047",
}
ROLE_COLORS = {0: "#e74c3c", 1: "#3498db", 2: "#2ecc71"}


def plot_overview(pdf, treatment_dist, save_path):
    n = len(pdf)
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 1.0], wspace=0.35)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_pie = fig.add_subplot(gs[0, 1])

    # ── Left: per-treatment role distribution ──
    treatments = sorted(treatment_dist.keys())
    x = np.arange(len(treatments))
    bar_w = 0.25
    for role_idx, role_char in enumerate(("F", "T", "M")):
        fracs = [treatment_dist[tx]["frac"][role_idx] for tx in treatments]
        ax_bar.bar(x + (role_idx - 1) * bar_w, fracs, bar_w,
                   color=ROLE_COLORS[role_idx], alpha=0.85, label=role_char,
                   edgecolor="white", linewidth=0.4)

    # Annotate stat-optimal / deviate-optimal under each treatment
    for i, tx in enumerate(treatments):
        info = treatment_dist[tx]
        stat_char = ROLE_SHORT[info["stat_optimal_role"]]
        dev_char = ROLE_SHORT[info["deviate_optimal_role"]]
        ax_bar.text(i, -0.08, f"stat:{stat_char}", ha="center", fontsize=7,
                    color="#666")
        ax_bar.text(i, -0.13, f"dev:{dev_char}", ha="center", fontsize=7,
                    color="#27ae60", fontweight="bold")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(
        [tx.split("__")[1] for tx in treatments], rotation=30, ha="right",
        fontsize=8,
    )
    ax_bar.set_ylim(-0.18, 1.0)
    ax_bar.axhline(0, color="black", linewidth=0.6)
    ax_bar.set_ylabel("Fraction of stages")
    ax_bar.set_title("Human role choices per treatment (bot rounds, clean)",
                     fontsize=11, fontweight="bold")
    ax_bar.legend(title="Role", fontsize=8, loc="upper right")
    ax_bar.grid(True, alpha=0.2, axis="y")
    for spine in ("top", "right"):
        ax_bar.spines[spine].set_visible(False)

    # ── Right: behavior-type pie ──
    cats = ["Stat-adherent", "Mixed/Explorer", "Deviator"]
    counts = [int((pdf["type"] == c).sum()) for c in cats]
    pie_colors = [CAT_COLORS[c] for c in cats]
    wedges, texts, autotexts = ax_pie.pie(
        counts, labels=[f"{c}\n({k})" for c, k in zip(cats, counts)],
        colors=pie_colors,
        autopct=lambda pct: f"{pct:.0f}%",
        startangle=90, textprops={"fontsize": 9}, pctdistance=0.72,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_color("white")
    ax_pie.set_title(f"Behavior types (n={n} participants)\n"
                     f"Stat-adherent: stat_rate ≥ 0.7  ·  "
                     f"Deviator: dev_rate ≥ 0.5",
                     fontsize=10, fontweight="bold")

    fig.suptitle("Section 3 — Adaptation to stubborn bot teammates",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Qualitative: pick a HUMAN team-round where someone flip-flops, then the
# team converges. Bot-round adaptation is already captured in the
# quantitative overview, so the qualitative complements it by showing the
# all-human setting (mutual uncertainty about teammates).
# ──────────────────────────────────────────────────────────────────────

def pick_human_flip_flop(human_records):
    """Pick a clean human team-round where at least one player switches roles
    ≥ 2 times across stages AND the team's last two stages share the same
    combo (i.e. converges after exploration)."""
    candidates = []
    for ri, rec in enumerate(human_records):
        sr = rec["stage_roles"]
        n_stages = len(sr)
        if n_stages < 4:
            continue

        # Per-player switch counts
        per_player_seqs = [[combo[i] for combo in sr] for i in range(3)]
        max_switches = max(
            sum(1 for k in range(1, len(seq)) if seq[k] != seq[k - 1])
            for seq in per_player_seqs
        )
        total_switches = sum(
            sum(1 for k in range(1, len(seq)) if seq[k] != seq[k - 1])
            for seq in per_player_seqs
        )
        if max_switches < 2:
            continue

        # Convergence: last two stages identical
        converged = (sr[-1] == sr[-2])
        last_distinct = len(set(sr[-1]))  # prefer 3-distinct so figure is readable

        # Score: convergence > role variety in final stage > total switches > #stages
        score = (int(converged) * 10000
                 + last_distinct * 1000
                 + total_switches * 50
                 + n_stages)
        candidates.append((score, ri, rec))

    if not candidates:
        # Relax: drop convergence requirement
        for ri, rec in enumerate(human_records):
            sr = rec["stage_roles"]
            if len(sr) < 3:
                continue
            per_player_seqs = [[combo[i] for combo in sr] for i in range(3)]
            max_switches = max(
                sum(1 for k in range(1, len(seq)) if seq[k] != seq[k - 1])
                for seq in per_player_seqs
            )
            if max_switches >= 2:
                candidates.append((
                    len(set(sr[-1])) * 1000 + max_switches * 10 + len(sr),
                    ri, rec,
                ))

    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][2]


def render_human_flip_flop(record, all_prs, save_path):
    """Render the picked human flip-flop round in 03-24 style (no inferences)."""
    return render_qualitative_human(
        record, all_prs, save_path,
        extra_subtitle=None,
        role_trajectory_caption=None,
    )


def _UNUSED_replay_bot_state(record):  # kept for reference, no longer called
    """Per-turn (team_hp, enemy_hp, intent, actions) starting from full HP."""
    cfg = record["config"]
    player_stats = record["player_stats"].astype(float)
    team_max_hp = int(cfg.get("maxTeamHealth"))
    enemy_max_hp = int(cfg.get("maxEnemyHealth"))
    boss_damage = float(cfg.get("bossDamage"))

    pid = record["human_pid"]
    bot_role_map = record["bot_role_map"]
    seq = record["human_role_seq"]
    lds = record["lds"]

    # Build in-game-position role combo per stage
    stage_combos = []
    for human_role in seq:
        combo = [0, 0, 0]
        combo[pid] = int(human_role)
        for pos, role in bot_role_map.items():
            combo[pos] = int(role)
        stage_combos.append(combo)

    team_hp = float(team_max_hp)
    enemy_hp = float(enemy_max_hp)
    turn_hp = [(team_hp, enemy_hp)]
    turn_intent = []
    turn_actions = []
    turn_idx = 0
    for s in range(len(seq)):
        roles = stage_combos[s]
        for _ in range(TURNS_PER_STAGE):
            if turn_idx >= len(lds) or team_hp <= 0 or enemy_hp <= 0:
                break
            intent = int(lds[turn_idx])
            actions = [preferred_action(roles[i], intent, team_hp, team_max_hp)
                       for i in range(3)]
            new_thp, new_ehp = game_step(intent, team_hp, enemy_hp, actions,
                                          player_stats, boss_damage, team_max_hp)
            turn_intent.append(intent)
            turn_actions.append((actions, roles[:]))
            turn_hp.append((new_thp, new_ehp))
            team_hp, enemy_hp = new_thp, new_ehp
            turn_idx += 1

    return {
        "stage_combos": stage_combos,
        "turn_hp": turn_hp,
        "turn_intent": turn_intent,
        "turn_actions": turn_actions,
        "team_max_hp": team_max_hp,
        "enemy_max_hp": enemy_max_hp,
        "n_stages": len(seq),
        "n_turns": len(turn_intent),
    }


def _UNUSED_render_bot_flip_flop(record, save_path):  # kept for reference
    bundle = replay_bot_state(record)
    n_stages = bundle["n_stages"]
    n_turns = bundle["n_turns"]
    pid = record["human_pid"]
    stat_opt = record["human_stat_optimal"]
    dev_opt = record["human_deviate_optimal"]
    seq = record["human_role_seq"]
    bot_role_map = record["bot_role_map"]

    PLAYER_COLORS = ["#9b59b6", "#e67e22", "#16a085"]
    ROLE_Y = {"F": 0, "T": 1, "M": 2}
    ACTION_MARKERS = {0: "^", 1: "s", 2: "v"}

    fig = plt.figure(figsize=(13, 8), facecolor="white")
    gs = fig.add_gridspec(3, 1, height_ratios=[2.6, 1.0, 1.0], hspace=0.4,
                          left=0.10, right=0.96, top=0.90, bottom=0.10)
    ax_role = fig.add_subplot(gs[0])
    ax_hp = fig.add_subplot(gs[1])
    ax_legend = fig.add_subplot(gs[2])
    ax_legend.axis("off")

    x_lo, x_hi = 0.5, n_turns + 0.5

    # Title
    title = (f"Bot round  ·  treatment {record['deviate_roles_id']}  "
             f"·  human at in-game pos {pid}  ·  stats {record['stat_profile']}  "
             f"·  game {record['game_id'][-6:]} r{record['round_number']}")
    fig.text(0.10, 0.96, title, fontsize=10, ha="left")
    fig.text(0.10, 0.93, f"Human stat-optimal: {ROLE_NAMES_FULL[stat_opt]}   ·   "
             f"Human deviate-optimal: {ROLE_NAMES_FULL[dev_opt]}   ·   "
             f"Bots fixed at: " + ", ".join(
                 f"P{p+1}={ROLE_NAMES_FULL[role]}"
                 for p, role in sorted(bot_role_map.items())),
             fontsize=9, ha="left", color="#444")

    # ── Role panel ──
    ax_role.set_xlim(x_lo, x_hi)
    ax_role.set_ylim(-0.4, 2.4)
    ax_role.invert_yaxis()
    ax_role.set_yticks([0, 1, 2])
    ax_role.set_yticklabels(["Fighter", "Tank", "Medic"], fontsize=10)
    ax_role.set_ylabel("Role")
    ax_role.tick_params(axis="x", labelbottom=False, length=0)

    # Stage backgrounds + boundaries
    for s in range(n_stages):
        if s % 2 == 1:
            ax_role.axvspan(s * TURNS_PER_STAGE + 0.5,
                            (s + 1) * TURNS_PER_STAGE + 0.5,
                            color="#000", alpha=0.03, zorder=0)
    for s in range(1, n_stages):
        ax_role.axvline(s * TURNS_PER_STAGE + 0.5,
                         color="#888", linewidth=0.5, alpha=0.4)
    for s in range(n_stages):
        ax_role.text(s * TURNS_PER_STAGE + 1.5, -0.32, f"S{s+1}",
                      ha="center", va="center", fontsize=8, color="#888")

    # Reference horizontals + stat/dev optimal lines for the HUMAN row
    for y in (0, 1, 2):
        ax_role.axhline(y, color="#eee", linewidth=0.5, zorder=0.5)
    ax_role.axhline(stat_opt, color="#E53935", linewidth=1.1,
                     linestyle=":", alpha=0.45, zorder=1)
    ax_role.text(x_hi - 0.2, stat_opt - 0.2,
                  f"human stat-opt ({ROLE_SHORT[stat_opt]})",
                  ha="right", fontsize=7, color="#E53935", alpha=0.7)
    ax_role.axhline(dev_opt, color="#27ae60", linewidth=1.4,
                     linestyle="-", alpha=0.40, zorder=1)
    ax_role.text(x_hi - 0.2, dev_opt + 0.32,
                  f"human dev-opt ({ROLE_SHORT[dev_opt]})",
                  ha="right", fontsize=7, color="#27ae60", alpha=0.85)

    # Plot each player's trajectory
    for in_game_pos in range(3):
        is_human = (in_game_pos == pid)
        color = PLAYER_COLORS[in_game_pos]
        # Step line
        xs, ys = [], []
        for s in range(n_stages):
            if is_human:
                role_idx = seq[s]
            else:
                role_idx = bot_role_map[in_game_pos]
            xs.extend([s * TURNS_PER_STAGE + 0.5,
                       (s + 1) * TURNS_PER_STAGE + 0.5])
            ys.extend([role_idx, role_idx])
        ax_role.plot(xs, ys, color=color,
                      linewidth=2.4 if is_human else 1.4,
                      alpha=0.95 if is_human else 0.6,
                      zorder=3 if is_human else 2,
                      label=f"P{in_game_pos+1}"
                             + (" (human)" if is_human else " (bot)"))

        # Action markers
        for ti, (actions, _) in enumerate(bundle["turn_actions"]):
            s = ti // TURNS_PER_STAGE
            if is_human:
                role_idx = seq[s]
            else:
                role_idx = bot_role_map[in_game_pos]
            marker = ACTION_MARKERS[actions[in_game_pos]]
            ax_role.scatter([ti + 1], [role_idx], marker=marker, s=44,
                             color=color, edgecolor="white", linewidth=0.6,
                             zorder=4 if is_human else 3)

    # Enemy attack markers
    for ti, intent in enumerate(bundle["turn_intent"]):
        if intent == 1:
            ax_role.scatter([ti + 1], [-0.18], marker="v", s=28,
                             color="#e74c3c", edgecolor="none", zorder=5)

    ax_role.legend(loc="upper right", fontsize=8, ncol=3, frameon=False,
                    bbox_to_anchor=(1.0, 1.10))
    for spine in ("top", "right"):
        ax_role.spines[spine].set_visible(False)

    # ── HP panel ──
    turns_x = np.arange(0, n_turns + 1)
    team_hps = np.array([h[0] for h in bundle["turn_hp"]])
    enemy_hps = np.array([h[1] for h in bundle["turn_hp"]])
    team_max = bundle["team_max_hp"]
    enemy_max = bundle["enemy_max_hp"]

    ax_hp.fill_between(turns_x + 0.5, 0, team_hps / team_max,
                        color="#3498db", alpha=0.35, step="post")
    ax_hp.fill_between(turns_x + 0.5, 0, enemy_hps / enemy_max,
                        color="#e74c3c", alpha=0.20, step="post")
    ax_hp.plot(turns_x + 0.5, team_hps / team_max,
                color="#2980b9", linewidth=1.4, drawstyle="steps-post")
    ax_hp.plot(turns_x + 0.5, enemy_hps / enemy_max,
                color="#c0392b", linewidth=1.4, drawstyle="steps-post")
    for s in range(n_stages):
        if s % 2 == 1:
            ax_hp.axvspan(s * TURNS_PER_STAGE + 0.5,
                          (s + 1) * TURNS_PER_STAGE + 0.5,
                          color="#000", alpha=0.03, zorder=0)
    for s in range(1, n_stages):
        ax_hp.axvline(s * TURNS_PER_STAGE + 0.5, color="#888",
                       linewidth=0.5, alpha=0.4)
    ax_hp.set_xlim(x_lo, x_hi)
    ax_hp.set_ylim(0, 1.02)
    ax_hp.set_yticks([0, 0.5, 1.0])
    ax_hp.set_yticklabels(["0", "½", "max"], fontsize=8)
    ax_hp.set_ylabel("HP (fraction\nof max)", fontsize=9)
    ax_hp.set_xticks(np.arange(1, n_turns + 1))
    ax_hp.set_xticklabels([str(i) for i in range(1, n_turns + 1)], fontsize=8)
    ax_hp.set_xlabel("turn", fontsize=9)
    for spine in ("top", "right"):
        ax_hp.spines[spine].set_visible(False)

    # ── Footer legend ──
    flips = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
    summary = (f"Human role trajectory: "
               + " → ".join(ROLE_SHORT[r] for r in seq)
               + f"   ({flips} role switches, "
               + ("ended on deviate-optimal ✓"
                  if seq[-1] == dev_opt else "did not converge to deviate-optimal ✗")
               + ")")
    ax_legend.text(0.02, 0.65, summary, fontsize=10, ha="left", va="center",
                    color="#222", fontweight="bold")
    ax_legend.text(0.02, 0.20,
                    "action markers: ▲ ATTACK   ■ BLOCK   ▼ HEAL    "
                    "red ▼ above role panel = enemy attacks    "
                    "thick line = human, faint = bots    "
                    "dotted red line = human's stat-optimal,  "
                    "solid green line = human's deviate-optimal",
                    fontsize=8, ha="left", va="center", color="#555")

    fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────

def write_summary(pdf, treatment_dist, n_records, save_path):
    cat_counts = pdf["type"].value_counts()
    n = len(pdf)
    lines = [
        "# Section 3 — Adaptation to stubborn bot teammates",
        "",
        f"**Data scope.** 5 exports, bot rounds, clean games only: "
        f"**{n_records} human bot-round observations** across "
        f"{len(treatment_dist)} treatments, **{n} unique participants**. "
        f"Bot positions and human stats are resolved per CLAUDE.md → "
        f"\"Bot Round Ground Truth\".",
        "",
        "## Headline numbers",
        "",
        f"- Aggregate stat-optimal play: **{pdf['stat_rate'].mean():.0%}** of stages.",
        f"- Aggregate deviate-optimal play: **{pdf['dev_rate'].mean():.0%}** of stages.",
        "",
        "## Behavior types per participant",
        "",
        "| Type | Criterion | N | % |",
        "|------|-----------|--:|--:|",
    ]
    for t in ("Stat-adherent", "Mixed/Explorer", "Deviator"):
        c = int(cat_counts.get(t, 0))
        crit = {
            "Stat-adherent": "stat_rate ≥ 0.70",
            "Mixed/Explorer": "neither threshold met",
            "Deviator": "dev_rate ≥ 0.50",
        }[t]
        lines.append(f"| **{t}** | {crit} | {c} | {c/n*100:.0f}% |")
    lines.append("")
    lines.append("## Per-treatment role-choice fractions")
    lines.append("")
    lines.append("| Treatment | N stages | N rounds | Stat-opt role | Dev-opt role | %F | %T | %M |")
    lines.append("|-----------|--------:|--------:|:-------------:|:------------:|---:|---:|---:|")
    for tx in sorted(treatment_dist):
        d = treatment_dist[tx]
        lines.append(
            f"| `{tx}` | {d['n_stages']} | {d['n_rounds']} | "
            f"{ROLE_SHORT[d['stat_optimal_role']]} | "
            f"{ROLE_SHORT[d['deviate_optimal_role']]} | "
            f"{d['frac'][0]:.0%} | {d['frac'][1]:.0%} | {d['frac'][2]:.0%} |"
        )
    lines.append("")
    lines.append(
        "## Interpretation\n\n"
        "Bot rounds pit the human's stat-suggested role against the true "
        "deviate-optimal role. **Some humans clearly adapt** — Deviators "
        "play deviate-optimal ≥ 50% of the time despite their stats "
        "suggesting a different role. **Others refuse to deviate** — "
        "Stat-adherents play stat-optimal ≥ 70% of the time. The middle "
        "group flips between strategies. This individual-difference "
        "pattern is what the paper's model needs to explain in Section 3."
    )
    Path(save_path).write_text("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 66)
    print("Section 3 — bot-round adaptation")
    print("=" * 66)

    print("\nLoading clean bot player-rounds (5 exports)...")
    records = load_bot_player_rounds()
    n_treatments = len({r["treatment_id"] for r in records})
    print(f"  {len(records)} bot player-rounds across {n_treatments} treatments")

    pdf = per_participant_rates(records)
    pdf["type"] = pdf.apply(categorize, axis=1)
    print(f"  {len(pdf)} unique participants")
    print("\n  Behavior type counts:")
    for t in ("Stat-adherent", "Mixed/Explorer", "Deviator"):
        c = int((pdf["type"] == t).sum())
        print(f"    {t:<18s}: {c:>3d}  ({c/len(pdf)*100:.0f}%)")

    treatment_dist = per_treatment_role_dist(records)

    print("\nWriting bot_adaptation_overview.png...")
    plot_overview(pdf, treatment_dist, FIGURES_DIR / "bot_adaptation_overview.png")

    print("Picking qualitative human-round flip-flop example...")
    human_records = load_human_team_records(verbose=False)
    flip = pick_human_flip_flop(human_records)
    if flip is None:
        print("  No suitable human record found.")
    else:
        sr = flip["stage_roles"]
        per_player_switches = [
            sum(1 for k in range(1, len(sr)) if sr[k][i] != sr[k - 1][i])
            for i in range(3)
        ]
        print(f"  Picked game {flip['game_id'][-6:]} r{flip['round_number']} "
              f"env {flip['env_id']}: {' → '.join(sr)} "
              f"(per-player switches: {per_player_switches})")
        print("  Loading raw PRs for inference rendering...")
        all_prs = load_all_exports(data_dirs=EXPORT_DIRS)
        render_human_flip_flop(
            flip, all_prs, FIGURES_DIR / "qualitative_flip_flop.png")

    print(f"\nWriting {SUMMARY_PATH}...")
    write_summary(pdf, treatment_dist, len(records), SUMMARY_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
