"""Behavioural figures (paper Figs 2, 3, 6, 7) — redesigned for texture.

Replaces the flat per-stage line/bar charts with decompositions that let
the reader see the structure behind the aggregate numbers:

 - Fig 3 (human role behaviour): stat-profile-conditional role choice
   (left panel) + role transition matrix (right panel).
 - Fig 6 (bot deviation): per-team faint trajectories with the per-stage
   mean overlaid, so the reader sees how many teams actually reach
   deviate-optimal vs how many never do.
 - Fig 7 (inference): confusion matrices (true × inferred) for human and
   bot rounds, revealing systematic vs random errors.
 - Fig 2 (win rates): unchanged from the flat version — outcomes per
   batch is the right level of granularity for that one.

Bot-round handling uses ``build_bot_round_layout`` (CLAUDE.md Bot Round
Ground Truth) so the human's in-game position is correct.
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
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

from shared import EXPORTS_DIR
from shared.constants import ROLE_SHORT
from shared.data_loading import load_all_exports, build_bot_round_layout


EXPORT_DIRS = sorted(EXPORTS_DIR.glob("bayesian-role-specialization-*"))


def _short_batch_label(name: str) -> str:
    parts = name.split("-")
    return f"{parts[5]}-{parts[6]}"


def load_clean_records():
    by_export = {}
    all_dropout_games = set()
    for d in EXPORT_DIRS:
        prs = load_all_exports(data_dirs=[d])
        dropout = {pr.game_id for pr in prs if pr.is_dropout}
        all_dropout_games |= dropout
        by_export[_short_batch_label(d.name)] = [
            pr for pr in prs if pr.game_id not in dropout
        ]
    return by_export, all_dropout_games


# Stat profile labels (the four-way space the experiment uses).
PROFILE_NAME = {
    "411": "411 (Fighter-leaning)",
    "141": "141 (Tank-leaning)",
    "114": "114 (Medic-leaning)",
    "222": "222 (symmetric)",
}
PROFILE_OPTIMAL = {"411": 0, "141": 1, "114": 2, "222": None}
ROLE_COLORS = {0: "#e74c3c", 1: "#3498db", 2: "#2ecc71"}
ROLE_NAMES = ["F", "T", "M"]


# ──────────────────────────────────────────────────────────────────────
# Fig 2 — win rates per batch (kept simple; outcomes don't have texture)
# ──────────────────────────────────────────────────────────────────────

def fig_outcome_win_rates(by_export):
    rows = []
    for batch, prs in by_export.items():
        seen_human = {}
        seen_bot = []
        for pr in prs:
            if pr.round.round_type == "human":
                seen_human[(pr.game_id, pr.round.round_number)] = pr.round.outcome
            elif pr.round.round_type == "bot":
                seen_bot.append(pr.round.outcome)
        n_h = len(seen_human)
        n_b = len(seen_bot)
        win_h = sum(1 for o in seen_human.values() if o == "WIN")
        win_b = sum(1 for o in seen_bot if o == "WIN")
        rows.append({
            "batch": batch,
            "n_human": n_h, "win_human": win_h / n_h if n_h else float("nan"),
            "n_bot": n_b, "win_bot": win_b / n_b if n_b else float("nan"),
        })

    fig, ax = plt.subplots(figsize=(9, 5))
    batches = [r["batch"] for r in rows]
    x = np.arange(len(batches))
    width = 0.4
    ax.bar(x - width / 2, [r["win_human"] for r in rows], width,
           color="#2ecc71", edgecolor="white", label="Human rounds")
    ax.bar(x + width / 2, [r["win_bot"] for r in rows], width,
           color="#e74c3c", edgecolor="white", label="Bot rounds")
    for i, r in enumerate(rows):
        ax.text(x[i] - width / 2, r["win_human"] + 0.01,
                f"{r['win_human']:.0%}\n(n={r['n_human']})",
                ha="center", va="bottom", fontsize=8)
        ax.text(x[i] + width / 2, r["win_bot"] + 0.01,
                f"{r['win_bot']:.0%}\n(n={r['n_bot']})",
                ha="center", va="bottom", fontsize=8)
    mean_h = np.mean([r["win_human"] for r in rows])
    mean_b = np.mean([r["win_bot"] for r in rows])
    ax.axhline(mean_h, color="#2ecc71", linestyle=":", alpha=0.5)
    ax.axhline(mean_b, color="#e74c3c", linestyle=":", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(batches)
    ax.set_xlabel("Batch (export date)")
    ax.set_ylabel("Win rate")
    ax.set_ylim(0, 1.1)
    ax.set_title(
        f"Round outcomes across {len(batches)} batches "
        f"(mean human {mean_h:.0%}, bot {mean_b:.0%})"
    )
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    out = FIGURES_DIR / "fig_outcome_win_rates.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ──────────────────────────────────────────────────────────────────────
# Fig 3 — stat-profile-conditional choice + transition matrix
# ──────────────────────────────────────────────────────────────────────

def fig_human_role_behavior(by_export):
    """Two panels:

    LEFT: For each of the 4 stat profile types, the F/T/M choice
        distribution at each stage 1..5. Lines per role with the
        stat-suggested role highlighted.

    RIGHT: Role transition matrix P(role_t | role_{t-1}) for stage>=2,
        aggregated across all human player-stage transitions.
    """
    # (profile_str, stage, role) -> count
    profile_stage_role = defaultdict(int)
    profile_stage_total = defaultdict(int)
    transitions = np.zeros((3, 3), dtype=int)  # rows = from, cols = to

    for batch, prs in by_export.items():
        for pr in prs:
            if pr.round.round_type != "human":
                continue
            stats_per_pos = np.array(
                [[int(c) for c in p] for p in pr.round.stat_profile_id.split("_")],
                dtype=int,
            )
            my_stats = stats_per_pos[pr.player_id]
            # Identify which canonical profile this player has
            profile_key = "".join(str(int(s)) for s in my_stats)

            prev = None
            for si, stage in enumerate(pr.round.stages):
                role = stage.role_idx
                profile_stage_role[(profile_key, si + 1, role)] += 1
                profile_stage_total[(profile_key, si + 1)] += 1
                if prev is not None:
                    transitions[prev, role] += 1
                prev = role

    profiles = sorted(PROFILE_NAME.keys())
    stages = sorted({s for (_, s, _) in profile_stage_role.keys()})

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.5, 1.5, 1.5, 1.5, 2.2],
                           wspace=0.32)

    # Four stat-profile panels
    for i, profile in enumerate(profiles):
        ax = fig.add_subplot(gs[0, i])
        for role in range(3):
            ys = []
            for s in stages:
                tot = profile_stage_total.get((profile, s), 0)
                n = profile_stage_role.get((profile, s, role), 0)
                ys.append(n / tot if tot else np.nan)
            color = ROLE_COLORS[role]
            optimal = PROFILE_OPTIMAL[profile]
            lw = 3 if role == optimal else 1.5
            ls = "-" if role == optimal else "--"
            ax.plot(stages, ys, marker="o", color=color, linewidth=lw,
                    linestyle=ls,
                    label=f"{ROLE_NAMES[role]}{' (stat-suggested)' if role == optimal else ''}")
        ax.axhline(1 / 3, color="gray", linestyle=":", alpha=0.5)
        ax.set_xticks(stages)
        ax.set_xticklabels([f"S{s}" for s in stages], fontsize=8)
        ax.set_ylim(0, 1)
        n_per_stage = profile_stage_total.get((profile, 1), 0)
        ax.set_title(f"{PROFILE_NAME[profile]}\n(n_stage1 = {n_per_stage})",
                     fontsize=9, fontweight="bold")
        if i == 0:
            ax.set_ylabel("P(role chosen)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="y", alpha=0.2)

    # Transition heatmap
    ax = fig.add_subplot(gs[0, 4])
    P = transitions.astype(float)
    row_sums = P.sum(axis=1, keepdims=True)
    P_norm = np.divide(P, row_sums, where=row_sums > 0)
    im = ax.imshow(P_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(ROLE_NAMES)
    ax.set_yticklabels(ROLE_NAMES)
    ax.set_xlabel("Role at stage t")
    ax.set_ylabel("Role at stage t−1")
    diag = float(np.trace(P_norm) / 3.0)
    ax.set_title(
        f"Role transitions (stage ≥ 2)\n"
        f"mean stay rate = {diag:.0%}",
        fontsize=9, fontweight="bold",
    )
    for r in range(3):
        for c in range(3):
            v = P_norm[r, c]
            n = int(transitions[r, c])
            color = "white" if v > 0.5 else "black"
            ax.text(c, r, f"{v:.0%}\n(n={n})",
                    ha="center", va="center", fontsize=9, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Human-round role behaviour — choice distribution by stat profile (left) "
        "and stage-to-stage transitions (right)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "fig_human_role_behavior.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")

    print("  Stat-profile conditional rates at stage 1:")
    for p in profiles:
        n = profile_stage_total.get((p, 1), 0)
        for r in range(3):
            cnt = profile_stage_role.get((p, 1, r), 0)
            print(f"    {p} -> {ROLE_NAMES[r]}: {cnt/n:.1%}" if n else "")
    print(f"  Diagonal stay rate (mean of P[r|r]): {diag:.1%}")


# ──────────────────────────────────────────────────────────────────────
# Fig 6 — bot-round deviation with per-team trajectories
# ──────────────────────────────────────────────────────────────────────

def fig_bot_role_choice(by_export):
    """Per-team trajectories of the human's chosen role across stages,
    overlaid with the deviate-optimal and stat-optimal references and the
    mean. Each trajectory is faint; colour encodes whether the team ended
    at deviate-optimal."""
    trajectories = []  # list of (stages, roles, dev_opt, stat_opt, outcome)
    for batch, prs in by_export.items():
        for pr in prs:
            if pr.round.round_type != "bot":
                continue
            cfg = pr.round.config
            dev_roles = cfg.get("deviateRoles") or []
            opt_roles = cfg.get("optimalRoles") or []
            if not dev_roles or not opt_roles:
                continue
            human_dev_opt = int(dev_roles[0])
            human_stat_opt = int(opt_roles[0])
            roles = [s.role_idx for s in pr.round.stages]
            if not roles:
                continue
            trajectories.append({
                "stages": list(range(1, len(roles) + 1)),
                "roles": roles,
                "dev_opt": human_dev_opt,
                "stat_opt": human_stat_opt,
                "outcome": pr.round.outcome,
                "reached_dev": int(roles[-1] == human_dev_opt),
            })

    # Per-stage mean rates
    by_stage_dev = defaultdict(lambda: [0, 0])
    by_stage_stat = defaultdict(lambda: [0, 0])
    for tr in trajectories:
        for s, r in zip(tr["stages"], tr["roles"]):
            by_stage_dev[s][0] += int(r == tr["dev_opt"])
            by_stage_dev[s][1] += 1
            by_stage_stat[s][0] += int(r == tr["stat_opt"])
            by_stage_stat[s][1] += 1
    stages = sorted(by_stage_dev.keys())
    dev_means = [by_stage_dev[s][0] / by_stage_dev[s][1] for s in stages]
    stat_means = [by_stage_stat[s][0] / by_stage_stat[s][1] for s in stages]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Each trajectory: jitter the (stage, role) points slightly and
    # connect with a faint line. Colour = green if ended at dev-opt, else
    # orange.
    rng = np.random.default_rng(0)
    for tr in trajectories:
        color = "#27ae60" if tr["reached_dev"] else "#e67e22"
        ys = [r - tr["dev_opt"] for r in tr["roles"]]
        # plot in "distance from dev-opt" coordinate so all trajectories
        # are aligned around 0 = dev-opt
        jitter = rng.normal(0, 0.05, size=len(ys))
        ax.plot([s + j for s, j in zip(tr["stages"], jitter * 0.1)],
                 [y + j for y, j in zip(ys, jitter)],
                 color=color, alpha=0.12, linewidth=0.8)

    # Mean lines (in same dev-opt-relative space, just for the dev-opt rate)
    dev_y0 = [0] * len(stages)
    ax.plot(stages, dev_y0, color="black", linestyle="-", linewidth=0,
            label=" ")  # spacer
    ax.axhline(0, color="black", linestyle="-", alpha=0.4, linewidth=1)
    ax.text(stages[-1] + 0.05, 0, "  dev-optimal", va="center",
            fontsize=9, fontweight="bold")

    # Overlay the *rates* on a secondary axis-style banner
    ax2 = ax.twinx()
    ax2.plot(stages, dev_means, marker="o", color="#27ae60", linewidth=2.5,
              label="P(human at dev-optimal)")
    ax2.plot(stages, stat_means, marker="s", color="#e67e22", linewidth=2.5,
              label="P(human at stat-optimal)")
    for s, d in zip(stages, dev_means):
        ax2.text(s, d + 0.025, f"{d:.0%}", ha="center", fontsize=8,
                  color="#27ae60", fontweight="bold")
    for s, st in zip(stages, stat_means):
        ax2.text(s, st - 0.05, f"{st:.0%}", ha="center", fontsize=8,
                  color="#e67e22", fontweight="bold")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Fraction of humans at this role", color="black")
    ax2.legend(loc="upper left", fontsize=9)

    ax.set_xticks(stages)
    ax.set_xticklabels([f"Stage {s}" for s in stages])
    ax.set_ylabel("Role distance from deviate-optimal (jittered)")
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_yticklabels(["−2", "−1", "0", "+1", "+2"], fontsize=8)
    ax.set_ylim(-2.5, 2.5)

    n_reached = sum(tr["reached_dev"] for tr in trajectories)
    ax.set_title(
        f"Bot rounds — the human's role across stages, one faint line per team "
        f"(n={len(trajectories)} teams).\n"
        f"Green = team ended at deviate-optimal ({n_reached}/{len(trajectories)} = "
        f"{n_reached / len(trajectories):.0%}); orange = ended elsewhere.",
        fontsize=11,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "fig_bot_role_choice.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ──────────────────────────────────────────────────────────────────────
# Fig 7 — inference confusion matrices
# ──────────────────────────────────────────────────────────────────────

def fig_inference_accuracy(by_export):
    """3x3 confusion matrices (rows = true role, cols = inferred role) for
    human and bot rounds, plus per-stage accuracy as a small inset bar."""
    human_conf = np.zeros((3, 3), dtype=int)
    bot_conf = np.zeros((3, 3), dtype=int)
    human_by_stage = defaultdict(lambda: [0, 0])
    bot_by_stage = defaultdict(lambda: [0, 0])

    # Human teams
    teams = defaultdict(list)
    for batch, prs in by_export.items():
        for pr in prs:
            if pr.round.round_type == "human":
                teams[(pr.game_id, pr.round.round_number)].append(pr)
    teams = {k: sorted(v, key=lambda p: p.player_id)
             for k, v in teams.items() if len(v) == 3}
    for team_prs in teams.values():
        player_roles = {pr.player_id: [s.role_idx for s in pr.round.stages]
                        for pr in team_prs}
        for pr in team_prs:
            for si, stage in enumerate(pr.round.stages):
                if si == 0 or not stage.inferred_roles:
                    continue
                stage_num = si + 1
                for target_pos, inferred_role in stage.inferred_roles.items():
                    if (target_pos not in player_roles
                            or si - 1 >= len(player_roles[target_pos])):
                        continue
                    true_role = player_roles[target_pos][si - 1]
                    human_conf[true_role, inferred_role] += 1
                    human_by_stage[stage_num][0] += int(inferred_role == true_role)
                    human_by_stage[stage_num][1] += 1

    # Bot rounds
    for batch, prs in by_export.items():
        for pr in prs:
            if pr.round.round_type != "bot":
                continue
            layout = build_bot_round_layout(pr)
            for si, stage in enumerate(pr.round.stages):
                if si == 0 or not stage.inferred_roles:
                    continue
                stage_num = si + 1
                for target_pos, inferred_role in stage.inferred_roles.items():
                    if target_pos not in layout.bot_role_map:
                        continue
                    true_role = layout.bot_role_map[target_pos]
                    bot_conf[true_role, inferred_role] += 1
                    bot_by_stage[stage_num][0] += int(inferred_role == true_role)
                    bot_by_stage[stage_num][1] += 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, conf, by_stage, title, color in [
        (axes[0], human_conf, human_by_stage, "Human rounds", "#2ecc71"),
        (axes[1], bot_conf, bot_by_stage, "Bot rounds", "#e74c3c"),
    ]:
        row_sums = conf.sum(axis=1, keepdims=True)
        conf_norm = np.divide(conf.astype(float), row_sums, where=row_sums > 0)
        im = ax.imshow(conf_norm, cmap="Greens", vmin=0, vmax=1, aspect="equal")
        for r in range(3):
            for c in range(3):
                v = conf_norm[r, c]
                n = int(conf[r, c])
                txt_color = "white" if v > 0.5 else "black"
                ax.text(c, r, f"{v:.0%}\n(n={n})", ha="center", va="center",
                        color=txt_color, fontsize=9)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(ROLE_NAMES)
        ax.set_yticklabels(ROLE_NAMES)
        ax.set_xlabel("Inferred role")
        ax.set_ylabel("Actual role at previous stage")
        diag_n = int(np.trace(conf))
        total = int(conf.sum())
        ax.set_title(
            f"{title}\noverall accuracy = {diag_n}/{total} "
            f"= {diag_n / total:.0%} (chance 33%)",
            fontsize=10, fontweight="bold",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Per-stage accuracy as inline annotation
        stages = sorted(by_stage.keys())
        rates = [by_stage[s][0] / by_stage[s][1] if by_stage[s][1] else 0
                 for s in stages]
        anno = "  per-stage: " + " | ".join(
            f"S{s}={r:.0%}" for s, r in zip(stages, rates))
        ax.text(0.5, -0.32, anno, transform=ax.transAxes, fontsize=8,
                ha="center", style="italic")

    fig.suptitle(
        "Inference confusion matrices — rows = actual role at the stage "
        "the player was guessing about, columns = the player's guess.\n"
        "Diagonal = correct inferences; off-diagonal cells reveal which "
        "errors are systematic.",
        fontsize=11, y=1.04,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "fig_inference_accuracy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 66)
    print("Behavioural figures — redesigned for texture")
    print("=" * 66)
    print("\nLoading and filtering data...")
    by_export, dropout_games = load_clean_records()
    n_clean = sum(len(prs) for prs in by_export.values())
    print(f"  {n_clean} clean player-rounds ({len(dropout_games)} dropout games "
          f"excluded entirely)")

    print("\n[Fig 2] Win rates per batch")
    fig_outcome_win_rates(by_export)

    print("\n[Fig 3] Human-round role behaviour (stat-conditional + transitions)")
    fig_human_role_behavior(by_export)

    print("\n[Fig 6] Bot-round per-team trajectories")
    fig_bot_role_choice(by_export)

    print("\n[Fig 7] Inference confusion matrices")
    fig_inference_accuracy(by_export)


if __name__ == "__main__":
    main()
