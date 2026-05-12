"""Current-export behavioral analysis.

This script is descriptive only: it reports what participants did in the
currently available exports without fitting computational model parameters.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_ROOT = SCRIPT_DIR.parent.parent
REPO_ROOT = ANALYSIS_ROOT.parent
sys.path.insert(0, str(ANALYSIS_ROOT))

from shared import EXPORTS_DIR
from shared.constants import ROLE_SHORT
from shared.data_loading import build_bot_round_layout, load_all_exports


FIGURES_DIR = SCRIPT_DIR / "figures"
TABLES_DIR = SCRIPT_DIR / "tables"
FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)


def pct(numer: int | float, denom: int | float) -> float:
    return float(numer) / float(denom) if denom else 0.0


def pct_str(numer: int | float, denom: int | float) -> str:
    return f"{100 * pct(numer, denom):.0f}%"


def label_exports(export_names: list[str]) -> dict[str, str]:
    """Map long export folder names to compact labels."""
    dates = {}
    total_by_date: Counter[str] = Counter()
    for name in sorted(export_names):
        parts = name.split("-")
        if len(parts) >= 7:
            date = f"{parts[4]}-{parts[5]}"
        else:
            date = name
        dates[name] = date
        total_by_date[date] += 1

    seen_by_date: Counter[str] = Counter()
    labels = {}
    for name in sorted(export_names):
        date = dates[name]
        seen_by_date[date] += 1
        suffix = (
            chr(ord("a") + seen_by_date[date] - 1)
            if total_by_date[date] > 1
            else ""
        )
        labels[name] = f"{date}{suffix}"
    return labels


def with_all(records):
    """Return per-export records plus a pooled All group."""
    grouped = defaultdict(list)
    for pr in records:
        grouped[pr.export_name].append(pr)
    grouped["All"] = list(records)
    return grouped


def outcome_counts(outcomes: Counter) -> dict:
    total = sum(outcomes.values())
    return {
        "n": total,
        "WIN": outcomes.get("WIN", 0),
        "LOSE": outcomes.get("LOSE", 0),
        "TIMEOUT": outcomes.get("TIMEOUT", 0),
        "win_rate": pct(outcomes.get("WIN", 0), total),
    }


def collect_scope(records, labels):
    rows = []
    for key, group in with_all(records).items():
        label = labels.get(key, key)
        games = {pr.game_id for pr in group}
        dropout_games = {pr.game_id for pr in group if pr.is_dropout}
        auto_stages = sum(
            1
            for pr in group
            for st in pr.round.stages
            if st.is_bot
        )
        rows.append({
            "export": label,
            "player_rounds": len(group),
            "games": len(games),
            "dropout_player_rounds": sum(1 for pr in group if pr.is_dropout),
            "dropout_games": len(dropout_games),
            "auto_submitted_stages": auto_stages,
            "human_player_rounds": sum(1 for pr in group if pr.round.round_type == "human"),
            "bot_player_rounds": sum(1 for pr in group if pr.round.round_type == "bot"),
        })
    return pd.DataFrame(rows)


def group_human_teams(records):
    teams = defaultdict(list)
    for pr in records:
        if pr.round.round_type == "human":
            teams[(pr.export_name, pr.game_id, pr.round.round_number)].append(pr)
    return {
        key: sorted(vals, key=lambda p: p.player_id)
        for key, vals in teams.items()
        if len(vals) == 3
    }


def collect_human(records, labels):
    team_groups = defaultdict(list)
    for key, team in group_human_teams(records).items():
        export, _, _ = key
        team_groups[export].append(team)
    team_groups["All"] = [team for teams in team_groups.values() for team in teams]

    outcome_rows = []
    stage_rows = []
    inference_rows = []

    for export, teams in team_groups.items():
        label = labels.get(export, export)
        clean_teams = [
            team for team in teams
            if not any(pr.is_dropout for pr in team)
            and not any(st.is_bot for pr in team for st in pr.round.stages)
        ]

        outcomes = Counter(team[0].round.outcome for team in clean_teams)
        row = {"export": label, **outcome_counts(outcomes)}
        row["complete_clean_team_rounds"] = len(clean_teams)
        outcome_rows.append(row)

        by_stage_opt = defaultdict(lambda: [0, 0])
        by_stage_switch = defaultdict(lambda: [0, 0])
        by_stage_infer = defaultdict(lambda: [0, 0])

        for team in clean_teams:
            player_roles = {
                pr.player_id: [st.role_idx for st in pr.round.stages]
                for pr in team
            }

            for pr in team:
                opt_roles = pr.round.optimal_roles or []
                opt_role = opt_roles[pr.player_id] if pr.player_id < len(opt_roles) else None
                prev_role = None
                for si, st in enumerate(pr.round.stages):
                    if st.role_idx is None or st.role_idx < 0:
                        continue
                    stage_n = si + 1
                    by_stage_opt[stage_n][1] += 1
                    if opt_role is not None and st.role_idx == int(opt_role):
                        by_stage_opt[stage_n][0] += 1
                    if prev_role is not None:
                        by_stage_switch[stage_n][1] += 1
                        if st.role_idx != prev_role:
                            by_stage_switch[stage_n][0] += 1
                    prev_role = st.role_idx

                    if si > 0 and st.inferred_roles:
                        for target_pos, inferred_role in st.inferred_roles.items():
                            if target_pos == pr.player_id:
                                continue
                            target_seq = player_roles.get(target_pos)
                            if not target_seq or si - 1 >= len(target_seq):
                                continue
                            true_role = target_seq[si - 1]
                            by_stage_infer[stage_n][1] += 1
                            if int(inferred_role) == int(true_role):
                                by_stage_infer[stage_n][0] += 1

        for stage, (correct, total) in sorted(by_stage_opt.items()):
            switches, switch_total = by_stage_switch.get(stage, (0, 0))
            stage_rows.append({
                "export": label,
                "stage": stage,
                "n_role_choices": total,
                "stat_optimal_rate": pct(correct, total),
                "n_switch_opportunities": switch_total,
                "switch_rate": pct(switches, switch_total),
            })

        for stage, (correct, total) in sorted(by_stage_infer.items()):
            inference_rows.append({
                "export": label,
                "stage": stage,
                "n_inferences": total,
                "accuracy": pct(correct, total),
            })

    return (
        pd.DataFrame(outcome_rows),
        pd.DataFrame(stage_rows),
        pd.DataFrame(inference_rows),
    )


def collect_bot(records, labels):
    record_groups = defaultdict(list)
    for pr in records:
        if pr.round.round_type == "bot":
            record_groups[pr.export_name].append(pr)
    record_groups["All"] = [pr for groups in record_groups.values() for pr in groups]

    outcome_rows = []
    stage_rows = []
    inference_rows = []

    for export, group in record_groups.items():
        label = labels.get(export, export)
        outcomes = Counter()
        by_stage_choice = defaultdict(Counter)
        by_stage_infer = defaultdict(lambda: [0, 0])
        n_rounds = 0
        n_ever_deviated = 0
        n_final_deviated = 0

        for pr in group:
            if pr.is_dropout:
                continue
            try:
                layout = build_bot_round_layout(pr)
            except ValueError:
                continue

            opt = pr.round.optimal_roles or []
            dev = pr.round.deviate_roles or []
            if not opt or not dev:
                continue
            stat_opt = int(opt[0])
            dev_opt = int(dev[0])

            outcomes[pr.round.outcome] += 1
            n_rounds += 1
            ever = False
            last_stage_role = None

            for si, st in enumerate(pr.round.stages):
                if st.is_bot or st.role_idx is None or st.role_idx < 0:
                    continue
                stage_n = si + 1
                if st.role_idx == dev_opt and dev_opt != stat_opt:
                    cat = "deviate"
                    ever = True
                elif st.role_idx == stat_opt:
                    cat = "stat"
                else:
                    cat = "other"
                by_stage_choice[stage_n][cat] += 1
                last_stage_role = st.role_idx

                if si > 0 and st.inferred_roles:
                    for target_pos, pred_role in st.inferred_roles.items():
                        if target_pos == layout.pid or target_pos not in layout.bot_role_map:
                            continue
                        by_stage_infer[stage_n][1] += 1
                        if int(pred_role) == int(layout.bot_role_map[target_pos]):
                            by_stage_infer[stage_n][0] += 1

            if ever:
                n_ever_deviated += 1
            if last_stage_role is not None and last_stage_role == dev_opt and dev_opt != stat_opt:
                n_final_deviated += 1

        out = {"export": label, **outcome_counts(outcomes)}
        out["usable_bot_rounds"] = n_rounds
        out["ever_deviated_rate"] = pct(n_ever_deviated, n_rounds)
        out["final_deviated_rate"] = pct(n_final_deviated, n_rounds)
        outcome_rows.append(out)

        for stage, counts in sorted(by_stage_choice.items()):
            total = sum(counts.values())
            stage_rows.append({
                "export": label,
                "stage": stage,
                "n_role_choices": total,
                "deviate_rate": pct(counts.get("deviate", 0), total),
                "stat_rate": pct(counts.get("stat", 0), total),
                "other_rate": pct(counts.get("other", 0), total),
            })

        for stage, (correct, total) in sorted(by_stage_infer.items()):
            inference_rows.append({
                "export": label,
                "stage": stage,
                "n_inferences": total,
                "accuracy": pct(correct, total),
            })

    return (
        pd.DataFrame(outcome_rows),
        pd.DataFrame(stage_rows),
        pd.DataFrame(inference_rows),
    )


def save_csvs(tables: dict[str, pd.DataFrame]):
    for name, df in tables.items():
        df.to_csv(TABLES_DIR / f"{name}.csv", index=False)


def ordered_exports(df: pd.DataFrame) -> list[str]:
    labels = [x for x in df["export"].dropna().unique().tolist() if x != "All"]
    return labels + (["All"] if "All" in set(df["export"]) else [])


def plot_outcomes(human_outcomes, bot_outcomes):
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = ordered_exports(bot_outcomes)
    x = np.arange(len(labels))
    width = 0.36
    human = human_outcomes.set_index("export").reindex(labels)
    bot = bot_outcomes.set_index("export").reindex(labels)
    ax.bar(x - width / 2, human["win_rate"], width, label="human rounds")
    ax.bar(x + width / 2, bot["win_rate"], width, label="bot rounds")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("win rate")
    ax.set_title("Win rate by export")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "outcome_win_rates.png", dpi=150)
    plt.close(fig)


def plot_bot_stage(bot_stage):
    labels = ordered_exports(bot_stage)
    fig, axes = plt.subplots(1, len(labels), figsize=(5.2 * len(labels), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, label in zip(axes, labels):
        sub = bot_stage[bot_stage["export"] == label].sort_values("stage")
        ax.plot(sub["stage"], sub["deviate_rate"], "o-", label="deviate-optimal")
        ax.plot(sub["stage"], sub["stat_rate"], "s-", label="stat-optimal")
        ax.plot(sub["stage"], sub["other_rate"], "^--", label="other")
        ax.set_title(label)
        ax.set_xlabel("stage")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("fraction of choices")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Bot rounds: human role choice by stage", y=1.03)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "bot_role_choice_by_stage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_human_stage(human_stage):
    labels = ordered_exports(human_stage)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for label in labels:
        sub = human_stage[human_stage["export"] == label].sort_values("stage")
        axes[0].plot(sub["stage"], sub["stat_optimal_rate"], "o-", label=label)
        axes[1].plot(sub["stage"], sub["switch_rate"], "o-", label=label)
    axes[0].set_title("Stat-optimal adherence")
    axes[1].set_title("Role switching")
    for ax in axes:
        ax.set_xlabel("stage")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("rate")
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "human_role_behavior_by_stage.png", dpi=150)
    plt.close(fig)


def plot_inference(human_inf, bot_inf):
    labels = ordered_exports(bot_inf)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for label in labels:
        h = human_inf[human_inf["export"] == label].sort_values("stage")
        b = bot_inf[bot_inf["export"] == label].sort_values("stage")
        if not h.empty:
            axes[0].plot(h["stage"], h["accuracy"], "o-", label=label)
        if not b.empty:
            axes[1].plot(b["stage"], b["accuracy"], "o-", label=label)
    for ax, title in zip(axes, ["Human rounds", "Bot rounds"]):
        ax.axhline(1 / 3, color="gray", ls="--", lw=1, label="chance")
        ax.set_title(title)
        ax.set_xlabel("stage")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("inference accuracy")
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "inference_accuracy_by_stage.png", dpi=150)
    plt.close(fig)


def plot_dropouts(scope):
    labels = ordered_exports(scope)
    sub = scope.set_index("export").reindex(labels)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, sub["dropout_games"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("games with dropout")
    ax.set_title("Dropout games by export")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dropout_games.png", dpi=150)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    if df.empty:
        return ["No data.", ""]
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df[columns].iterrows():
        vals = []
        for value in row:
            if isinstance(value, float):
                vals.append(f"{value:.3f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    return lines


def write_summary(tables):
    scope = tables["scope_summary"].copy()
    human_outcomes = tables["human_outcomes"].copy()
    bot_outcomes = tables["bot_outcomes"].copy()

    for df, cols in [
        (human_outcomes, ["win_rate"]),
        (bot_outcomes, ["win_rate", "ever_deviated_rate", "final_deviated_rate"]),
    ]:
        for col in cols:
            if col in df:
                df[col] = df[col].map(lambda x: f"{100 * x:.0f}%")

    lines = [
        "# Current Export Behavioral Analysis",
        "",
        "Descriptive behavioral analysis over the current exports in `analysis/data/exports/`.",
        "No model parameters were fit in this analysis.",
        "",
        "## Scope",
        "",
    ]
    lines += markdown_table(scope, [
        "export", "games", "player_rounds", "human_player_rounds",
        "bot_player_rounds", "dropout_games", "dropout_player_rounds",
    ])
    lines += ["## Human Rounds", ""]
    lines += markdown_table(human_outcomes, [
        "export", "complete_clean_team_rounds", "WIN", "LOSE", "TIMEOUT", "win_rate",
    ])
    lines += ["## Bot Rounds", ""]
    lines += markdown_table(bot_outcomes, [
        "export", "usable_bot_rounds", "WIN", "LOSE", "TIMEOUT",
        "win_rate", "ever_deviated_rate", "final_deviated_rate",
    ])
    lines += [
        "## Figures",
        "",
        "- `figures/outcome_win_rates.png`",
        "- `figures/human_role_behavior_by_stage.png`",
        "- `figures/bot_role_choice_by_stage.png`",
        "- `figures/inference_accuracy_by_stage.png`",
        "- `figures/dropout_games.png`",
        "",
        "## Tables",
        "",
        "- `tables/scope_summary.csv`",
        "- `tables/human_outcomes.csv`",
        "- `tables/human_stage_behavior.csv`",
        "- `tables/human_inference.csv`",
        "- `tables/bot_outcomes.csv`",
        "- `tables/bot_stage_behavior.csv`",
        "- `tables/bot_inference.csv`",
        "",
    ]
    (SCRIPT_DIR / "summary.md").write_text("\n".join(lines))


def main():
    records = load_all_exports()
    export_names = sorted({pr.export_name for pr in records})
    labels = label_exports(export_names)

    scope = collect_scope(records, labels)
    human_outcomes, human_stage, human_inf = collect_human(records, labels)
    bot_outcomes, bot_stage, bot_inf = collect_bot(records, labels)

    tables = {
        "scope_summary": scope,
        "human_outcomes": human_outcomes,
        "human_stage_behavior": human_stage,
        "human_inference": human_inf,
        "bot_outcomes": bot_outcomes,
        "bot_stage_behavior": bot_stage,
        "bot_inference": bot_inf,
    }

    save_csvs(tables)
    plot_outcomes(human_outcomes, bot_outcomes)
    plot_human_stage(human_stage)
    plot_bot_stage(bot_stage)
    plot_inference(human_inf, bot_inf)
    plot_dropouts(scope)
    write_summary(tables)

    print(f"Wrote {SCRIPT_DIR / 'summary.md'}")
    print(f"Wrote {TABLES_DIR}/*.csv")
    print(f"Wrote {FIGURES_DIR}/*.png")


if __name__ == "__main__":
    main()
