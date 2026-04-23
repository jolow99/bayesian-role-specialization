"""Bot-round behavioral analysis for the 04-23 export.

Bot rounds: 1 human + 2 fixed-strategy AI bots. The bots play their
deviate-optimal roles; the human's stat-suggested role differs from the
deviate-optimal slot they need to fill. The research question is whether
humans (a) infer what the bots are doing and (b) deviate from their natural
stat role to fill the missing deviate-optimal slot.

This script reports per-stage learning curves for deviation, inference
accuracy, and outcomes. 04-23 numbers are compared against 03-18 as a
sanity reference.

Outputs:
  comparison_table.md  — markdown summary
  figures/              — per-stage curves
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ANALYSIS_ROOT))

from shared import EXPORTS_DIR
from shared.constants import ROLE_NAMES, ROLE_SHORT
from shared.data_loading import load_all_exports

EXPORTS = {
    "04-23": "bayesian-role-specialization-2026-04-23-09-12-55",
    "03-18": "bayesian-role-specialization-2026-03-18-15-47-09",
}


def bot_round_layout(pr):
    """Return (human_pos, bot_role_map, stat_opt_human, deviate_opt_human).

    Encodes the position-vs-logical mapping documented in CLAUDE.md →
    "Bot Round Ground Truth". Returns None when the config is malformed.
    """
    cfg = pr.round.config or {}
    bots = cfg.get("botPlayers") or []
    if len(bots) < 2:
        return None
    pid = pr.player_id
    others = sorted(i for i in range(3) if i != pid)
    bot_role_map = {
        others[0]: int(bots[0]["strategy"]["role"]),
        others[1]: int(bots[1]["strategy"]["role"]),
    }
    optimal_roles = pr.round.optimal_roles or []
    deviate_roles = pr.round.deviate_roles or []
    if len(optimal_roles) < 1 or len(deviate_roles) < 1:
        return None
    return pid, bot_role_map, int(optimal_roles[0]), int(deviate_roles[0])


def collect(records):
    """Flatten bot-round behavior into per-stage observations."""
    outcomes = Counter()
    # per stage: choice category + inference accuracy
    by_stage_choice = defaultdict(Counter)        # stage -> Counter({deviate, stat, other})
    by_stage_infer = defaultdict(lambda: [0, 0])  # stage -> [correct, total]
    n_rounds = 0
    n_ever_deviated = 0
    n_final_deviated = 0  # deviated on final stage of round

    for pr in records:
        if pr.round.round_type != "bot" or pr.is_dropout:
            continue
        layout = bot_round_layout(pr)
        if layout is None:
            continue
        pid, bot_role_map, stat_opt, dev_opt = layout
        outcomes[pr.round.outcome] += 1
        n_rounds += 1

        ever = False
        last_stage_role = None
        for i, st in enumerate(pr.round.stages):
            if st.is_bot or st.role_idx is None:
                continue
            stage_n = i + 1  # 1-based logical stage
            if st.role_idx == dev_opt and dev_opt != stat_opt:
                cat = "deviate"
                ever = True
            elif st.role_idx == stat_opt:
                cat = "stat"
            else:
                cat = "other"
            by_stage_choice[stage_n][cat] += 1
            last_stage_role = st.role_idx

            # Inferences made at stage_n are about stage_n - 1; bots never
            # switch, so the target's role is bot_role_map[target_pos]
            # constant. We can score every inference present.
            if i > 0 and st.inferred_roles:
                for tgt_pos, pred_role in st.inferred_roles.items():
                    if tgt_pos == pid or tgt_pos not in bot_role_map:
                        continue
                    if pred_role is None:
                        continue
                    by_stage_infer[stage_n][1] += 1
                    if int(pred_role) == bot_role_map[tgt_pos]:
                        by_stage_infer[stage_n][0] += 1
        if ever:
            n_ever_deviated += 1
        if last_stage_role is not None and last_stage_role == dev_opt and dev_opt != stat_opt:
            n_final_deviated += 1

    return {
        "n_rounds": n_rounds,
        "outcomes": outcomes,
        "by_stage_choice": dict(by_stage_choice),
        "by_stage_infer": {k: tuple(v) for k, v in by_stage_infer.items()},
        "n_ever_deviated": n_ever_deviated,
        "n_final_deviated": n_final_deviated,
    }


def stage_choice_pcts(by_stage_choice):
    out = {}
    for s, c in sorted(by_stage_choice.items()):
        n = sum(c.values())
        out[s] = {
            "n": n,
            "deviate": c.get("deviate", 0) / n if n else 0,
            "stat": c.get("stat", 0) / n if n else 0,
            "other": c.get("other", 0) / n if n else 0,
        }
    return out


def main():
    summaries = {}
    for label, name in EXPORTS.items():
        records = load_all_exports(data_dirs=[EXPORTS_DIR / name])
        summaries[label] = collect(records)

    # ─── Figure: per-stage deviation rate ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (label, s) in zip(axes, summaries.items()):
        pcts = stage_choice_pcts(s["by_stage_choice"])
        stages = list(pcts.keys())
        dev = [pcts[k]["deviate"] for k in stages]
        stat = [pcts[k]["stat"] for k in stages]
        other = [pcts[k]["other"] for k in stages]
        ns = [pcts[k]["n"] for k in stages]
        ax.plot(stages, dev, "o-", label="deviate-optimal", color="#2ca02c")
        ax.plot(stages, stat, "s-", label="stat-optimal", color="#d62728")
        ax.plot(stages, other, "^--", label="other", color="#7f7f7f")
        for x, n in zip(stages, ns):
            ax.text(x, -0.05, f"n={n}", ha="center", fontsize=7)
        ax.set_title(f"{label}: human role vs stage (bot rounds)")
        ax.set_xlabel("stage (1=first)")
        ax.set_ylabel("fraction of human-stage choices")
        ax.set_ylim(-0.1, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / "figures" / "deviation_curve.png", dpi=140)
    plt.close(fig)

    # ─── Figure: per-stage human→bot inference accuracy ────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, s in summaries.items():
        items = sorted(s["by_stage_infer"].items())
        stages = [k for k, _ in items]
        accs = [c / t if t else float("nan") for _, (c, t) in items]
        ns = [t for _, (c, t) in items]
        ax.plot(stages, accs, "o-", label=f"{label} (Σn={sum(ns)})")
    ax.axhline(1 / 3, ls="--", color="gray", alpha=0.7, label="chance")
    ax.set_xlabel("stage")
    ax.set_ylabel("inference accuracy (human → bot)")
    ax.set_title("Bot-round inference accuracy by stage")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / "figures" / "inference_curve.png", dpi=140)
    plt.close(fig)

    # ─── Markdown summary ──────────────────────────────────────────────
    lines = ["# Bot-Round Behavioral Analysis (04-23 vs 03-18 reference)", ""]
    lines.append("Per-stage learning curves for the human player only. Each "
                 "bot round has 2 fixed-strategy AI bots; the human's stat-"
                 "suggested role differs from the deviate-optimal slot they "
                 "should fill (see CLAUDE.md → Bot Round Ground Truth).")
    lines.append("")

    lines.append("## Round outcomes")
    lines.append("")
    lines.append("| Export | n rounds | WIN | LOSE | TIMEOUT |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, s in summaries.items():
        n = s["n_rounds"]
        oc = s["outcomes"]
        lines.append(f"| {label} | {n} | "
                     f"{oc['WIN']} ({100*oc['WIN']/n:.0f}%) | "
                     f"{oc['LOSE']} ({100*oc['LOSE']/n:.0f}%) | "
                     f"{oc['TIMEOUT']} ({100*oc['TIMEOUT']/n:.0f}%) |")
    lines.append("")

    lines.append("## Did the human deviate at all?")
    lines.append("")
    lines.append("| Export | rounds | ever deviated | deviated on final stage |")
    lines.append("|---|---:|---:|---:|")
    for label, s in summaries.items():
        n = s["n_rounds"]
        lines.append(f"| {label} | {n} | "
                     f"{s['n_ever_deviated']} ({100*s['n_ever_deviated']/n:.0f}%) | "
                     f"{s['n_final_deviated']} ({100*s['n_final_deviated']/n:.0f}%) |")
    lines.append("")

    lines.append("## Per-stage human role choice")
    lines.append("")
    for label, s in summaries.items():
        pcts = stage_choice_pcts(s["by_stage_choice"])
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| Stage | n | % deviate-opt | % stat-opt | % other |")
        lines.append("|---:|---:|---:|---:|---:|")
        for k, v in pcts.items():
            lines.append(f"| {k} | {v['n']} | {100*v['deviate']:.0f}% | "
                         f"{100*v['stat']:.0f}% | {100*v['other']:.0f}% |")
        lines.append("")

    lines.append("## Per-stage inference accuracy (human → bot)")
    lines.append("")
    for label, s in summaries.items():
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| Stage | n inferences | % correct |")
        lines.append("|---:|---:|---:|")
        for stage, (c, t) in sorted(s["by_stage_infer"].items()):
            pct = 100 * c / t if t else 0
            lines.append(f"| {stage} | {t} | {pct:.0f}% |")
        lines.append("")

    lines.append("Chance baseline for inference = 33%.")
    lines.append("")

    out_path = SCRIPT_DIR / "comparison_table.md"
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")
    print(f"Wrote {SCRIPT_DIR / 'figures'}/*.png")


if __name__ == "__main__":
    main()
