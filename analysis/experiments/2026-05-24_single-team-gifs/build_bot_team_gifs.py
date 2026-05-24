"""One GIF per bot-round human, telling that human's specific story.

Mirrors `build_team_gifs.py` but for bot rounds, where the "team" is a
single human + 2 fixed-strategy bots. The narrative question is:
*did this particular human manage to deviate from their stat-optimal
role to the deviate-optimal one their team needs?*

Frame layout:

    ┌──────────────────────────────────────────────────────────────┐
    │ human <id>  •  bot env <treatment_id>                       │
    │ stat-opt X → dev-opt Y   bots play [Z, W]   stage s of S    │
    ├──────────────────────────────────────────────────────────────┤
    │ HP across turns (this human's actual trajectory)            │
    ├──────────────────────────────────────────────────────────────┤
    │ Role choices (1 row × stages); each cell = human's role     │
    │ tile; green border = chose dev-opt; dashed brown border =   │
    │ stuck on stat-opt; current stage emphasised                 │
    ├──────────────────────────────────────────────────────────────┤
    │ Stage-s 3-role distribution: each role gets one row;        │
    │ three horizontal bars (one per model) for predicted prob;   │
    │ ★ tags chosen role; dev-opt / stat-opt rows highlighted     │
    └──────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
METRIC_DIR = EXP_ROOT / "2026-05-12-current-export-metric-comparison"
TEAM_AGG_DIR = EXP_ROOT / "2026-05-23_team-trajectory-gifs"
sys.path.insert(0, str(METRIC_DIR))
sys.path.insert(0, str(TEAM_AGG_DIR))

from bot_pipeline import (  # noqa: E402
    load_bot_round_records, precompute_bot_trajectories,
    replay_bot_hp_timeline,
)
from bot_models import run_bot_predictions  # noqa: E402
from pipeline import strategy_from_params  # noqa: E402
from shared.constants import (  # noqa: E402
    ROLE_COLORS, ROLE_SHORT, TURNS_PER_STAGE,
)

FIGURES_DIR = HERE / "figures_bot"
FRAMES_DIR = HERE / "frames_bot"

TAU_PRIOR = 4.638476144217848
EPSILON = 0.06241645791201582
MEMORY_STRATEGY = "drift_prior_0.500"

MODEL_SPECS = [
    {"name": "Bayesian-Belief", "short": "B", "color": "#1f3a93"},
    {"name": "Bayesian-Value",  "short": "V", "color": "#c0392b"},
    {"name": "Bayesian-Walk",   "short": "W", "color": "#16a085"},
]

STAT_OPT_COLOR = "#7a4f01"
DEV_OPT_COLOR = "#1a7f37"


def short_id(s: str) -> str:
    return s[-6:]


def single_human_predictions(record):
    """Returns {'B': [...], 'V': [...] or None, 'W': [...] or None}"""
    strategy = strategy_from_params(MEMORY_STRATEGY, None, None)
    trajectory = precompute_bot_trajectories(
        [record], TAU_PRIOR, EPSILON, strategy)
    all_preds = run_bot_predictions([record], trajectory)
    return {k: v[0] for k, v in all_preds.items()}


# ──────────────────────────────────────────────────────────────────────
# Frame renderer
# ──────────────────────────────────────────────────────────────────────

def render_frame(record, predictions, stage, out_path):
    treat = record["treatment_id"]
    stat_profile, dev_id = treat.split("__")
    stat_opt = record["human_stat_optimal"]
    dev_opt = record["human_deviate_optimal"]
    n_stages = len(record["stage_roles"])
    bot_letters = [
        ROLE_SHORT[record["bot_role_map"][p]]
        for p in record["bot_positions"]
    ]
    chosen_role = record["human_role_seq"][stage]

    fig = plt.figure(figsize=(11.5, 7.4), dpi=110)
    gs = GridSpec(
        nrows=3, ncols=1,
        height_ratios=[1.0, 1.0, 2.2],
        hspace=0.55,
        left=0.07, right=0.97, top=0.91, bottom=0.07,
    )
    ax_hp = fig.add_subplot(gs[0])
    ax_strip = fig.add_subplot(gs[1])
    ax_dist = fig.add_subplot(gs[2])

    # Title strip
    fig.text(0.07, 0.965,
             f"human {short_id(record['game_id'])}_r{record['round_number']}"
             f"   •   bot env {treat}",
             fontsize=13, fontweight="bold", family="monospace")
    fig.text(0.07, 0.940,
             f"stat-opt {ROLE_SHORT[stat_opt]}  →  dev-opt "
             f"{ROLE_SHORT[dev_opt]}"
             f"    bots play {', '.join(bot_letters)}",
             fontsize=10, color="#444", family="monospace")
    fig.text(0.97, 0.965, f"stage {stage + 1} of {n_stages}",
             fontsize=13, fontweight="bold", ha="right", family="monospace")

    # ── HP ────────────────────────────────────────────────────────────
    team_hp, enemy_hp, intents = replay_bot_hp_timeline(record)
    team_max_hp = record["env_config"]["team_max_hp"]
    enemy_max_hp = record["env_config"]["enemy_max_hp"]
    team_frac = np.array(team_hp) / team_max_hp
    enemy_frac = np.array(enemy_hp) / enemy_max_hp
    max_T = len(team_hp)
    x = np.arange(max_T)
    ax_hp.plot(x, team_frac, color="#2c3e50", lw=2.0,
               label=f"team HP / {team_max_hp}")
    ax_hp.plot(x, enemy_frac, color="#c0392b", lw=2.0, ls="--",
               label=f"enemy HP / {enemy_max_hp}")
    ax_hp.scatter(x, team_frac, color="#2c3e50", s=14, zorder=3)
    ax_hp.scatter(x, enemy_frac, color="#c0392b", s=14, zorder=3, marker="s")
    stage_start = stage * TURNS_PER_STAGE
    stage_end = min(stage_start + TURNS_PER_STAGE, max_T - 1)
    ax_hp.axvspan(stage_start, stage_end, color="#f6c200", alpha=0.18, lw=0)
    for s_idx in range(n_stages + 1):
        ax_hp.axvline(s_idx * TURNS_PER_STAGE, color="#bbb", lw=0.4, alpha=0.6)
    for ti, intent in enumerate(intents):
        if intent == 1:
            ax_hp.plot(ti + 1, 1.10, marker="v", ms=5.5,
                       color="#c0392b", clip_on=False)
    ax_hp.set_xlim(-0.3, max_T - 0.7)
    ax_hp.set_ylim(0, 1.08)
    ax_hp.set_yticks([0, 0.5, 1.0])
    ax_hp.set_yticklabels(["0", "½", "max"], fontsize=7.5, color="#666")
    stage_tick_pos = list(range(0, max_T, TURNS_PER_STAGE))
    if stage_tick_pos[-1] != max_T - 1:
        stage_tick_pos.append(max_T - 1)
    ax_hp.set_xticks(stage_tick_pos)
    labels = []
    for pos in stage_tick_pos:
        if pos == 0:
            labels.append("start")
        elif pos == max_T - 1:
            labels.append("end")
        else:
            labels.append(f"end s{pos // TURNS_PER_STAGE}")
    ax_hp.set_xticklabels(labels, fontsize=7.5, color="#666")
    ax_hp.tick_params(axis="x", length=0, pad=2)
    ax_hp.tick_params(axis="y", length=0, pad=2)
    for s in ("top", "right"):
        ax_hp.spines[s].set_visible(False)
    ax_hp.spines["left"].set_color("#ccc")
    ax_hp.spines["bottom"].set_color("#ccc")
    ax_hp.legend(loc="lower left", fontsize=8, frameon=False,
                 handlelength=1.4, borderpad=0.1, ncol=2)
    ax_hp.set_title(
        "HP across turns  (this game only; red ▼ = enemy attack)",
        fontsize=9, color="#444", loc="left", pad=10,
    )

    # ── Role-history strip ────────────────────────────────────────────
    ax_strip.set_xlim(0, n_stages)
    ax_strip.set_ylim(1, 0)
    ax_strip.set_yticks([])
    ax_strip.set_xticks([])
    for s in ("top", "right", "left", "bottom"):
        ax_strip.spines[s].set_visible(False)
    ax_strip.add_patch(mpatches.Rectangle(
        (stage, 0), 1, 1, facecolor="#f6c200", alpha=0.18,
        edgecolor="none", lw=0,
    ))
    cell_pad = 0.10
    for si in range(n_stages):
        ax_strip.text(si + 0.5, 1.18, f"stage {si + 1}",
                      ha="center", va="top", fontsize=8.5, color="#444")
        if si < len(record["human_role_seq"]):
            r_idx = record["human_role_seq"][si]
            r_char = ROLE_SHORT[r_idx]
            faded = si > stage
            alpha = 0.18 if faded else 1.0
            x0 = si + cell_pad
            y0 = cell_pad
            w = 1 - 2 * cell_pad
            h = 1 - 2 * cell_pad
            ax_strip.add_patch(mpatches.Rectangle(
                (x0, y0), w, h,
                facecolor=ROLE_COLORS[r_char], edgecolor="none",
                alpha=alpha, lw=0,
            ))
            if not faded:
                if r_idx == dev_opt:
                    ax_strip.add_patch(mpatches.Rectangle(
                        (x0, y0), w, h,
                        facecolor="none", edgecolor=DEV_OPT_COLOR, lw=2.2,
                    ))
                elif r_idx == stat_opt:
                    ax_strip.add_patch(mpatches.Rectangle(
                        (x0, y0), w, h,
                        facecolor="none", edgecolor=STAT_OPT_COLOR, lw=2.2,
                        linestyle=(0, (3, 2)),
                    ))
                if si == stage:
                    ax_strip.add_patch(mpatches.Rectangle(
                        (x0, y0), w, h,
                        facecolor="none", edgecolor="#222", lw=1.0,
                    ))
            ax_strip.text(si + 0.5, -0.06, r_char,
                          ha="center", va="bottom",
                          fontsize=10, family="monospace",
                          color="#222" if si <= stage else "#bbb")
    for s_idx in range(1, n_stages):
        ax_strip.axvline(s_idx, color="#fff", lw=1.2)
    ax_strip.set_title(
        f"human's role-choice history  "
        f"(green border = chose dev-opt {ROLE_SHORT[dev_opt]}, "
        f"dashed brown = stuck on stat-opt {ROLE_SHORT[stat_opt]})",
        fontsize=9, color="#444", loc="left", pad=14,
    )

    # ── Distribution panel: 3 rows (roles), grouped bars (models) ─────
    role_chars = ["F", "T", "M"]
    role_indices = [0, 1, 2]
    bar_h = 0.22
    gap = 0.08
    y_centers = np.arange(3)
    ymin = -0.5
    ymax = 2.5

    # Highlight rows: stat-opt brown band, dev-opt green band,
    # chosen role yellow band
    for yi, ri in enumerate(role_indices):
        if ri == chosen_role:
            ax_dist.add_patch(mpatches.Rectangle(
                (0, yi - 0.45), 1.0, 0.9,
                facecolor="#f6c200", alpha=0.18,
                edgecolor="none", lw=0, zorder=0,
            ))

    # Plot bars
    for mi, spec in enumerate(MODEL_SPECS):
        preds = predictions[spec["short"]]
        if preds is None:
            continue
        marg = preds[stage]["human_marginal"]
        for yi, ri in enumerate(role_indices):
            ay = yi - (1 - mi) * (bar_h + gap)
            ax_dist.barh(
                ay, float(marg[ri]), height=bar_h,
                color=spec["color"], edgecolor="white", lw=0.4,
                alpha=0.95, zorder=3,
            )

    # Stat-opt / dev-opt reference lines (subtle)
    ax_dist.axhline(stat_opt, color=STAT_OPT_COLOR, lw=1.6,
                    alpha=0.25, zorder=1)
    ax_dist.axhline(dev_opt, color=DEV_OPT_COLOR, lw=1.6,
                    alpha=0.25, zorder=1)

    # y-axis tick labels
    y_labels = []
    for ri, ch in zip(role_indices, role_chars):
        tag = ""
        if ri == dev_opt:
            tag += "  dev-opt ▶"
        elif ri == stat_opt:
            tag += "  stat-opt ▶"
        if ri == chosen_role:
            tag += "   ★ chose"
        y_labels.append(f"{ch}{tag}")
    ax_dist.set_yticks(y_centers)
    ax_dist.set_yticklabels(y_labels, family="monospace", fontsize=10)
    for tick, ri in zip(ax_dist.get_yticklabels(), role_indices):
        if ri == dev_opt:
            tick.set_color(DEV_OPT_COLOR); tick.set_fontweight("bold")
        elif ri == stat_opt:
            tick.set_color(STAT_OPT_COLOR); tick.set_fontweight("bold")
    ax_dist.invert_yaxis()
    ax_dist.set_xlim(0, 1.0)
    ax_dist.set_ylim(ymax, ymin)
    ax_dist.set_xlabel("predicted probability for the human's role",
                       fontsize=9, color="#444")
    for s in ("top", "right"):
        ax_dist.spines[s].set_visible(False)
    ax_dist.spines["left"].set_color("#aaa")
    ax_dist.spines["bottom"].set_color("#aaa")
    ax_dist.tick_params(colors="#444")

    legend_handles = []
    for spec in MODEL_SPECS:
        if predictions[spec["short"]] is None:
            continue
        legend_handles.append(
            mpatches.Patch(color=spec["color"], label=spec["name"])
        )
    ax_dist.legend(handles=legend_handles, loc="lower right",
                   fontsize=8.5, frameon=False, borderpad=0.2)
    ax_dist.set_title(
        f"stage {stage + 1}: each model's predicted role for THIS human",
        fontsize=10, color="#222", loc="left", pad=6,
    )

    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_gif(record):
    preds = single_human_predictions(record)
    n_stages = len(record["stage_roles"])
    tag = (f"{record['treatment_id']}__{short_id(record['game_id'])}"
           f"_r{record['round_number']}")
    fdir = FRAMES_DIR / tag
    fdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for stage in range(n_stages):
        fp = fdir / f"stage_{stage:02d}.png"
        render_frame(record, preds, stage, fp)
        paths.append(fp)
    frames = [Image.open(fp).convert("RGB") for fp in paths]
    if not frames:
        return None
    max_w = max(f.width for f in frames)
    max_h = max(f.height for f in frames)
    padded = []
    for f in frames:
        canvas = Image.new("RGB", (max_w, max_h), "white")
        canvas.paste(f, ((max_w - f.width) // 2, (max_h - f.height) // 2))
        padded.append(canvas)
    out = FIGURES_DIR / f"{tag}.gif"
    padded[0].save(
        out,
        save_all=True,
        append_images=padded[1:] + [padded[-1]],
        duration=[1400] * (len(padded) - 1) + [2000, 2000],
        loop=0,
        optimize=True,
    )
    return out


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    records = load_bot_round_records(verbose=True)
    print(f"\n[build_bot_team_gifs] rendering {len(records)} per-human GIFs")
    for i, rec in enumerate(records, 1):
        out = make_gif(rec)
        if i % 20 == 0 or i == len(records):
            print(f"  [{i}/{len(records)}] last: {out.name if out else 'skipped'}")


if __name__ == "__main__":
    main()
