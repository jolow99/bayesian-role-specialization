"""Per-treatment trajectory GIFs for bot rounds.

In bot rounds, 1 human plays alongside 2 fixed-strategy bots whose roles
never change. The experimental question is: did the human manage to
*deviate* from their stat-optimal role (the one their stats suggest)
to the deviate-optimal role (the one the team actually needs)?

Each frame answers, at a given stage:
    - Which role did each human pick (per-human grid)?
    - Was that pick the deviate-optimal target (succeeded), the
      stat-optimal lure (failed), or neither?
    - What fraction of humans landed on each role at this stage?
    - What does each model predict the human would do?

One GIF per treatment (= stat_profile × optimalDeviateRolesId). 15
treatments in the current exports.
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(METRIC_DIR))

from bot_pipeline import (  # noqa: E402
    load_bot_round_records, group_by_treatment,
    precompute_bot_trajectories, replay_bot_hp_timeline,
)
from bot_models import run_bot_predictions  # noqa: E402
from pipeline import strategy_from_params  # noqa: E402
from shared.constants import (  # noqa: E402
    ROLE_COLORS, ROLE_CHAR_TO_IDX, ROLE_SHORT, TURNS_PER_STAGE,
)

FIGURES_DIR = HERE / "figures_bot"
FRAMES_DIR = HERE / "frames_bot"

TAU_PRIOR = 4.638476144217848
EPSILON = 0.06241645791201582
MEMORY_STRATEGY = "drift_prior_0.500"

MODEL_SPECS = [
    {"name": "Bayesian-Belief", "short": "B", "color": "#1f3a93", "marker": "o"},
    {"name": "Bayesian-Value",  "short": "V", "color": "#c0392b", "marker": "D"},
    {"name": "Bayesian-Walk",   "short": "W", "color": "#16a085", "marker": "^"},
]

# Cosmetic: stat-optimal lure / deviate-optimal target colours, distinct
# from the F/T/M role colours.
STAT_OPT_COLOR = "#7a4f01"     # earthy red-brown
DEV_OPT_COLOR = "#1a7f37"      # darker green


# ──────────────────────────────────────────────────────────────────────
# Drawing primitives
# ──────────────────────────────────────────────────────────────────────

def draw_single_tile(ax, x, y, w, h, role_char, faded=False,
                      success=False, failure=False):
    """One role-coloured tile with an optional success / failure border."""
    alpha = 0.18 if faded else 1.0
    fill = ROLE_COLORS.get(role_char, "#888")
    ax.add_patch(mpatches.Rectangle(
        (x, y), w, h,
        facecolor=fill, edgecolor="none", alpha=alpha, lw=0,
    ))
    if success:
        ax.add_patch(mpatches.Rectangle(
            (x, y), w, h,
            facecolor="none", edgecolor=DEV_OPT_COLOR, lw=2.0, alpha=1.0,
        ))
    elif failure:
        ax.add_patch(mpatches.Rectangle(
            (x, y), w, h,
            facecolor="none", edgecolor=STAT_OPT_COLOR, lw=2.0,
            linestyle=(0, (3, 2)), alpha=1.0,
        ))


# ──────────────────────────────────────────────────────────────────────
# Frame renderer
# ──────────────────────────────────────────────────────────────────────

def render_frame(treatment_id, records, predictions, stage, out_path):
    stat_profile, dev_id = treatment_id.split("__")
    stat_opt = records[0]["human_stat_optimal"]
    dev_opt = records[0]["human_deviate_optimal"]
    n_humans = len(records)
    n_stages_max = max(len(r["stage_roles"]) for r in records)

    # ── Figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11.5, 8.5), dpi=110)
    gs = GridSpec(
        nrows=3, ncols=1,
        height_ratios=[1.0, 2.4, 2.2],
        hspace=0.55,
        left=0.07, right=0.97, top=0.91, bottom=0.06,
    )
    ax_hp = fig.add_subplot(gs[0])
    ax_grid = fig.add_subplot(gs[1])
    ax_dist = fig.add_subplot(gs[2])

    # Title strip
    fig.text(0.07, 0.965,
             f"bot env  {stat_profile}  ·  {dev_id}",
             fontsize=13, fontweight="bold", family="monospace")

    # Bots — for display, pick any record (treatment-constant by definition)
    sample = records[0]
    bot_letters = [
        ROLE_SHORT[sample["bot_role_map"][p]]
        for p in sample["bot_positions"]
    ]
    fig.text(
        0.07, 0.940,
        f"human stat-opt {ROLE_SHORT[stat_opt]}  →  deviate-opt {ROLE_SHORT[dev_opt]}"
        f"    bots play {', '.join(bot_letters)}    n={n_humans} humans",
        fontsize=10, color="#444", family="monospace",
    )
    fig.text(0.97, 0.965, f"stage {stage + 1} of {n_stages_max}",
             fontsize=13, fontweight="bold", ha="right", family="monospace")

    # ── HP timeline (average across humans in this treatment) ─────────
    timelines = [replay_bot_hp_timeline(r) for r in records]
    max_T = max(len(t[0]) for t in timelines)
    avg_team = np.full(max_T, np.nan)
    avg_enemy = np.full(max_T, np.nan)
    for ti in range(max_T):
        ts = [t[0][ti] for t in timelines if ti < len(t[0])]
        es = [t[1][ti] for t in timelines if ti < len(t[1])]
        if ts:
            avg_team[ti] = np.mean(ts)
        if es:
            avg_enemy[ti] = np.mean(es)
    team_max_hp = sample["env_config"]["team_max_hp"]
    enemy_max_hp = sample["env_config"]["enemy_max_hp"]
    team_frac = avg_team / team_max_hp
    enemy_frac = avg_enemy / enemy_max_hp
    x = np.arange(max_T)
    ax_hp.plot(x, team_frac, color="#2c3e50", lw=2.0,
               label=f"team HP / {team_max_hp}")
    ax_hp.plot(x, enemy_frac, color="#c0392b", lw=2.0, ls="--",
               label=f"enemy HP / {enemy_max_hp}")
    ax_hp.scatter(x, team_frac, color="#2c3e50", s=14, zorder=3)
    ax_hp.scatter(x, enemy_frac, color="#c0392b", s=14, zorder=3,
                  marker="s")
    stage_start = stage * TURNS_PER_STAGE
    stage_end = min(stage_start + TURNS_PER_STAGE, max_T - 1)
    ax_hp.axvspan(stage_start, stage_end, color="#f6c200", alpha=0.18, lw=0)
    for s_idx in range(n_stages_max + 1):
        ax_hp.axvline(s_idx * TURNS_PER_STAGE, color="#bbb",
                      lw=0.4, alpha=0.6)
    intents_all = timelines[0][2]
    for ti, intent in enumerate(intents_all):
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
        "HP across turns  (fraction of max; red ▼ = enemy attack)",
        fontsize=9, color="#444", loc="left", pad=10,
    )

    # ── Per-human role grid ──────────────────────────────────────────
    ax_grid.set_xlim(0, n_stages_max)
    ax_grid.set_ylim(n_humans, 0)
    ax_grid.set_yticks([])
    ax_grid.set_xticks([])
    for s in ("top", "right", "left", "bottom"):
        ax_grid.spines[s].set_visible(False)

    # Stage column headers
    for s_idx in range(n_stages_max):
        ax_grid.text(s_idx + 0.5, -0.45, f"stage {s_idx + 1}",
                     ha="center", va="bottom", fontsize=8.5, color="#444")

    # Current-stage highlight band
    ax_grid.add_patch(mpatches.Rectangle(
        (stage, 0), 1, n_humans, facecolor="#f6c200", alpha=0.13,
        edgecolor="none", lw=0,
    ))

    # Sort humans by full role-choice trajectory for visual coherence
    sorted_recs = sorted(
        records,
        key=lambda r: tuple(r["human_role_seq"][s] if s < len(r["human_role_seq"])
                            else 99 for s in range(n_stages_max)),
    )
    cell_pad = 0.08
    for hi, rec in enumerate(sorted_recs):
        for si in range(n_stages_max):
            if si >= len(rec["human_role_seq"]):
                continue
            r_idx = rec["human_role_seq"][si]
            r_char = ROLE_SHORT[r_idx]
            faded = si > stage
            draw_single_tile(
                ax_grid,
                x=si + cell_pad,
                y=hi + cell_pad,
                w=1 - 2 * cell_pad,
                h=1 - 2 * cell_pad,
                role_char=r_char,
                faded=faded,
                success=(r_idx == dev_opt) and not faded,
                failure=(r_idx == stat_opt) and not faded,
            )
        ax_grid.text(-0.08, hi + 0.5, f"{hi+1:>2}",
                     ha="right", va="center", fontsize=7.0, color="#888",
                     family="monospace")
    for s_idx in range(1, n_stages_max):
        ax_grid.axvline(s_idx, color="#fff", lw=1.2)

    ax_grid.set_title(
        "per-human role choice  "
        f"(green border = chose deviate-optimal {ROLE_SHORT[dev_opt]}, "
        f"dashed brown border = stuck on stat-optimal {ROLE_SHORT[stat_opt]})",
        fontsize=9, color="#444", loc="left", pad=18,
    )

    # Role legend
    legend_y = n_humans + 0.45
    legend_x = 0.05
    for label, key in [("F fighter", "F"), ("T tank", "T"), ("M medic", "M")]:
        ax_grid.add_patch(mpatches.Rectangle(
            (legend_x, legend_y), 0.12, 0.32,
            facecolor=ROLE_COLORS[key], edgecolor="none",
            clip_on=False,
        ))
        ax_grid.text(legend_x + 0.16, legend_y + 0.16, label,
                     fontsize=8, va="center", color="#333", clip_on=False)
        legend_x += 0.85

    # ── Stage distribution: 3 horizontal bars + model dots ───────────
    role_chars = ["F", "T", "M"]
    role_indices = [0, 1, 2]
    counts = [0, 0, 0]
    n_played = 0
    for r in records:
        if stage < len(r["human_role_seq"]):
            counts[r["human_role_seq"][stage]] += 1
            n_played += 1
    freq = np.array(counts) / max(1, n_played)

    y_pos = np.arange(3)
    bar_colors = [ROLE_COLORS[c] for c in role_chars]
    ax_dist.barh(y_pos, freq, height=0.62,
                 color=bar_colors, edgecolor="#444", lw=0.5,
                 alpha=0.32, zorder=2)
    for yi, (cnt, fr) in enumerate(zip(counts, freq)):
        ax_dist.text(fr + 0.012, yi, f"{cnt}/{n_played}",
                     va="center", fontsize=8.5, color="#333")

    # Average model marginals across humans in this treatment with predictions
    for spec in MODEL_SPECS:
        short = spec["short"]
        per_human = predictions[short]
        all_at_stage = []
        for ph in per_human:
            if ph is None:
                continue
            if stage < len(ph):
                all_at_stage.append(ph[stage]["human_marginal"])
        if not all_at_stage:
            continue
        mean_marg = np.mean(all_at_stage, axis=0)
        n_covered = len(all_at_stage)
        ax_dist.scatter(
            mean_marg, y_pos,
            marker=spec["marker"], s=64,
            facecolor=spec["color"], edgecolor="white", lw=0.9,
            label=f"{spec['name']} (n={n_covered})", zorder=5,
        )

    # Stat-optimal and deviate-optimal markers on the y-axis
    ax_dist.axhline(stat_opt, color=STAT_OPT_COLOR, lw=2.0,
                    alpha=0.30, zorder=1)
    ax_dist.axhline(dev_opt, color=DEV_OPT_COLOR, lw=2.0,
                    alpha=0.30, zorder=1)

    # Combine role + dev-opt/stat-opt label into a single y-tick label
    y_labels = []
    for role_idx, c in zip(role_indices, role_chars):
        tag = ""
        if role_idx == dev_opt:
            tag = "  dev-opt ▶"
        elif role_idx == stat_opt:
            tag = "  stat-opt ▶"
        y_labels.append(f"{c}{tag}")
    ax_dist.set_yticks(y_pos)
    ax_dist.set_yticklabels(y_labels, family="monospace", fontsize=10)
    for tick, role_idx in zip(ax_dist.get_yticklabels(), role_indices):
        if role_idx == dev_opt:
            tick.set_color(DEV_OPT_COLOR)
            tick.set_fontweight("bold")
        elif role_idx == stat_opt:
            tick.set_color(STAT_OPT_COLOR)
            tick.set_fontweight("bold")
    ax_dist.invert_yaxis()
    upper = max(0.65, float(freq.max() + 0.18) if len(freq) else 0.65)
    ax_dist.set_xlim(0, upper)
    ax_dist.set_xlabel("probability / empirical frequency", fontsize=9,
                       color="#444")
    for s in ("top", "right"):
        ax_dist.spines[s].set_visible(False)
    ax_dist.spines["left"].set_color("#aaa")
    ax_dist.spines["bottom"].set_color("#aaa")
    ax_dist.tick_params(colors="#444")
    ax_dist.set_title(
        f"stage {stage + 1} role-choice distribution  "
        f"(bars = humans, dots = model predictions)",
        fontsize=10, color="#222", loc="left", pad=6,
    )
    ax_dist.legend(loc="lower right", fontsize=8.5, frameon=False,
                   scatterpoints=1, handletextpad=0.3, borderpad=0.2)

    fig.savefig(out_path, dpi=110, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Per-treatment predictions + GIF stitching
# ──────────────────────────────────────────────────────────────────────

def per_treatment_predictions(records):
    strategy = strategy_from_params(MEMORY_STRATEGY, None, None)
    from bot_pipeline import precompute_bot_trajectories
    trajectories = precompute_bot_trajectories(
        records, TAU_PRIOR, EPSILON, strategy)
    return run_bot_predictions(records, trajectories)


def make_gif(treatment_id, records, predictions):
    n_stages = max(len(r["stage_roles"]) for r in records)
    tdir = FRAMES_DIR / treatment_id
    tdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for stage in range(n_stages):
        fp = tdir / f"stage_{stage:02d}.png"
        render_frame(treatment_id, records, predictions, stage, fp)
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
    out = FIGURES_DIR / f"{treatment_id}.gif"
    padded[0].save(
        out,
        save_all=True,
        append_images=padded[1:] + [padded[-1]],
        duration=[1600] * (len(padded) - 1) + [2200, 2200],
        loop=0,
        optimize=True,
    )
    return out


def main(min_humans: int = 10):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    records = load_bot_round_records(verbose=True)
    by_t = group_by_treatment(records)
    print(f"\n[build_bot_gifs] {len(by_t)} treatments. "
          f"keeping treatments with ≥ {min_humans} humans.")
    for tid, recs in sorted(by_t.items()):
        if len(recs) < min_humans:
            print(f"  skip {tid}: only {len(recs)} humans")
            continue
        print(f"  → {tid}  ({len(recs)} humans)")
        preds = per_treatment_predictions(recs)
        gif = make_gif(tid, recs, preds)
        if gif:
            print(f"     saved {gif.relative_to(EXP_ROOT.parent)}")


if __name__ == "__main__":
    main()
