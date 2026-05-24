"""One GIF per human team-round, telling that team's specific story.

The 2026-05-23 GIFs aggregate across teams within an environment (averaged
HP, empirical bars over combos, model predictions averaged across teams).
This experiment flips the perspective: each GIF follows a *single* team
through their actual stages, with HP and model predictions reflecting
that team's own trajectory.

Frame layout:

    ┌─────────────────────────────────────────────────────────────┐
    │ team <short id>  •  env <env_id>                            │
    │ stats <stat_profile>  •  optimal <combo>  •  stage s of S   │
    ├─────────────────────────────────────────────────────────────┤
    │ This team's HP across every turn                            │
    ├─────────────────────────────────────────────────────────────┤
    │ This team's role choices, one column per stage (3-tile      │
    │ glyph per cell). Past stages saturated, future faded.       │
    ├─────────────────────────────────────────────────────────────┤
    │ Stage-s prediction: each canonical combo gets one row.      │
    │ Three horizontal bars (one per model) show each model's     │
    │ predicted probability for that combo. ★ marks the combo     │
    │ this team actually chose at this stage; ◆ marks optimal.    │
    └─────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import sys
from collections import defaultdict
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

from pipeline import (  # noqa: E402
    load_human_team_records, precompute_trajectories, strategy_from_params,
)
from models import belief_factory, value_factory, walk_factory  # noqa: E402
from build_gifs import replay_hp_timeline  # noqa: E402
from shared.constants import (  # noqa: E402
    ROLE_COLORS, ROLE_CHAR_TO_IDX, ROLE_SHORT, TURNS_PER_STAGE,
)
from shared.parsing import canonical_combo, get_canonical_combos  # noqa: E402

FIGURES_DIR = HERE / "figures"
FRAMES_DIR = HERE / "frames"


TAU_PRIOR = 4.638476144217848
EPSILON = 0.06241645791201582
MEMORY_STRATEGY = "drift_prior_0.500"

MODEL_SPECS = [
    {
        "name": "Bayesian-Belief", "short": "B",
        "color": "#1f3a93",
        "factory_builder": lambda: belief_factory(),
    },
    {
        "name": "Bayesian-Value", "short": "V",
        "color": "#c0392b",
        "factory_builder": lambda: value_factory(tau_softmax=13.71598290227467),
    },
    {
        "name": "Bayesian-Walk", "short": "W",
        "color": "#16a085",
        "factory_builder": lambda: walk_factory(
            tau_softmax=7.20651148477258,
            epsilon_switch=0.5589855617201609,
        ),
    },
]


# ──────────────────────────────────────────────────────────────────────
# Predictions for a single team
# ──────────────────────────────────────────────────────────────────────

def single_team_predictions(record):
    """Return predictions[model_short] = list of per-stage prediction dicts
    for THIS team only."""
    strategy = strategy_from_params(MEMORY_STRATEGY, None, None)
    records_subset = [record]
    trajectories = precompute_trajectories(records_subset, TAU_PRIOR,
                                            EPSILON, strategy)
    out = {}
    for spec in MODEL_SPECS:
        factory = spec["factory_builder"]()
        predict_fn = factory(records_subset, trajectories)
        out[spec["short"]] = predict_fn(record)
    return out


# ──────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────

def draw_combo_glyph(ax, x, y, w, h, combo, faded=False, emphasised=False):
    """Three coloured tiles for a 3-letter role combo."""
    if combo is None:
        return
    n = len(combo)
    tile_w = w / n
    alpha = 0.18 if faded else 1.0
    for i, ch in enumerate(combo):
        c = ROLE_COLORS.get(ch, "#888")
        ax.add_patch(mpatches.Rectangle(
            (x + i * tile_w, y), tile_w * 0.92, h,
            facecolor=c, edgecolor="none", alpha=alpha, lw=0,
        ))
    if emphasised:
        ax.add_patch(mpatches.Rectangle(
            (x, y), w, h, facecolor="none",
            edgecolor="#222", lw=1.6,
        ))


def short_game_id(game_id: str) -> str:
    return game_id[-6:]


# ──────────────────────────────────────────────────────────────────────
# Frame renderer
# ──────────────────────────────────────────────────────────────────────

def render_frame(record, predictions, stage, out_path):
    env_id = record["env_id"]
    stat_profile = record["stat_profile"]
    optimal = record["optimal_roles"]
    canon_combos = get_canonical_combos(stat_profile)
    canon_optimal = canonical_combo(optimal, stat_profile)
    n_stages = len(record["stage_roles"])
    chosen_combo_canonical = canonical_combo(
        record["stage_roles"][stage], stat_profile,
    )

    fig = plt.figure(figsize=(11.5, 8.2), dpi=110)
    gs = GridSpec(
        nrows=3, ncols=1,
        height_ratios=[1.0, 1.0, 3.4],
        hspace=0.55,
        left=0.07, right=0.97, top=0.91, bottom=0.06,
    )
    ax_hp = fig.add_subplot(gs[0])
    ax_strip = fig.add_subplot(gs[1])
    ax_dist = fig.add_subplot(gs[2])

    # Title strip
    fig.text(0.07, 0.965,
             f"team {short_game_id(record['game_id'])}_r{record['round_number']}"
             f"   •   env {env_id}",
             fontsize=13, fontweight="bold", family="monospace")
    fig.text(0.07, 0.940,
             f"stats {stat_profile}   •   optimal {optimal}",
             fontsize=10, color="#444", family="monospace")
    fig.text(0.97, 0.965, f"stage {stage + 1} of {n_stages}",
             fontsize=13, fontweight="bold", ha="right", family="monospace")

    # ── HP timeline (this team only) ─────────────────────────────────
    team_hp, enemy_hp, intents = replay_hp_timeline(record)
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
        "HP across turns  (this team only; red ▼ = enemy attack)",
        fontsize=9, color="#444", loc="left", pad=10,
    )

    # ── Role-history strip (single row, one cell per stage) ──────────
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
    cell_pad = 0.08
    for si in range(n_stages):
        ax_strip.text(si + 0.5, 1.18, f"stage {si + 1}",
                      ha="center", va="top", fontsize=8.5, color="#444")
        if si < len(record["stage_roles"]):
            draw_combo_glyph(
                ax_strip,
                x=si + cell_pad,
                y=cell_pad,
                w=1 - 2 * cell_pad,
                h=1 - 2 * cell_pad,
                combo=record["stage_roles"][si],
                faded=(si > stage),
                emphasised=(si == stage),
            )
            # combo letters underneath
            ax_strip.text(si + 0.5, -0.06, record["stage_roles"][si],
                          ha="center", va="bottom",
                          fontsize=9, family="monospace",
                          color="#222" if si <= stage else "#bbb")
    for s_idx in range(1, n_stages):
        ax_strip.axvline(s_idx, color="#fff", lw=1.2)
    ax_strip.set_title(
        "team's role-combo history  (P1·P2·P3 tiles per stage)",
        fontsize=9, color="#444", loc="left", pad=14,
    )

    # ── Per-model prediction panel ───────────────────────────────────
    # Select combos to show: union of top-6 per model + actual chosen +
    # optimal. Keep stable y-axis across the GIF by computing the union
    # across all stages (so combos don't reshuffle frame-to-frame).
    all_to_show = set()
    for s_idx in range(n_stages):
        for spec in MODEL_SPECS:
            pred = predictions[spec["short"]][s_idx]
            # canonicalise then merge probabilities
            agg = defaultdict(float)
            for combo, prob in pred["predicted_dist"].items():
                agg[canonical_combo(combo, stat_profile)] += prob
            top = sorted(agg.items(), key=lambda kv: -kv[1])[:6]
            all_to_show.update(cc for cc, _ in top)
        all_to_show.add(canonical_combo(
            record["stage_roles"][s_idx], stat_profile))
    all_to_show.add(canon_optimal)
    # Order: by mean model probability at *current* stage, ties by string.
    cur_mean = {}
    for cc in all_to_show:
        ps = []
        for spec in MODEL_SPECS:
            pred = predictions[spec["short"]][stage]
            agg = defaultdict(float)
            for combo, prob in pred["predicted_dist"].items():
                agg[canonical_combo(combo, stat_profile)] += prob
            ps.append(agg.get(cc, 0.0))
        cur_mean[cc] = float(np.mean(ps))
    combos_ordered = sorted(all_to_show, key=lambda cc: (-cur_mean[cc], cc))

    # Per combo, draw 3 thin horizontal bars (one per model)
    n_combos = len(combos_ordered)
    bar_h = 0.22
    gap = 0.08   # vertical gap between models within a combo "block"
    block_h = 3 * bar_h + 2 * gap
    y_centers = np.arange(n_combos)
    ymin = -0.5
    ymax = n_combos - 0.5
    ax_dist.set_ylim(ymax, ymin)

    # Highlight rows
    for yi, cc in enumerate(combos_ordered):
        # team's chosen combo: yellow band
        if cc == chosen_combo_canonical:
            ax_dist.add_patch(mpatches.Rectangle(
                (0, yi - 0.45), 1.0, 0.9,
                facecolor="#f6c200", alpha=0.18,
                edgecolor="none", lw=0, zorder=0,
            ))

    # Bars per model
    for mi, spec in enumerate(MODEL_SPECS):
        pred = predictions[spec["short"]][stage]
        agg = defaultdict(float)
        for combo, prob in pred["predicted_dist"].items():
            agg[canonical_combo(combo, stat_profile)] += prob
        for yi, cc in enumerate(combos_ordered):
            p = agg.get(cc, 0.0)
            ay = yi - (1 - mi) * (bar_h + gap)  # mi=0 top, mi=2 bottom
            ax_dist.barh(
                ay, p, height=bar_h,
                color=spec["color"], edgecolor="white", lw=0.4,
                alpha=0.95, zorder=3,
            )

    # Combo labels (with star on chosen, ◆ on optimal)
    yticklabels = []
    for cc in combos_ordered:
        tag = ""
        if cc == chosen_combo_canonical:
            tag += " ★ chose"
        if cc == canon_optimal:
            tag += " ◆ optimal"
        yticklabels.append(f"{cc}{tag}")
    ax_dist.set_yticks(y_centers)
    ax_dist.set_yticklabels(yticklabels, family="monospace", fontsize=9)
    for tick, cc in zip(ax_dist.get_yticklabels(), combos_ordered):
        if cc == chosen_combo_canonical:
            tick.set_fontweight("bold")
            tick.set_color("#7a5d00")
        elif cc == canon_optimal:
            tick.set_color("#1a7f37")
            tick.set_fontweight("bold")

    # x-axis
    max_p = 0.0
    for spec in MODEL_SPECS:
        pred = predictions[spec["short"]][stage]
        agg = defaultdict(float)
        for combo, prob in pred["predicted_dist"].items():
            agg[canonical_combo(combo, stat_profile)] += prob
        for cc in combos_ordered:
            max_p = max(max_p, agg.get(cc, 0.0))
    ax_dist.set_xlim(0, max(0.4, max_p * 1.15))
    ax_dist.set_xlabel("predicted probability", fontsize=9, color="#444")
    for s in ("top", "right"):
        ax_dist.spines[s].set_visible(False)
    ax_dist.spines["left"].set_color("#aaa")
    ax_dist.spines["bottom"].set_color("#aaa")
    ax_dist.tick_params(colors="#444")

    # Legend
    legend_handles = [
        mpatches.Patch(color=spec["color"], label=spec["name"])
        for spec in MODEL_SPECS
    ]
    ax_dist.legend(handles=legend_handles, loc="lower right",
                    fontsize=8.5, frameon=False, borderpad=0.2)
    ax_dist.set_title(
        f"stage {stage + 1}: each model's predicted distribution for THIS team",
        fontsize=10, color="#222", loc="left", pad=6,
    )

    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# GIF assembly
# ──────────────────────────────────────────────────────────────────────

def make_gif(record):
    preds = single_team_predictions(record)
    n_stages = len(record["stage_roles"])
    tag = f"{record['env_id']}__{short_game_id(record['game_id'])}_r{record['round_number']}"
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
    records = load_human_team_records(verbose=True)
    print(f"\n[build_team_gifs] rendering {len(records)} per-team GIFs")
    for i, rec in enumerate(records, 1):
        out = make_gif(rec)
        if i % 20 == 0 or i == len(records):
            print(f"  [{i}/{len(records)}] last: {out.name if out else 'skipped'}")


if __name__ == "__main__":
    main()
