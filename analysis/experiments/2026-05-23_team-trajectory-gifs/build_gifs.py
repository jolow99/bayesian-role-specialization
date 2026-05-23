"""Per-environment trajectory GIFs with model predictions overlaid.

For each environment that has ≥ N teams, we render one animated GIF showing
how teams' role choices unfold stage-by-stage alongside three model
predictions (Bayesian-Belief, Bayesian-Value, Bayesian-Walk).

The trajectory engine + tuned params are imported from the
2026-05-12 current-export metric-comparison experiment so we get exactly
the same trajectories / params used in published metric numbers.

Layout per frame (Tufte: layered, multivariate, small-multiples):

    ┌─────────────────────────────────────────────────────────────┐
    │ Env <env_id>   optimal <combo>   stats <stat_profile>       │
    │ Stage s of S                          teams n=K             │
    ├─────────────────────────────────────────────────────────────┤
    │  HP timeline (range-framed sparkline): team / enemy across  │
    │  every turn; vertical band = current stage; intent ticks    │
    ├─────────────────────────────────────────────────────────────┤
    │  Per-team trajectory grid (rows=teams, cols=stages)         │
    │  Each cell = 3 role-coloured tiles (P1/P2/P3); column for   │
    │  the current stage is emphasised; optimal combo as caption  │
    ├─────────────────────────────────────────────────────────────┤
    │  Stage-s distribution (canonical combos):                   │
    │   horizontal bar = empirical frequency among teams;         │
    │   coloured dots = each model's predicted probability        │
    │   reference tick = optimal combo                            │
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

# Reuse the existing trajectory engine + tuned params.
HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
METRIC_DIR = EXP_ROOT / "2026-05-12-current-export-metric-comparison"
sys.path.insert(0, str(METRIC_DIR))

from pipeline import (  # noqa: E402
    load_human_team_records,
    precompute_trajectories,
    strategy_from_params,
)
from models import (  # noqa: E402
    belief_factory,
    value_factory,
    walk_factory,
)
from shared.constants import (  # noqa: E402
    ROLE_COLORS, ROLE_CHAR_TO_IDX, ROLE_SHORT, TURNS_PER_STAGE,
)
from shared.parsing import canonical_combo, get_canonical_combos  # noqa: E402
from shared.inference import preferred_action, game_step  # noqa: E402

FIGURES_DIR = HERE / "figures"
FRAMES_DIR = HERE / "frames"


# Tuned params from results.json (Stage 1 + per-model best-on-combo_r).
TAU_PRIOR = 4.638476144217848
EPSILON = 0.06241645791201582
MEMORY_STRATEGY = "drift_prior_0.500"

MODEL_SPECS = [
    {
        "name": "Bayesian-Belief",
        "short": "B",
        "color": "#1f3a93",
        "marker": "o",
        "factory_builder": lambda: belief_factory(),
    },
    {
        "name": "Bayesian-Value",
        "short": "V",
        "color": "#c0392b",
        "marker": "D",
        "factory_builder": lambda: value_factory(tau_softmax=13.71598290227467),
    },
    {
        "name": "Bayesian-Walk",
        "short": "W",
        "color": "#16a085",
        "marker": "^",
        "factory_builder": lambda: walk_factory(
            tau_softmax=7.20651148477258,
            epsilon_switch=0.5589855617201609,
        ),
    },
]


# ──────────────────────────────────────────────────────────────────────
# Game-state replay (for HP sparkline)
# ──────────────────────────────────────────────────────────────────────

def replay_hp_timeline(record):
    """Replay one team's HP for every turn. Returns:
        team_hp[T+1], enemy_hp[T+1], intents[T] (one per turn)
    where T = num turns actually played before any death."""
    env = record["env_config"]
    player_stats = env["player_stats"]
    boss_damage = env["boss_damage"]
    team_max_hp = env["team_max_hp"]
    enemy_max_hp = env["enemy_max_hp"]
    lds = record["lds"]
    stage_roles_list = record["stage_roles"]

    team_hp = [float(team_max_hp)]
    enemy_hp = [float(enemy_max_hp)]
    intents = []
    turn_idx = 0
    thp = float(team_max_hp)
    ehp = float(enemy_max_hp)

    for combo in stage_roles_list:
        roles = [ROLE_CHAR_TO_IDX[c] for c in combo]
        for _ in range(TURNS_PER_STAGE):
            if turn_idx >= len(lds) or thp <= 0 or ehp <= 0:
                break
            intent = int(lds[turn_idx])
            actions = [
                preferred_action(roles[i], intent, thp, team_max_hp)
                for i in range(3)
            ]
            thp, ehp = game_step(
                intent, thp, ehp, actions,
                player_stats, boss_damage, team_max_hp,
            )
            intents.append(intent)
            team_hp.append(thp)
            enemy_hp.append(ehp)
            turn_idx += 1
    return team_hp, enemy_hp, intents


# ──────────────────────────────────────────────────────────────────────
# Per-env aggregation
# ──────────────────────────────────────────────────────────────────────

def group_by_env(records):
    by_env = defaultdict(list)
    for r in records:
        by_env[r["env_id"]].append(r)
    return by_env


def per_env_model_predictions(records):
    """Return predictions[model_short][record_index] = list of per-stage dicts."""
    strategy = strategy_from_params(MEMORY_STRATEGY, window=None, drift_delta=None)
    trajectories = precompute_trajectories(records, TAU_PRIOR, EPSILON, strategy)

    out = {}
    for spec in MODEL_SPECS:
        factory = spec["factory_builder"]()
        predict_fn = factory(records, trajectories)
        out[spec["short"]] = [predict_fn(r) for r in records]
    return out


def empirical_distribution(records, canon_combos, stat_profile, stage):
    """Histogram of teams' canonical role combo at this stage."""
    counts = {cc: 0 for cc in canon_combos}
    n = 0
    for r in records:
        if stage < len(r["stage_roles"]):
            cc = canonical_combo(r["stage_roles"][stage], stat_profile)
            counts[cc] += 1
            n += 1
    if n == 0:
        return counts, 0
    return counts, n


def average_model_dist(predictions, canon_combos, stat_profile, stage, records):
    """Average each model's predicted_dist over teams that have this stage."""
    out = {}
    for short, per_record_preds in predictions.items():
        agg = {cc: 0.0 for cc in canon_combos}
        n = 0
        for rec, preds in zip(records, per_record_preds):
            if stage < len(preds):
                pred = preds[stage]
                for combo, prob in pred["predicted_dist"].items():
                    agg[canonical_combo(combo, stat_profile)] += prob
                n += 1
        if n > 0:
            agg = {cc: v / n for cc, v in agg.items()}
        out[short] = (agg, n)
    return out


# ──────────────────────────────────────────────────────────────────────
# Drawing primitives
# ──────────────────────────────────────────────────────────────────────

def draw_combo_glyph(ax, x, y, w, h, combo, faded=False):
    """Three coloured tiles for a 3-letter role combo (F red, T blue, M green)."""
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


def render_frame(env_id, records, predictions, stage, out_path):
    stat_profile = records[0]["stat_profile"]
    optimal = records[0]["optimal_roles"]
    canon_combos = get_canonical_combos(stat_profile)
    canon_optimal = canonical_combo(optimal, stat_profile)
    n_stages_max = max(len(r["stage_roles"]) for r in records)
    n_teams = len(records)

    # ── Build the figure ──────────────────────────────────────────────
    fig = plt.figure(figsize=(11.5, 8.5), dpi=110)
    gs = GridSpec(
        nrows=3, ncols=1,
        height_ratios=[1.0, 2.0, 3.0],
        hspace=0.55,
        left=0.07, right=0.97, top=0.91, bottom=0.05,
    )
    ax_hp = fig.add_subplot(gs[0])
    ax_grid = fig.add_subplot(gs[1])
    ax_dist = fig.add_subplot(gs[2])

    # Title strip ─────────────────────────────────────────────────────
    fig.text(0.07, 0.965, f"env {env_id}", fontsize=13, fontweight="bold",
             family="monospace")
    fig.text(0.07, 0.940,
             f"stats {stat_profile}   •   optimal {optimal}   •   "
             f"n={n_teams} teams",
             fontsize=10, color="#444", family="monospace")
    fig.text(0.97, 0.965, f"stage {stage + 1} of {n_stages_max}",
             fontsize=13, fontweight="bold", ha="right", family="monospace")

    # ── HP sparkline (averaged across teams) ─────────────────────────
    timelines = [replay_hp_timeline(r) for r in records]
    max_T = max(len(t[0]) for t in timelines)  # in HP-points (turns+1)
    avg_team = np.full(max_T, np.nan)
    avg_enemy = np.full(max_T, np.nan)
    for ti in range(max_T):
        ts = [t[0][ti] for t in timelines if ti < len(t[0])]
        es = [t[1][ti] for t in timelines if ti < len(t[1])]
        if ts:
            avg_team[ti] = np.mean(ts)
        if es:
            avg_enemy[ti] = np.mean(es)
    team_max_hp = records[0]["env_config"]["team_max_hp"]
    enemy_max_hp = records[0]["env_config"]["enemy_max_hp"]
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

    # current-stage shading
    stage_start = stage * TURNS_PER_STAGE
    stage_end = min(stage_start + TURNS_PER_STAGE, max_T - 1)
    ax_hp.axvspan(stage_start, stage_end, color="#f6c200", alpha=0.18, lw=0)
    # subtle stage gridlines
    for s_idx in range(n_stages_max + 1):
        ax_hp.axvline(s_idx * TURNS_PER_STAGE, color="#bbb",
                      lw=0.4, alpha=0.6)
    # intent ticks (red ▼ above the chart marks enemy-attack turns)
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
    for i, pos in enumerate(stage_tick_pos):
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

    # ── Per-team trajectory grid ─────────────────────────────────────
    ax_grid.set_xlim(0, n_stages_max)
    ax_grid.set_ylim(n_teams, 0)
    ax_grid.set_aspect("auto")
    ax_grid.set_yticks([])
    ax_grid.set_xticks([])
    for s in ("top", "right", "left", "bottom"):
        ax_grid.spines[s].set_visible(False)

    # subtle stage-column header
    for s_idx in range(n_stages_max):
        x0 = s_idx + 0.04
        ax_grid.text(s_idx + 0.5, -0.35, f"stage {s_idx + 1}",
                     ha="center", va="bottom", fontsize=8.5, color="#444")

    # current-stage highlight band
    ax_grid.add_patch(mpatches.Rectangle(
        (stage, 0), 1, n_teams, facecolor="#f6c200", alpha=0.13,
        edgecolor="none", lw=0))

    # sort teams by their stage-0 combo for visual coherence
    sorted_recs = sorted(
        records,
        key=lambda r: tuple(r["stage_roles"][s] if s < len(r["stage_roles"]) else "~"
                            for s in range(n_stages_max)),
    )

    cell_pad = 0.07
    for ti, rec in enumerate(sorted_recs):
        for si in range(n_stages_max):
            if si >= len(rec["stage_roles"]):
                continue
            faded = si > stage   # future stages: faded
            draw_combo_glyph(
                ax_grid,
                x=si + cell_pad,
                y=ti + cell_pad,
                w=1 - 2 * cell_pad,
                h=1 - 2 * cell_pad,
                combo=rec["stage_roles"][si],
                faded=faded,
            )
        # label team index lightly
        ax_grid.text(-0.08, ti + 0.5, f"{ti+1:>2}",
                     ha="right", va="center", fontsize=7.0, color="#888",
                     family="monospace")

    # column separator gridlines
    for s_idx in range(1, n_stages_max):
        ax_grid.axvline(s_idx, color="#fff", lw=1.2)
    ax_grid.set_title(
        f"per-team role choices (rows sorted by full trajectory; faded = future stages)",
        fontsize=9, color="#444", loc="left", pad=18,
    )

    # role-colour legend (single line under the grid)
    legend_y = n_teams + 0.35
    legend_x = 0.05
    for label, key in [("F fighter", "F"), ("T tank", "T"), ("M medic", "M")]:
        ax_grid.add_patch(mpatches.Rectangle(
            (legend_x, legend_y), 0.12, 0.32,
            facecolor=ROLE_COLORS[key], edgecolor="none",
            clip_on=False,
        ))
        ax_grid.text(legend_x + 0.16, legend_y + 0.16, label,
                     fontsize=8, va="center", color="#333",
                     clip_on=False)
        legend_x += 0.95

    # ── Stage-s distribution + model overlay ────────────────────────
    emp_counts, n_played = empirical_distribution(
        records, canon_combos, stat_profile, stage)
    model_dists = average_model_dist(
        predictions, canon_combos, stat_profile, stage, records)

    # rank by empirical frequency at this stage, breaking ties by combo string
    combos_sorted = sorted(
        canon_combos,
        key=lambda cc: (-emp_counts[cc], cc),
    )
    # keep only combos that have any empirical or model mass (>1%)
    visible = [
        cc for cc in combos_sorted
        if emp_counts[cc] > 0
        or any(model_dists[s][0].get(cc, 0) > 0.01 for s in model_dists)
    ]
    if not visible:
        visible = combos_sorted[:8]

    y_pos = np.arange(len(visible))
    if n_played > 0:
        emp_freq = np.array([emp_counts[cc] / n_played for cc in visible])
    else:
        emp_freq = np.zeros(len(visible))

    # empirical bars
    ax_dist.barh(y_pos, emp_freq, height=0.62, color="#dadce0",
                 edgecolor="#aaa", lw=0.4, zorder=2)
    # raw count annotation
    for yi, cc in zip(y_pos, visible):
        c = emp_counts[cc]
        if c > 0:
            ax_dist.text(emp_freq[yi] + 0.012, yi, f"{c}/{n_played}",
                         va="center", fontsize=7.5, color="#444")

    # model dots
    for spec in MODEL_SPECS:
        short = spec["short"]
        agg, _ = model_dists[short]
        xs = [agg.get(cc, 0.0) for cc in visible]
        ax_dist.scatter(
            xs, y_pos,
            marker=spec["marker"], s=42,
            facecolor=spec["color"], edgecolor="white", lw=0.8,
            label=spec["name"], zorder=4,
        )

    # optimal-combo reference
    if canon_optimal in visible:
        idx = visible.index(canon_optimal)
        ax_dist.axhline(idx, color="#f6c200", lw=2.0, alpha=0.35, zorder=1)
        ax_dist.text(
            -0.02, idx, "★", fontsize=12, ha="right", va="center",
            color="#d4a000", clip_on=False,
        )

    ax_dist.set_yticks(y_pos)
    ax_dist.set_yticklabels(visible, family="monospace", fontsize=9)
    ax_dist.invert_yaxis()
    ax_dist.set_xlim(0, max(0.45, float(emp_freq.max() + 0.10)
                             if len(emp_freq) else 0.45))
    ax_dist.set_xlabel("probability / empirical frequency", fontsize=9, color="#444")
    for s in ("top", "right"):
        ax_dist.spines[s].set_visible(False)
    ax_dist.spines["left"].set_color("#aaa")
    ax_dist.spines["bottom"].set_color("#aaa")
    ax_dist.tick_params(colors="#444")
    ax_dist.set_title(
        f"stage {stage + 1} role-combo distribution  "
        f"(bars = teams, dots = model predictions)",
        fontsize=10, color="#222", loc="left", pad=6,
    )
    ax_dist.legend(
        loc="lower right", fontsize=8.5, frameon=False, scatterpoints=1,
        handletextpad=0.3, borderpad=0.2,
    )

    fig.savefig(out_path, dpi=110, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Stitch frames into GIF
# ──────────────────────────────────────────────────────────────────────

def make_gif(env_id, records, predictions):
    n_stages = max(len(r["stage_roles"]) for r in records)
    env_frame_dir = FRAMES_DIR / env_id
    env_frame_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = []
    for stage in range(n_stages):
        fp = env_frame_dir / f"stage_{stage:02d}.png"
        render_frame(env_id, records, predictions, stage, fp)
        frame_paths.append(fp)

    frames = [Image.open(fp).convert("RGB") for fp in frame_paths]
    if not frames:
        return None
    # pad all to identical size (take max h/w)
    max_w = max(f.width for f in frames)
    max_h = max(f.height for f in frames)
    padded = []
    for f in frames:
        canvas = Image.new("RGB", (max_w, max_h), "white")
        canvas.paste(f, ((max_w - f.width) // 2, (max_h - f.height) // 2))
        padded.append(canvas)

    out_path = FIGURES_DIR / f"{env_id}.gif"
    padded[0].save(
        out_path,
        save_all=True,
        append_images=padded[1:] + [padded[-1]],   # hold final stage longer
        duration=[1600] * (len(padded) - 1) + [2200, 2200],
        loop=0,
        optimize=True,
    )
    return out_path


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────

def main(min_teams: int = 12):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    all_records = load_human_team_records(verbose=True)
    by_env = group_by_env(all_records)

    print(f"\n[build_gifs] {len(by_env)} envs total. "
          f"keeping envs with ≥ {min_teams} teams.")

    for env_id, recs in sorted(by_env.items()):
        if len(recs) < min_teams:
            print(f"  skip {env_id}: only {len(recs)} teams")
            continue
        print(f"  → {env_id}  ({len(recs)} teams)")
        preds = per_env_model_predictions(recs)
        gif_path = make_gif(env_id, recs, preds)
        if gif_path:
            print(f"     saved {gif_path.relative_to(EXP_ROOT.parent)}")


if __name__ == "__main__":
    main()
