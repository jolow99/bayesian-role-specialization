"""2026-05-28 paper figures.

Aggregate metric table + individual-fitting figure across 14 models on
the 5-export, human-only, clean-team data scope from the 2026-05-25 full
pipeline (204 team-rounds, 15 envs).

Models:
  - 7 Bayesian models (from 05-25 pipeline; params taken from its
    `results.json` under the agg_ll fit objective).
  - 7 baselines (Random, Top-7, Random-to-Optimal, Copy Others,
    Contradict Others, Random Walk, Optimal) — the baselines from
    2026-03-30-summary-of-all-models/analysis.ipynb, ported here and
    re-run on the new scope. Random Walk's eps is re-fit on agg_ll.

Outputs:
  - results.json        — fitted params + (combo_r, marg_r, agg_ll, mean_ll) per model
  - aggregate_table.md  — markdown table sorted by combo_r
  - figures/individual_fitting.png — pie + stacked bar of dominant model per participant
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent / "2026-05-25-full-pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from pipeline import (
    load_human_team_records, precompute_trajectories,
    strategy_from_params, eval_subset,
)
from models import (
    belief_factory, value_factory, walk_factory, thresh_factory,
    walk_ps_factory, thresh_ps_factory, mixture_ps_factory,
)
from shared.constants import (
    ROLE_CHAR_TO_IDX, ALL_ROLE_COMBOS, TURNS_PER_STAGE,
)
from shared.data_loading import load_all_exports
from shared import EXPORTS_DIR


PIPELINE_RESULTS = PIPELINE_DIR / "results.json"
STAGE1_PARAMS = PIPELINE_DIR / "stage1_inference" / "best_inference_params.json"
RESULTS_PATH = SCRIPT_DIR / "results.json"
TABLE_PATH = SCRIPT_DIR / "aggregate_table.md"
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Bayesian fits use the agg_ll objective (proper scoring rule, per 05-25 README).
OBJ = "agg_ll"

# Display order: Bayesian first (best→worst expected), then baselines.
MODEL_ORDER = [
    "Bayesian Walk",
    "Bayesian Walk-PS",
    "Mixture-PS",
    "Bayesian-Belief",
    "Bayesian-Value",
    "Bayesian Threshold",
    "Bayesian Thresh-PS",
    "Random Walk",
    "Top-7",
    "Random-to-Optimal",
    "Optimal",
    "Copy Others",
    "Contradict Others",
    "Random",
]

MODEL_COLORS = {
    # Bayesian
    "Bayesian Walk":       "#8e44ad",
    "Bayesian Walk-PS":    "#af7ac5",
    "Mixture-PS":          "#6c3483",
    "Bayesian-Belief":     "#2ecc71",
    "Bayesian-Value":      "#3498db",
    "Bayesian Threshold":  "#e67e22",
    "Bayesian Thresh-PS":  "#f5b041",
    # Baselines
    "Random Walk":         "#f39c12",
    "Top-7":               "#1abc9c",
    "Random-to-Optimal":   "#34495e",
    "Optimal":             "#e74c3c",
    "Copy Others":         "#7f8c8d",
    "Contradict Others":   "#c0392b",
    "Random":              "#95a5a6",
}


# ──────────────────────────────────────────────────────────────────────
# 1. Records (with participant_ids attached)
# ──────────────────────────────────────────────────────────────────────

def load_records_with_pids(verbose: bool = True):
    """Load the 05-25 pipeline records and attach participant_ids.

    Sorted-by-player_id order matches what load_human_team_records uses
    internally, so participant_ids[k] is the participant at in-game
    position k.
    """
    records = load_human_team_records(verbose=verbose)

    export_dirs = sorted(EXPORTS_DIR.glob("bayesian-role-specialization-*"))
    all_prs = load_all_exports(data_dirs=export_dirs)
    pid_map: dict[tuple[str, int], list[str | None]] = defaultdict(
        lambda: [None, None, None])
    for pr in all_prs:
        if pr.round.round_type != "human":
            continue
        pid_map[(pr.game_id, pr.round.round_number)][pr.player_id] = pr.participant_id

    for rec in records:
        rec["participant_ids"] = pid_map[(rec["game_id"], rec["round_number"])]
    return records


# ──────────────────────────────────────────────────────────────────────
# 2. Baseline predict_fns
# ──────────────────────────────────────────────────────────────────────

def _marginals_from_joint(combo_dist: dict[str, float]) -> list[np.ndarray]:
    """Per-player marginals from a joint distribution over combos.

    For models that build the joint as the outer product of per-player
    marginals (every Bayesian model in this pipeline), this exactly
    recovers those marginals. For baselines that don't factorise (Copy
    Others, Contradict Others), it returns the true marginal.
    """
    margs = [np.zeros(3) for _ in range(3)]
    for combo, p in combo_dist.items():
        for i, c in enumerate(combo):
            margs[i][ROLE_CHAR_TO_IDX[c]] += p
    return margs


def _wrap_baseline(dist_fn):
    """Adapt a per-stage dist_fn into a record→[stage_pred] predict_fn."""
    def predict_fn(record):
        preds = []
        for s, human_combo in enumerate(record["stage_roles"]):
            prev = record["stage_roles"][s - 1] if s > 0 else None
            dist = dist_fn(record, s, prev, record["env_config"])
            margs = _marginals_from_joint(dist)
            preds.append({
                "predicted_dist": dist,
                "human_combo": human_combo,
                "model_marginal": np.mean(margs, axis=0),
            })
        return preds
    return predict_fn


def _value_at_stage(env_config, lds, stage_idx) -> dict[str, float]:
    values = env_config["values"]
    turn_idx = stage_idx * TURNS_PER_STAGE
    intent = lds[turn_idx] if turn_idx < len(lds) else 0
    thp = min(int(env_config["team_max_hp"]), values.shape[2] - 1)
    ehp = min(int(env_config["enemy_max_hp"]), values.shape[3] - 1)
    vals: dict[str, float] = {}
    for combo in ALL_ROLE_COMBOS:
        idx = (ROLE_CHAR_TO_IDX[combo[0]] * 9
               + ROLE_CHAR_TO_IDX[combo[1]] * 3
               + ROLE_CHAR_TO_IDX[combo[2]])
        vals[combo] = float(values[idx, intent, thp, ehp])
    return vals


def predict_random(record, stage_idx, prev_combo, env_config):
    return {c: 1.0 / 27 for c in ALL_ROLE_COMBOS}


def predict_optimal(record, stage_idx, prev_combo, env_config):
    vals = _value_at_stage(env_config, record["lds"], stage_idx)
    max_v = max(vals.values())
    winners = [c for c, v in vals.items() if abs(v - max_v) < 1e-8]
    dist = {c: 0.0 for c in ALL_ROLE_COMBOS}
    for c in winners:
        dist[c] = 1.0 / len(winners)
    return dist


def make_predict_top_k(k):
    def fn(record, stage_idx, prev_combo, env_config):
        vals = _value_at_stage(env_config, record["lds"], stage_idx)
        ordered = sorted(vals.items(), key=lambda kv: -kv[1])[:k]
        dist = {c: 0.0 for c in ALL_ROLE_COMBOS}
        for c, _ in ordered:
            dist[c] = 1.0 / k
        return dist
    return fn


def predict_random_to_optimal(record, stage_idx, prev_combo, env_config):
    n_stages = len(record["stage_roles"])
    alpha = stage_idx / max(n_stages - 1, 1)
    opt = predict_optimal(record, stage_idx, prev_combo, env_config)
    return {c: (1 - alpha) / 27 + alpha * opt[c] for c in ALL_ROLE_COMBOS}


def predict_copy_others(record, stage_idx, prev_combo, env_config):
    if prev_combo is None:
        return {c: 1.0 / 27 for c in ALL_ROLE_COMBOS}
    dist = {}
    for combo in ALL_ROLE_COMBOS:
        p = 1.0
        for i in range(3):
            others = [prev_combo[j] for j in range(3) if j != i]
            p *= 0.5 if combo[i] in others else 0.0
        dist[combo] = p
    total = sum(dist.values())
    if total <= 0:
        return {c: 1.0 / 27 for c in ALL_ROLE_COMBOS}
    return {c: p / total for c, p in dist.items()}


def predict_contradict(record, stage_idx, prev_combo, env_config):
    if prev_combo is None:
        return {c: 1.0 / 27 for c in ALL_ROLE_COMBOS}
    dist = {}
    for combo in ALL_ROLE_COMBOS:
        p = 1.0
        for i in range(3):
            others = {prev_combo[j] for j in range(3) if j != i}
            p *= 1.0 if combo[i] not in others else 0.1
        dist[combo] = p
    total = sum(dist.values())
    return {c: p / total for c, p in dist.items()}


def make_predict_random_walk(eps):
    """ε-stick: prob (1-eps) stay, prob eps switch uniformly to one of two others."""
    def fn(record, stage_idx, prev_combo, env_config):
        if prev_combo is None:
            return {c: 1.0 / 27 for c in ALL_ROLE_COMBOS}
        dist = {}
        for combo in ALL_ROLE_COMBOS:
            p = 1.0
            for i in range(3):
                p *= (1.0 - eps) if combo[i] == prev_combo[i] else (eps / 2.0)
            dist[combo] = p
        return dist
    return fn


def fit_random_walk_eps(records, grid=None) -> tuple[float, dict]:
    """Grid-search eps over agg_ll. Returns (best_eps, eval_at_best)."""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    best_eps = float(grid[0])
    best_metric = None
    best_ll = -np.inf
    for eps in grid:
        m = eval_subset(records, _wrap_baseline(make_predict_random_walk(float(eps))))
        if m["agg_ll"] > best_ll:
            best_ll = m["agg_ll"]
            best_eps = float(eps)
            best_metric = m
    return best_eps, best_metric


# ──────────────────────────────────────────────────────────────────────
# 3. Build all 14 predict_fns
# ──────────────────────────────────────────────────────────────────────

def build_predict_fns(records, trajectories, pipeline_cells):
    """Returns dict[name → predict_fn] for all 14 models + the fitted params used.

    Bayesian model params come from the 05-25 pipeline's agg_ll fits.
    Random Walk is fit here.
    """
    fns: dict[str, callable] = {}
    params: dict[str, dict] = {}

    # Bayesian models — use the agg_ll fits from 05-25 pipeline.
    walk_p = pipeline_cells["bayesian_walk"][OBJ]["fitted_params"]
    fns["Bayesian Walk"] = walk_factory(
        walk_p["tau_softmax"], walk_p["epsilon_switch"]
    )(records, trajectories)
    params["Bayesian Walk"] = walk_p

    walk_ps_p = pipeline_cells["bayesian_walk_ps"][OBJ]["fitted_params"]
    fns["Bayesian Walk-PS"] = walk_ps_factory(
        walk_ps_p["epsilon_switch"]
    )(records, trajectories)
    params["Bayesian Walk-PS"] = walk_ps_p

    mix_p = pipeline_cells["mixture_ps"][OBJ]["fitted_params"]
    fns["Mixture-PS"] = mixture_ps_factory(
        mix_p["walk_eps_frozen"], mix_p["thresh_eps_frozen"],
        mix_p["thresh_delta_frozen"], mix_p["w"],
    )(records, trajectories)
    params["Mixture-PS"] = mix_p

    fns["Bayesian-Belief"] = belief_factory()(records, trajectories)
    params["Bayesian-Belief"] = {}

    val_p = pipeline_cells["bayesian_value"][OBJ]["fitted_params"]
    fns["Bayesian-Value"] = value_factory(val_p["tau_softmax"])(records, trajectories)
    params["Bayesian-Value"] = val_p

    thr_p = pipeline_cells["bayesian_thresh"][OBJ]["fitted_params"]
    fns["Bayesian Threshold"] = thresh_factory(
        thr_p["tau_softmax"], thr_p["delta"]
    )(records, trajectories)
    params["Bayesian Threshold"] = thr_p

    thr_ps_p = pipeline_cells["bayesian_thresh_ps"][OBJ]["fitted_params"]
    fns["Bayesian Thresh-PS"] = thresh_ps_factory(
        thr_ps_p["epsilon_switch"], thr_ps_p["delta"]
    )(records, trajectories)
    params["Bayesian Thresh-PS"] = thr_ps_p

    # Baselines.
    rw_eps, _ = fit_random_walk_eps(records)
    print(f"  Random Walk fitted eps = {rw_eps:.3f} (agg_ll grid search)")
    fns["Random Walk"] = _wrap_baseline(make_predict_random_walk(rw_eps))
    params["Random Walk"] = {"eps": rw_eps}

    fns["Top-7"] = _wrap_baseline(make_predict_top_k(7))
    params["Top-7"] = {"k": 7}

    fns["Random-to-Optimal"] = _wrap_baseline(predict_random_to_optimal)
    params["Random-to-Optimal"] = {}

    fns["Optimal"] = _wrap_baseline(predict_optimal)
    params["Optimal"] = {}

    fns["Copy Others"] = _wrap_baseline(predict_copy_others)
    params["Copy Others"] = {}

    fns["Contradict Others"] = _wrap_baseline(predict_contradict)
    params["Contradict Others"] = {}

    fns["Random"] = _wrap_baseline(predict_random)
    params["Random"] = {}

    return fns, params


# ──────────────────────────────────────────────────────────────────────
# 4. Aggregate eval (combo_r, marg_r, agg_ll, mean_ll)
# ──────────────────────────────────────────────────────────────────────

def evaluate_all(records, predict_fns) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name, fn in predict_fns.items():
        m = eval_subset(records, fn)
        out[name] = {
            "combo_r": float(m["combo_r"]),
            "marg_r":  float(m["marg_r"]),
            "agg_ll":  float(m["agg_ll"]),
            "mean_ll": float(m["mean_ll"]),
            "n_records": int(m["n_records"]),
        }
    return out


def write_table(metrics_by_model, params_by_model, n_records, stage1, path: Path):
    ordered = [m for m in MODEL_ORDER if m in metrics_by_model]
    df = pd.DataFrame(
        [{"Model": m, **metrics_by_model[m]} for m in ordered]
    ).sort_values("combo_r", ascending=False)

    lines = [
        "# Aggregate model comparison — paper Section: "
        "How well does the model explain human team behavior?",
        "",
        f"Fitted on **{n_records}** clean human team-rounds from 5 exports "
        f"(2026-05-25 pipeline scope). Bayesian models use the pipeline's "
        f"`agg_ll`-objective fits; Random Walk's ε is grid-searched here on "
        f"the same objective; other baselines are parameter-free. Stage-1 "
        f"params: τ_prior = {stage1['tau_prior']:.4f}, ε = {stage1['epsilon']:.6f}, "
        f"memory_strategy = `{stage1['memory_strategy']}`.",
        "",
        "Models sorted by `combo_r` (descending).",
        "",
        "| Model | combo_r | marg_r | agg_ll | mean_ll |",
        "|-------|--------:|-------:|-------:|--------:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['Model']} "
            f"| {row['combo_r']:.4f} | {row['marg_r']:.4f} "
            f"| {row['agg_ll']:.4f} | {row['mean_ll']:.4f} |"
        )
    lines.append("")

    lines.append("## Fitted parameters")
    lines.append("")
    lines.append("| Model | Params |")
    lines.append("|-------|--------|")
    for m in ordered:
        p = params_by_model[m]
        ps = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in p.items() if not k.endswith("_frozen")
        )
        lines.append(f"| {m} | {ps or '(none)'} |")
    lines.append("")

    path.write_text("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────
# 5. Individual fitting — P(model | participant)
# ──────────────────────────────────────────────────────────────────────

LAPSE = 0.05
UNIFORM = np.ones(3) / 3.0


def precompute_player_marginals(records, predict_fns):
    """For each (model, record, stage), cache the per-player marginals.

    Returns out[name][record_index] = list-per-stage of list-per-player of (3,) arrays.
    """
    out: dict[str, list] = {}
    for name, fn in predict_fns.items():
        per_record = []
        for rec in records:
            preds = fn(rec)
            stage_player = []
            for stage_pred in preds:
                margs = _marginals_from_joint(stage_pred["predicted_dist"])
                stage_player.append(margs)
            per_record.append(stage_player)
        out[name] = per_record
    return out


def individual_posteriors(records, player_margs_by_model):
    """For each participant, compute log-likelihoods per model and the softmax posterior."""
    model_names = list(player_margs_by_model.keys())
    player_ll = {name: defaultdict(float) for name in model_names}
    player_n_stages: dict[str, int] = defaultdict(int)
    all_pids: set[str] = set()

    for ri, rec in enumerate(records):
        for s, human_combo in enumerate(rec["stage_roles"]):
            for pos in range(3):
                pid = rec["participant_ids"][pos]
                if pid is None:
                    continue
                all_pids.add(pid)
                actual_idx = ROLE_CHAR_TO_IDX[human_combo[pos]]
                player_n_stages[pid] += 1
                for name in model_names:
                    stages = player_margs_by_model[name][ri]
                    marg = (stages[s][pos] if s < len(stages)
                            else np.ones(3) / 3.0)
                    marg = (1.0 - LAPSE) * marg + LAPSE * UNIFORM
                    player_ll[name][pid] += float(
                        np.log(max(marg[actual_idx], 1e-20)))

    posteriors: dict[str, dict[str, float]] = {}
    for pid in all_pids:
        lls = np.array([player_ll[name][pid] for name in model_names])
        log_post = lls - lls.max()
        post = np.exp(log_post)
        post /= post.sum()
        posteriors[pid] = dict(zip(model_names, post))

    return posteriors, player_ll, dict(player_n_stages)


def plot_individual_fitting(posteriors, model_names, save_path: Path):
    n = len(posteriors)
    # Wider figure with explicit width ratios; legend placed outside the bar axes.
    fig = plt.figure(figsize=(22, 7.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 2.6, 0.5], wspace=0.25)
    ax_pie = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_leg = fig.add_subplot(gs[0, 2])
    ax_leg.axis("off")

    # ── Pie ──
    dominant = {pid: max(model_names, key=lambda m: post[m])
                for pid, post in posteriors.items()}
    counts = pd.Series(dominant).value_counts()
    counts = counts.reindex([m for m in MODEL_ORDER if m in counts.index]).dropna()
    if len(counts) == 0:
        counts = pd.Series([1], index=["Random"])

    pie_colors = [MODEL_COLORS.get(m, "#95a5a6") for m in counts.index]
    wedges, texts, autotexts = ax_pie.pie(
        counts.values, labels=counts.index, colors=pie_colors,
        autopct=lambda pct: f"{pct:.0f}%\n({int(round(pct/100*n))})",
        startangle=90, textprops={"fontsize": 9},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax_pie.set_title(f"Best-fitting model per participant (n={n})",
                     fontsize=12, fontweight="bold")

    # ── Stacked bar ──
    rows = []
    for pid, post in posteriors.items():
        row = {"pid": pid, "dominant": dominant[pid]}
        for m in model_names:
            row[m] = post[m]
        rows.append(row)
    df = pd.DataFrame(rows)

    order_idx = {m: i for i, m in enumerate(MODEL_ORDER)}
    df["sort_dominant"] = df["dominant"].map(order_idx).fillna(99)
    df["sort_weight"] = df.apply(
        lambda r: r[r["dominant"]] if r["dominant"] in r else 0.0, axis=1)
    df = df.sort_values(["sort_dominant", "sort_weight"],
                        ascending=[True, False]).reset_index(drop=True)

    x = np.arange(len(df))
    bottoms = np.zeros(len(df))
    handles = []
    labels = []
    for m in MODEL_ORDER:
        if m not in model_names:
            continue
        vals = df[m].values
        bar = ax_bar.bar(x, vals, bottom=bottoms,
                          color=MODEL_COLORS.get(m, "#95a5a6"),
                          label=m, width=0.95,
                          edgecolor="white", linewidth=0.25)
        bottoms += vals
        handles.append(bar)
        labels.append(m)
    ax_bar.set_xlim(-0.5, len(df) - 0.5)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_xlabel("Participants (sorted by dominant model)")
    ax_bar.set_ylabel("P(model | player data)")
    ax_bar.set_title("Posterior over models per participant",
                     fontsize=12, fontweight="bold")
    ax_bar.set_xticks([])

    # Legend in dedicated axes (no overlap with bar data).
    ax_leg.legend(handles, labels, loc="center left", fontsize=9,
                  frameon=False, title="Model",
                  title_fontsize=10)

    fig.suptitle(
        f"Individual-level model fitting "
        f"({len(model_names)} models, lapse={LAPSE})",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 66)
    print("2026-05-28 paper figures")
    print("=" * 66)

    with open(STAGE1_PARAMS) as f:
        s1 = json.load(f)
    strat = strategy_from_params(
        s1.get("memory_strategy", "full"),
        s1.get("window"),
        s1.get("drift_delta", 0.0),
    )
    print(f"Stage 1: tau_prior={s1['tau_prior']:.4f} "
          f"epsilon={s1['epsilon']:.6f} strategy={strat.name}")

    with open(PIPELINE_RESULTS) as f:
        pipeline_out = json.load(f)
    pipeline_cells = pipeline_out["cells"]

    print("\nLoading records (with participant_ids)...")
    records = load_records_with_pids()

    print("\nPrecomputing trajectories...")
    trajectories = precompute_trajectories(
        records, s1["tau_prior"], s1["epsilon"], memory_strategy=strat)

    print("\nBuilding predict_fns for 14 models...")
    predict_fns, params_by_model = build_predict_fns(
        records, trajectories, pipeline_cells)

    print("\nEvaluating aggregate metrics...")
    metrics_by_model = evaluate_all(records, predict_fns)
    for m in MODEL_ORDER:
        if m in metrics_by_model:
            x = metrics_by_model[m]
            print(f"  {m:22s}: combo_r={x['combo_r']:.4f}  "
                  f"marg_r={x['marg_r']:.4f}  "
                  f"agg_ll={x['agg_ll']:.4f}  "
                  f"mean_ll={x['mean_ll']:.4f}")

    print(f"\nWriting {TABLE_PATH.name}...")
    write_table(metrics_by_model, params_by_model, len(records),
                {"tau_prior": s1["tau_prior"], "epsilon": s1["epsilon"],
                 "memory_strategy": strat.name},
                TABLE_PATH)

    print("\nPrecomputing per-player marginals for individual fitting...")
    player_margs = precompute_player_marginals(records, predict_fns)

    print("Computing individual posteriors...")
    posteriors, player_ll, player_n_stages = individual_posteriors(
        records, player_margs)
    n_participants = len(posteriors)
    print(f"  {n_participants} participants with at least one stage")

    dominant_counts: dict[str, int] = defaultdict(int)
    for pid, post in posteriors.items():
        dom = max(post, key=post.get)
        dominant_counts[dom] += 1
    print("\n  Dominant-model counts:")
    for m in MODEL_ORDER:
        if m in dominant_counts:
            c = dominant_counts[m]
            print(f"    {m:22s}: {c:3d}  ({c/n_participants*100:.0f}%)")

    fig_path = FIGURES_DIR / "individual_fitting.png"
    print(f"\nWriting {fig_path}...")
    plot_individual_fitting(posteriors, list(predict_fns.keys()), fig_path)

    out = {
        "scope": {
            "exports": pipeline_out["exports"],
            "n_records": len(records),
            "n_participants": n_participants,
            "stage1": {
                "tau_prior": s1["tau_prior"],
                "epsilon": s1["epsilon"],
                "memory_strategy": strat.name,
            },
            "bayesian_fit_objective": OBJ,
            "lapse_rate": LAPSE,
        },
        "aggregate": {
            "metrics_by_model": metrics_by_model,
            "params_by_model": {
                k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating, np.integer))
                          else vv)
                    for kk, vv in v.items()}
                for k, v in params_by_model.items()
            },
        },
        "individual": {
            "posteriors": posteriors,
            "log_likelihoods": {name: dict(d) for name, d in player_ll.items()},
            "n_stages_per_participant": player_n_stages,
            "dominant_counts": dict(dominant_counts),
        },
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2, default=_np_default)
    print(f"Saved results to {RESULTS_PATH}")


def _np_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == "__main__":
    main()
