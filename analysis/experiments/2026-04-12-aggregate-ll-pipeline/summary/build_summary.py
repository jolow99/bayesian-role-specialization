"""Build the comparison summary across all 7 Stage-2 models.

Reads each model's ``params.json`` and ``cv/cv_results.json`` (if present)
and emits:
    - summary/results.json            structured results table
    - summary/summary_table.md        human-readable markdown table
    - summary/agg_ll_vs_combo_r.png   scatter plot across model checkpoints
      (only for models whose checkpoint directories exist)
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from stage2_common import (
    load_stage1_params, load_records, precompute_trajectories,
    compute_disaggregated_metrics,
)
from memory_strategies import strategy_from_params

MODELS = [
    "bayesian_belief",
    "bayesian_value",
    "bayesian_walk",
    "bayesian_thresh",
    "bayesian_walk_ps",
    "bayesian_thresh_ps",
    "mixture_ps",
]

UNIFORM_27_BASELINE = float(-np.log(27))  # approx -3.296


# ──────────────────────────────────────────────────────────────────────
# Bootstrap 95% CI on agg_ll over team-rounds
# ──────────────────────────────────────────────────────────────────────

def bootstrap_agg_ll_ci(records, trajectories, predict_fn_factory,
                         n_boot: int = 200, seed: int = 0):
    """Resample team-round records with replacement and recompute agg_ll."""
    rng = np.random.default_rng(seed)
    n = len(records)
    if n == 0:
        return {"lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    from stage2_common import compute_pooled_metric
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sub_recs = [records[i] for i in idx]
        sub_trajs = [trajectories[i] for i in idx]
        val = compute_pooled_metric(sub_recs, sub_trajs, predict_fn_factory,
                                     metric="agg_ll")
        if not np.isnan(val):
            samples.append(val)
    if not samples:
        return {"lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi), "n_boot": len(samples),
            "mean": float(np.mean(samples))}


def _factory_for(model_name, params, frozen=None):
    """Recreate a prediction factory given fitted params (no CV)."""
    import importlib
    mod = importlib.import_module(f"{model_name}.tune")
    if model_name == "bayesian_belief":
        return mod.make_predict_fn
    if model_name == "bayesian_value":
        return mod.make_factory(params["tau_softmax"])
    if model_name == "bayesian_walk":
        return mod.make_factory(params["tau_softmax"], params["epsilon_switch"])
    if model_name == "bayesian_thresh":
        return mod.make_factory(params["tau_softmax"], params["delta"])
    if model_name == "bayesian_walk_ps":
        return mod.make_factory(params["epsilon_switch"])
    if model_name == "bayesian_thresh_ps":
        return mod.make_factory(params["epsilon_switch"], params["delta"])
    if model_name == "mixture_ps":
        return mod.make_factory(params["walk_eps_frozen"],
                                 params["thresh_eps_frozen"],
                                 params["thresh_delta_frozen"],
                                 params["w"])
    raise ValueError(f"unknown model {model_name!r}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 66)
    print("Summary builder — 2026-04-12-aggregate-ll-pipeline")
    print("=" * 66)

    s1 = load_stage1_params(EXPERIMENT_DIR)
    strat = strategy_from_params(
        s1.get("memory_strategy", "full"), s1.get("window"),
        s1.get("drift_delta", 0.0))

    records = load_records(include_bot_rounds=True)
    print("Precomputing trajectories (for bootstrap CIs)...")
    trajectories = precompute_trajectories(
        records, s1["tau_prior"], s1["epsilon"], memory_strategy=strat)

    cv_path = EXPERIMENT_DIR / "cv" / "cv_results.json"
    cv_data = json.load(open(cv_path)) if cv_path.exists() else {}

    summary = {"models": {}, "stage1_params": s1}
    table_rows = []
    scatter_points = []  # (agg_ll, combo_r, model) from checkpoints

    for model in MODELS:
        params_path = EXPERIMENT_DIR / model / "params.json"
        if not params_path.exists():
            print(f"  [skip] {model}: params.json missing")
            continue
        data = json.load(open(params_path))
        ev = data.get("eval", {})
        tuned = data.get("tuned_params", {})

        # Bootstrap CI on pooled agg_ll at the fitted point.
        factory = _factory_for(model, tuned)
        ci = bootstrap_agg_ll_ci(records, trajectories, factory, n_boot=200)

        human = ev.get("human", {})
        bot = ev.get("bot", {})
        pooled = ev.get("pooled", {})

        cv_entry = cv_data.get(model, {})
        cv_pooled = cv_entry.get("heldout_pooled_agg_ll", {})
        cv_spread = cv_entry.get("human_bot_spread", {})

        spread = (pooled.get("agg_ll", float("nan"))
                  - human.get("agg_ll", float("nan"))) if False else (
            human.get("agg_ll", float("nan")) - bot.get("agg_ll", float("nan"))
        )

        n_floor = pooled.get("n_floor_hits", 0)
        mean_ll_pooled = pooled.get("mean_ll", float("nan"))
        disagreement_flag = (not np.isnan(mean_ll_pooled)
                             and mean_ll_pooled < UNIFORM_27_BASELINE
                             and pooled.get("agg_ll", float("-inf")) > UNIFORM_27_BASELINE)

        row = {
            "model": model,
            "tuned_params": tuned,
            "eval": ev,
            "bootstrap_agg_ll_ci": ci,
            "cv_heldout_pooled_agg_ll": cv_pooled,
            "cv_human_bot_spread": cv_spread,
            "human_bot_spread": spread,
            "disagreement_flag": bool(disagreement_flag),
            "floor_hit_flag": bool(n_floor > 0),
        }
        summary["models"][model] = row
        table_rows.append(row)

        # Collect checkpoint points for scatter.
        for ck in ("coarse_results", "refined_results"):
            p = EXPERIMENT_DIR / model / "checkpoints" / f"{ck}.json"
            if p.exists():
                try:
                    arr = json.load(open(p))
                    for r in arr:
                        a = r.get("agg_ll", float("nan"))
                        c = r.get("combo_r", float("nan"))
                        if not (np.isnan(a) or np.isnan(c)):
                            scatter_points.append((a, c, model))
                except Exception:
                    pass

    # ── Table ─────────────────────────────────────────────────────────
    header = (
        "| Model | Params | Pooled agg_ll | Human agg_ll | Bot agg_ll | "
        "H-B spread | Pooled mean_ll | Pooled combo_r | CV pooled agg_ll | "
        "Bootstrap 95% CI | Flags |\n"
        "|-------|--------|---------------|--------------|------------|"
        "------------|----------------|-----------------|------------------|"
        "------------------|-------|"
    )
    lines = [header]
    for row in table_rows:
        ev = row["eval"]
        p = ev.get("pooled", {})
        h = ev.get("human", {})
        b = ev.get("bot", {})
        cv = row["cv_heldout_pooled_agg_ll"]
        ci = row["bootstrap_agg_ll_ci"]
        flags = []
        if row["disagreement_flag"]:
            flags.append("disagree")
        if row["floor_hit_flag"]:
            flags.append("floor")
        fmt = lambda x: ("n/a" if x is None or (isinstance(x, float) and np.isnan(x))
                         else f"{x:.4f}")
        lines.append(
            f"| {row['model']} "
            f"| {row['tuned_params']} "
            f"| {fmt(p.get('agg_ll'))} "
            f"| {fmt(h.get('agg_ll'))} "
            f"| {fmt(b.get('agg_ll'))} "
            f"| {fmt(row['human_bot_spread'])} "
            f"| {fmt(p.get('mean_ll'))} "
            f"| {fmt(p.get('combo_r'))} "
            f"| {fmt(cv.get('mean'))} ± {fmt(cv.get('std'))} "
            f"| [{fmt(ci.get('lo'))}, {fmt(ci.get('hi'))}] "
            f"| {','.join(flags) or '-'} |"
        )

    table_md = "\n".join(lines)
    (SCRIPT_DIR / "summary_table.md").write_text(table_md + "\n")
    print("\n" + table_md)

    with open(SCRIPT_DIR / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Scatter plot ─────────────────────────────────────────────────
    if scatter_points:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            by_model = {}
            for a, c, m in scatter_points:
                by_model.setdefault(m, []).append((a, c))
            for m, pts in by_model.items():
                arr = np.array(pts)
                ax.scatter(arr[:, 0], arr[:, 1], label=m, alpha=0.5, s=15)
            ax.set_xlabel("agg_ll")
            ax.set_ylabel("combo_r")
            ax.set_title("Grid evaluations: agg_ll vs combo_r by model")
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(SCRIPT_DIR / "agg_ll_vs_combo_r.png", dpi=150)
            plt.close(fig)
            print(f"\nSaved scatter to {SCRIPT_DIR / 'agg_ll_vs_combo_r.png'}")
        except Exception as e:
            print(f"Scatter skipped: {e}")

    print(f"\nSaved results to {SCRIPT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
