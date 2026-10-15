"""2026-10-15 — pure Threshold-PS refit + regenerate dependent outputs.

Context (2026-10-12 call with Xuan): Bayesian-Threshold-PS was
(1-eps_s)*stick + eps_s*[threshold-gated switch], i.e. a stick/threshold
MIXTURE, not a pure threshold rule. Fit under agg_ll it landed on
delta = 0 with eps_s = 0.39. Fix: drop eps_s, so

    pi(r) = stick                                  if no admissible alternative
          = marginal renormalised over admissible  otherwise

with admissible = {r != prev : P(r) - P(prev) > delta}. One free param (delta).

Everything else is IDENTICAL to the 05-25 / 05-28 pipelines (same records,
Stage-1 params, objective, other models' fitted params, 5% lapse for the
individual fit). Mixture-PS is dropped (already dropped from the paper), so
the individual figure has 13 models (6 Bayesian + 7 baselines).

Run from analysis/:
    uv run python experiments/2026-10-15-pure-threshold-ps/pure_threshold_ps.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXP = SCRIPT_DIR.parent
PF_DIR = EXP / "2026-05-28-paper-figures"
PIPE_DIR = EXP / "2026-05-25-full-pipeline"
sys.path.insert(0, str(PF_DIR))
sys.path.insert(0, str(PIPE_DIR))

import paper_figures as pf  # noqa: E402  (also puts pipeline on path)
from pipeline import (  # noqa: E402
    build_joint_dist, compute_pooled_metric, posterior_marginal,
    precompute_trajectories, strategy_from_params,
)

OUT = SCRIPT_DIR
FIG_DIR = OUT / "stuff to incorporate"
FIG_DIR.mkdir(exist_ok=True)
OBJECTIVES = ["combo_r", "agg_ll", "mean_ll"]

# ── pure Threshold-PS ────────────────────────────────────────────────

def _pure_dist(marg, prev, delta):
    cur = marg[prev]
    cand = [r for r in range(3) if r != prev and (marg[r] - cur) > delta]
    d = np.zeros(3)
    if not cand:
        d[prev] = 1.0
        return d
    w = np.array([marg[r] for r in cand])
    w = w / w.sum()
    for r, x in zip(cand, w):
        d[r] = x
    return d


def _predict(trajectory, delta):
    out = []
    for st in trajectory:
        prev = st["prev_roles"]
        per_agent = []
        for i in range(3):
            m = posterior_marginal(st["prior"], i)
            per_agent.append(m if prev is None else _pure_dist(m, prev[i], delta))
        out.append({"predicted_dist": build_joint_dist(per_agent),
                    "human_combo": st["human_combo"],
                    "model_marginal": np.mean(per_agent, axis=0)})
    return out


def thresh_ps_pure_factory(delta):
    def factory(records_subset, traj_subset):
        idx = {id(r): i for i, r in enumerate(records_subset)}
        return lambda rec: _predict(traj_subset[idx[id(rec)]], delta)
    return factory


def fit_delta(records, trajs, objective):
    f = lambda d: compute_pooled_metric(records, trajs, thresh_ps_pure_factory(d),
                                        metric=objective)
    grid = np.round(np.linspace(0.0, 1.0, 201), 4)
    vals = np.array([f(float(d)) for d in grid])
    ok = ~np.isnan(vals)
    best = vals[ok].max()
    plateau = grid[ok & (vals >= best - 1e-9)]
    fine = np.round(np.arange(max(0, plateau.min() - 0.005),
                              min(1, plateau.max() + 0.005) + 1e-9, 0.0005), 4)
    fv = np.array([f(float(d)) for d in fine])
    fok = ~np.isnan(fv)
    best = max(best, fv[fok].max())
    plateau = np.unique(np.concatenate([grid[ok & (vals >= best - 1e-9)],
                                        fine[fok & (fv >= best - 1e-9)]]))
    return {"delta": float(plateau.min()), "objective_value": float(best),
            "plateau": [float(plateau.min()), float(plateau.max())],
            "profile": {f"{d:.2f}": float(v) for d, v in zip(grid[::10], vals[::10])}}


# ── display names / palette (matches paper's R4) ─────────────────────

DISPLAY = {
    "Bayesian Walk": "Bayesian-Walk-BR", "Bayesian Walk-PS": "Bayesian-Walk-PS",
    "Bayesian-Belief": "Bayesian-PS", "Bayesian-Value": "Bayesian-BR",
    "Bayesian Threshold": "Bayesian-Threshold-BR",
    "Bayesian Thresh-PS": "Bayesian-Threshold-PS",
}
dn = lambda m: DISPLAY.get(m, m)
MODEL_ORDER = [m for m in pf.MODEL_ORDER if m != "Mixture-PS"]
BAYES = ["Bayesian-Value", "Bayesian Walk", "Bayesian Threshold",
         "Bayesian-Belief", "Bayesian Walk-PS", "Bayesian Thresh-PS"]  # legend order
BASE = ["Optimal", "Top-7", "Random-to-Optimal", "Copy Others",
        "Contradict Others", "Random Walk", "Random"]
PALETTE = {
    "Bayesian Walk": "#08519c", "Bayesian Walk-PS": "#9ecae1",
    "Bayesian-Belief": "#41ab5d", "Bayesian-Value": "#1c9099",
    "Bayesian Threshold": "#6a51a3", "Bayesian Thresh-PS": "#bcbddc",
    "Random Walk": "#e6550d", "Top-7": "#c51b7d", "Random-to-Optimal": "#8c510a",
    "Optimal": "#b2182b", "Copy Others": "#fff7bc", "Contradict Others": "#fb9a29",
    "Random": "#969696",
}


def posteriors_from_ll(lls, order):
    out = {}
    pids = list(next(iter(lls.values())))
    for p in pids:
        v = np.array([lls[m][p] for m in order])
        e = np.exp(v - v.max())
        out[p] = dict(zip(order, e / e.sum()))
    return out


def plot_r4(post, name="R4_individual_fitting"):
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "axes.linewidth": 0.6})
    dom = {p: max(MODEL_ORDER, key=lambda m: v[m]) for p, v in post.items()}
    idx = {m: i for i, m in enumerate(MODEL_ORDER)}
    pids = sorted(post, key=lambda p: (idx[dom[p]], -post[p][dom[p]]))
    fig, ax = plt.subplots(figsize=(3.42, 2.95))
    x = np.arange(len(pids)); bottoms = np.zeros(len(pids))
    for m in MODEL_ORDER:
        vals = np.array([post[p][m] for p in pids])
        ax.bar(x, vals, bottom=bottoms, width=1.0, color=PALETTE[m],
               edgecolor="white", linewidth=0.35)
        bottoms += vals
    assert np.allclose(bottoms, 1.0, atol=1e-9)
    ax.set_xlim(-0.5, len(pids) - 0.5); ax.set_ylim(0, 1)
    ax.set_xlabel(f"Participants (n = {len(pids)}, sorted by best-fitting model)")
    ax.set_ylabel("P(model | participant's choices)")
    ax.set_xticks([]); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.spines[["top", "right"]].set_visible(False)
    proxy = lambda m: mpatches.Patch(facecolor=PALETTE[m], edgecolor="white", linewidth=0.35)
    fig.tight_layout(pad=0.3, rect=(0, 0.23, 1, 1))
    kw = dict(loc="upper center", ncols=2, fontsize=5, title_fontsize=5.5, frameon=False,
              handlelength=1.2, handleheight=1.1, columnspacing=0.5, labelspacing=0.35,
              borderpad=0.2)
    l1 = fig.legend([proxy(m) for m in BAYES], [dn(m) for m in BAYES],
                    title="Bayesian models", bbox_to_anchor=(0.25, 0.22), **kw)
    l2 = fig.legend([proxy(m) for m in BASE], BASE,
                    title="Non-Bayesian baselines", bbox_to_anchor=(0.79, 0.22), **kw)
    for l in (l1, l2):
        l._legend_box.align = "left"
    for ext, k in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", facecolor="white", **k)
    plt.close(fig)
    return dom


# ── main ─────────────────────────────────────────────────────────────

def main():
    s1 = json.load(open(pf.STAGE1_PARAMS))
    strat = strategy_from_params(s1["memory_strategy"], s1.get("window"), s1.get("drift_delta", 0.0))
    cells = json.load(open(pf.PIPELINE_RESULTS))["cells"]
    old = json.load(open(pf.RESULTS_PATH))
    records = pf.load_records_with_pids(verbose=False)
    trajs = precompute_trajectories(records, s1["tau_prior"], s1["epsilon"], memory_strategy=strat)

    # 1. fit pure Threshold-PS under each objective
    fits = {}
    for obj in OBJECTIVES:
        fits[obj] = fit_delta(records, trajs, obj)
        print(f"pure Thresh-PS fit on {obj:8s}: delta={fits[obj]['delta']:.4f} "
              f"(plateau {fits[obj]['plateau']}) value={fits[obj]['objective_value']:.4f}")
    per_obj_eval = {}
    for obj in OBJECTIVES:
        fn = thresh_ps_pure_factory(fits[obj]["delta"])(records, trajs)
        per_obj_eval[obj] = {k: float(v) for k, v in pf.eval_subset(records, fn).items()
                             if isinstance(v, (int, float, np.floating))}

    # 2. 13 predict_fns: 05-28 set, Mixture-PS dropped, Thresh-PS swapped
    fns, params = pf.build_predict_fns(records, trajs, cells)
    fns.pop("Mixture-PS"); params.pop("Mixture-PS")
    fns["Bayesian Thresh-PS"] = thresh_ps_pure_factory(fits["agg_ll"]["delta"])(records, trajs)
    params["Bayesian Thresh-PS"] = {"delta": fits["agg_ll"]["delta"]}
    fns = {m: fns[m] for m in MODEL_ORDER}
    metrics = pf.evaluate_all(records, fns)

    # sanity: untouched models reproduce the 05-28 numbers
    for m in MODEL_ORDER:
        if m == "Bayesian Thresh-PS":
            continue
        for k in ("combo_r", "marg_r", "agg_ll", "mean_ll"):
            assert abs(metrics[m][k] - old["aggregate"]["metrics_by_model"][m][k]) < 1e-9, (m, k)
    print("unchanged models reproduce 05-28 metrics exactly")

    # 3. individual fit (same method as 05-28, pooled params, 5% lapse)
    margs = pf.precompute_player_marginals(records, fns)
    _, player_ll, _ = pf.individual_posteriors(records, margs)
    lls = {m: dict(d) for m, d in player_ll.items()}
    post = posteriors_from_ll(lls, MODEL_ORDER)
    dom = plot_r4(post)
    counts = Counter(dom.values())

    # old 13-model counts (renormalised stored LLs; Mixture-PS excluded)
    old_post = posteriors_from_ll(old["individual"]["log_likelihoods"], MODEL_ORDER)
    old_counts = Counter(max(MODEL_ORDER, key=lambda m: v[m]) for v in old_post.values())
    moved = Counter((max(MODEL_ORDER, key=lambda m: old_post[p][m]), dom[p])
                    for p in post if max(MODEL_ORDER, key=lambda m: old_post[p][m]) != dom[p])

    # 4. write outputs
    out = {"fits": fits, "per_objective_eval": per_obj_eval,
           "metrics_by_model": metrics, "params_by_model": params,
           "dominant_counts_new": dict(counts), "dominant_counts_old_13": dict(old_counts),
           "reassigned": {f"{a} -> {b}": n for (a, b), n in moved.items()},
           "individual_ll": lls, "n_participants": len(post), "lapse": pf.LAPSE}
    json.dump(out, open(OUT / "results.json", "w"), indent=2)
    write_md(out)
    write_tex(out)
    print("\nDominant counts (new vs old 13-model):")
    for m in MODEL_ORDER:
        print(f"  {dn(m):24s} {counts.get(m,0):3d}  (was {old_counts.get(m,0)})")
    print("reassigned:", dict(out["reassigned"]))


def write_md(o):
    L = ["# Pure Threshold-PS refit", "",
         "## Fitted delta (pure Threshold-PS, no eps_s)", "",
         "| Fit on | delta | plateau of equal-best delta | combo-r | marg-r | agg-LL | mean-LL |",
         "|---|--:|---|--:|--:|--:|--:|"]
    for obj in OBJECTIVES:
        f, e = o["fits"][obj], o["per_objective_eval"][obj]
        L.append(f"| {obj} | {f['delta']:.3f} | {f['plateau'][0]:.3f}–{f['plateau'][1]:.3f} | "
                 f"{e['combo_r']:.3f} | {e['marg_r']:.3f} | {e['agg_ll']:.3f} | {e['mean_ll']:.3f} |")
    L += ["", "## Aggregate table (agg_ll-fit params)", "",
          "| Model | combo-r | marg-r | agg-LL | mean-LL |", "|---|--:|--:|--:|--:|"]
    for m in MODEL_ORDER:
        x = o["metrics_by_model"][m]
        L.append(f"| {dn(m)} | {x['combo_r']:.3f} | {x['marg_r']:.3f} | {x['agg_ll']:.3f} | {x['mean_ll']:.3f} |")
    L += ["", "## Dominant model per participant (n=102)", "", "| Model | new | old (13-model) |", "|---|--:|--:|"]
    for m in MODEL_ORDER:
        L.append(f"| {dn(m)} | {o['dominant_counts_new'].get(m,0)} | {o['dominant_counts_old_13'].get(m,0)} |")
    L += ["", "Reassigned participants: " + (", ".join(f"{k}: {v}" for k, v in o["reassigned"].items()) or "none"), "",
          "## delta profile (agg_ll), every 0.1", "", "| delta | agg_ll |", "|--:|--:|"]
    L += [f"| {d} | {v:.4f} |" for d, v in o["fits"]["agg_ll"]["profile"].items()]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")


def dstr(f):
    lo, hi = f["plateau"]
    return rf"\ge {lo:.3f}" if hi - lo > 0.1 else f"= {f['delta']:.2f}"


def write_tex(o):
    keys = ["combo_r", "marg_r", "agg_ll", "mean_ll"]
    BR = ["Bayesian-Value", "Bayesian Walk", "Bayesian Threshold"]
    PS = ["Bayesian-Belief", "Bayesian Walk-PS", "Bayesian Thresh-PS"]
    allm = o["metrics_by_model"]
    best = {k: max(v[k] for v in allm.values()) for k in keys}
    def fmt(k, v):
        s = f"{v:.3f}".replace("-", "$-$") if v < 0 else f"{v:.3f}"
        if abs(v - best[k]) < 5e-4:
            s = r"$\mathbf{" + f"{v:.3f}" + "}$" if v < 0 else r"\textbf{" + s + "}"
        return s
    rows = [f"{dn(m):<22s}& " + " & ".join(fmt(k, allm[m][k]) for k in keys) + r" \\" for m in BR + PS]
    rows.insert(3, "% ---")
    L = ["% main-text Table (aggregate): 6 Bayesian rows (order as in paper: BR then PS)", *rows, ""]
    L += ["% SI Table S3 rows for Bayesian-Threshold-PS (a delta plateau is reported as >=)"]
    for i, obj in enumerate(OBJECTIVES):
        f, e = o["fits"][obj], o["per_objective_eval"][obj]
        lab = {"combo_r": "combo-$r$", "agg_ll": "agg-LL", "mean_ll": "mean-LL"}[obj]
        fm = lambda v: f"{v:.3f}".replace("-", "$-$")
        L.append(("Bayesian-Threshold-PS" if i == 0 else "") + f" & {lab} & "
                 f"$\\delta_{{\\mathrm{{PS}}}}{{{dstr(f)}}}$ & {e['combo_r']:.3f} & {e['marg_r']:.3f} "
                 f"& {fm(e['agg_ll'])} & {fm(e['mean_ll'])} \\\\")
    (OUT / "tables_snippets.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
