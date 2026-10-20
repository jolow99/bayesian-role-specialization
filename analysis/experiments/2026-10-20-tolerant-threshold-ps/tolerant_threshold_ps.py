"""2026-10-20 — tolerant Threshold-PS + lapse (λ = 0.01, no floor) refit.

Context (2026-10 Slack thread with Xuan / Joseph): the 10-15 pure
Threshold-PS never includes the current role in its candidate set, so it
forces a switch whenever any role clears δ and assigns 0 to staying and to
every non-admissible switch. δ = 0 therefore does not recover Bayesian-PS,
every δ gives −∞ log-likelihood, and the reported numbers only exist
because of the 1e-12 / 1e-20 floors. Xuan's fix (option 2):

    allowed(δ) = {prev} ∪ {r ≠ prev : P(r) − P(prev) > δ},   δ ∈ [−1, 1]
    pi(r)      = posterior renormalised over allowed(δ)

δ > 0: only switch to roles clearly better than the current one.
δ < 0: tolerate switches to roles up to |δ| less likely ("tolerance").
δ = −1: every role allowed → exactly Bayesian-PS.

Likelihood scoring is standardised to a λ = 0.01 uniform lapse with no
floor, in both fits:
  * aggregate (agg-LL, mean-LL): joint over the 27 team combos,
    (1 − λ)·p + λ/27  (a data point is a team-stage)
  * individual: per-player marginal, (1 − λ)·p + λ/3  (a data point is a
    player-stage; same placement as the 05-28 5% lapse)
combo-r / marg-r are correlations, not likelihoods, so they use the raw
(un-lapsed) predictions, as before.

Everything else matches 10-15: same records, Stage-1 params, and the other
12 models held at their paper (agg-LL) fits. Those were fit under the old
floored objective and are NOT refit here; Bayesian-Threshold-BR is still
the old forced-switch version.

Run from analysis/:
    uv run python experiments/2026-10-20-tolerant-threshold-ps/tolerant_threshold_ps.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
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
    build_joint_dist, posterior_marginal, precompute_trajectories,
    strategy_from_params,
)
from shared.constants import ROLE_SHORT  # noqa: E402
from shared.evaluation import compute_pearson, extract_metrics, run_predictions  # noqa: E402
from shared.parsing import canonical_combo  # noqa: E402

OUT = SCRIPT_DIR
FIG_DIR = OUT / "stuff to incorporate"
FIG_DIR.mkdir(exist_ok=True)
OBJECTIVES = ["combo_r", "agg_ll", "mean_ll"]
LAPSE = 0.01
ALL_COMBOS = [ROLE_SHORT[a] + ROLE_SHORT[b] + ROLE_SHORT[c]
              for a in range(3) for b in range(3) for c in range(3)]
ROLE_IDX = {ROLE_SHORT[i]: i for i in range(3)}

# ── tolerant Threshold-PS ────────────────────────────────────────────

def _tolerant_dist(marg, prev, delta):
    cur = marg[prev]
    allowed = [prev] + [r for r in range(3) if r != prev and (marg[r] - cur) > delta]
    d = np.zeros(3)
    d[allowed] = marg[allowed] / marg[allowed].sum()
    return d


def _predict(trajectory, delta):
    out = []
    for st in trajectory:
        prev = st["prev_roles"]
        per_agent = []
        for i in range(3):
            m = posterior_marginal(st["prior"], i)
            per_agent.append(m if prev is None else _tolerant_dist(m, prev[i], delta))
        out.append({"predicted_dist": build_joint_dist(per_agent),
                    "human_combo": st["human_combo"],
                    "model_marginal": np.mean(per_agent, axis=0)})
    return out


def tolerant_factory(delta):
    def factory(records_subset, traj_subset):
        idx = {id(r): i for i, r in enumerate(records_subset)}
        return lambda rec: _predict(traj_subset[idx[id(rec)]], delta)
    return factory

# ── lapse metrics (no floor) ─────────────────────────────────────────

def _class_sizes(stat_profile):
    return Counter(canonical_combo(c, stat_profile) for c in ALL_COMBOS)


def lapse_metrics(records, predict_fn, lam=LAPSE):
    """combo-r / marg-r on raw predictions; agg-LL / mean-LL under a λ lapse
    on the 27-combo joint, with no floor (a zero would give −inf)."""
    full = run_predictions(records, predict_fn)
    out = extract_metrics(compute_pearson(full))

    total_ce = total_w = 0.0
    env_means = []
    for data in full.values():
        sizes = _class_sizes(data["stat_profile"])
        for s in range(data["max_stages"]):
            pred = data["stage_predicted"].get(s)
            n = data["stage_counts"].get(s, 0)
            if pred is None or n == 0:
                continue
            for cc, k in data["stage_human"].get(s, {}).items():
                p = (1 - lam) * pred.get(cc, 0.0) + lam * sizes[cc] / 27
                total_ce += k * np.log(p)
            total_w += n
        lls = []
        for team in data["team_predictions"]:
            for pr in team:
                assert set(pr["predicted_dist"]) <= set(ALL_COMBOS)
                p = (1 - lam) * pr["predicted_dist"].get(pr["human_combo"], 0.0) + lam / 27
                lls.append(np.log(p))
        env_means.append(np.mean(lls))
    out["agg_ll"] = float(total_ce / total_w)
    out["mean_ll"] = float(np.mean(env_means))
    return {k: float(v) for k, v in out.items()}


def individual_ll(records, margs, lam=LAPSE):
    """Per-participant log-likelihood, λ lapse on each player's marginal, no floor."""
    ll = {name: defaultdict(float) for name in margs}
    for ri, rec in enumerate(records):
        for s, combo in enumerate(rec["stage_roles"]):
            for pos in range(3):
                pid = rec["participant_ids"][pos]
                if pid is None:
                    continue
                a = ROLE_IDX[combo[pos]]
                for name in margs:
                    stages = margs[name][ri]
                    m = stages[s][pos] if s < len(stages) else np.ones(3) / 3.0
                    ll[name][pid] += float(np.log((1 - lam) * m[a] + lam / 3))
    return {m: dict(d) for m, d in ll.items()}

# ── fitting ──────────────────────────────────────────────────────────

def _metrics_at(records, trajs, delta):
    return lapse_metrics(records, tolerant_factory(delta)(records, trajs))


def fit_delta(records, trajs):
    """Grid over [-1, 1] (step 0.01) then a 0.0005 sweep around each
    objective's best; the objective is piecewise constant in δ, so we report
    the plateau of equal-best δ and use its lower end (as in 10-15)."""
    cache = {}
    def at(d):
        d = round(float(d), 4)
        if d not in cache:
            cache[d] = _metrics_at(records, trajs, d)
        return cache[d]

    grid = np.round(np.linspace(-1.0, 1.0, 201), 4)
    for d in grid:
        at(d)
    fits = {}
    for obj in OBJECTIVES:
        best = max(cache[d][obj] for d in grid)
        tops = [d for d in grid if cache[d][obj] >= best - 1e-9]
        for d in np.round(np.arange(max(-1, min(tops) - 0.01),
                                    min(1, max(tops) + 0.01) + 1e-9, 0.0005), 4):
            at(d)
        best = max(v[obj] for v in cache.values())
        plateau = sorted(d for d, v in cache.items() if v[obj] >= best - 1e-9)
        fits[obj] = {"delta": plateau[0], "objective_value": float(best),
                     "plateau": [plateau[0], plateau[-1]],
                     "profile": {f"{d:.2f}": cache[round(float(d), 4)][obj] for d in grid[::10]}}
    return fits


def switch_gaps(records, trajs):
    """P(chosen) − P(prev) for every observed switch: the most negative one is
    the δ the likelihood fit needs to go below to give every switch non-zero
    (pre-lapse) probability."""
    gaps = []
    for traj in trajs:
        for st in traj:
            prev = st["prev_roles"]
            if prev is None:
                continue
            for i in range(3):
                a = ROLE_IDX[st["human_combo"][i]]
                if a != prev[i]:
                    m = posterior_marginal(st["prior"], i)
                    gaps.append(float(m[a] - m[prev[i]]))
    g = np.array(gaps)
    return {"n_switches": int(g.size), "min": float(g.min()),
            "pct": {str(q): float(np.percentile(g, q)) for q in (1, 5, 25, 50, 75, 95)},
            "frac_below_0": float((g <= 0).mean())}

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
    paper = json.load(open(EXP / "2026-10-15-pure-threshold-ps" / "results.json"))
    records = pf.load_records_with_pids(verbose=False)
    trajs = precompute_trajectories(records, s1["tau_prior"], s1["epsilon"], memory_strategy=strat)

    # 0. sanity: untouched models reproduce the paper (floored) numbers, which
    #    also checks the local data + value matrices match the paper's
    fns, params = pf.build_predict_fns(records, trajs, cells)
    fns.pop("Mixture-PS"); params.pop("Mixture-PS"); fns.pop("Bayesian Thresh-PS")
    floored = pf.evaluate_all(records, fns)
    for m in fns:
        for k in ("combo_r", "marg_r", "agg_ll", "mean_ll"):
            assert abs(floored[m][k] - old["aggregate"]["metrics_by_model"][m][k]) < 1e-9, (m, k)
    print("untouched models reproduce the paper's (floored) metrics exactly")

    # 1. δ = −1 must be exactly Bayesian-PS
    tol = tolerant_factory(-1.0)(records, trajs)
    for rec in records:
        for a, b in zip(tol(rec), fns["Bayesian-Belief"](rec)):
            assert all(abs(a["predicted_dist"][c] - b["predicted_dist"].get(c, 0.0)) < 1e-12
                       for c in ALL_COMBOS)
    print("tolerant Threshold-PS at delta=-1 reproduces Bayesian-PS exactly")

    # 2. fit δ under each objective (lapse, no floor)
    fits = fit_delta(records, trajs)
    per_obj_eval = {obj: _metrics_at(records, trajs, fits[obj]["delta"]) for obj in OBJECTIVES}
    for obj in OBJECTIVES:
        print(f"tolerant Thresh-PS fit on {obj:8s}: delta={fits[obj]['delta']:.4f} "
              f"(plateau {fits[obj]['plateau']}) value={fits[obj]['objective_value']:.4f}")
    gaps = switch_gaps(records, trajs)

    # 3. all 13 models under the lapse metrics (others at their paper params)
    fns["Bayesian Thresh-PS"] = tolerant_factory(fits["agg_ll"]["delta"])(records, trajs)
    params["Bayesian Thresh-PS"] = {"delta": fits["agg_ll"]["delta"]}
    fns = {m: fns[m] for m in MODEL_ORDER}
    metrics = {m: lapse_metrics(records, fn) for m, fn in fns.items()}
    ps = metrics["Bayesian-Belief"]
    assert metrics["Bayesian Thresh-PS"]["agg_ll"] >= ps["agg_ll"] - 1e-12

    # 4. individual fit (pooled params, λ lapse on marginals, no floor)
    margs = pf.precompute_player_marginals(records, fns)
    lls = individual_ll(records, margs)
    post = posteriors_from_ll(lls, MODEL_ORDER)
    dom = plot_r4(post)
    counts = Counter(dom.values())
    paper_post = posteriors_from_ll(paper["individual_ll"], MODEL_ORDER)
    paper_counts = Counter(max(MODEL_ORDER, key=lambda m: v[m]) for v in paper_post.values())

    # 5. write outputs
    out = {"lapse": LAPSE, "fits": fits, "per_objective_eval": per_obj_eval,
           "switch_gaps": gaps, "metrics_by_model": metrics,
           "paper_metrics_by_model": paper["metrics_by_model"],
           "params_by_model": params, "dominant_counts_new": dict(counts),
           "dominant_counts_paper": dict(paper_counts),
           "individual_ll": lls, "n_participants": len(post)}
    json.dump(out, open(OUT / "results.json", "w"), indent=2)
    write_md(out)
    write_tex(out)
    print("\nDominant counts (new vs paper):")
    for m in MODEL_ORDER:
        print(f"  {dn(m):24s} {counts.get(m,0):3d}  (paper {paper_counts.get(m,0)})")


def write_md(o):
    g = o["switch_gaps"]
    L = [f"# Tolerant Threshold-PS refit (λ = {o['lapse']}, no floor)", "",
         "## Fitted delta", "",
         "| Fit on | delta | plateau of equal-best delta | combo-r | marg-r | agg-LL | mean-LL |",
         "|---|--:|---|--:|--:|--:|--:|"]
    for obj in OBJECTIVES:
        f, e = o["fits"][obj], o["per_objective_eval"][obj]
        L.append(f"| {obj} | {f['delta']:.3f} | {f['plateau'][0]:.3f} to {f['plateau'][1]:.3f} | "
                 f"{e['combo_r']:.3f} | {e['marg_r']:.3f} | {e['agg_ll']:.3f} | {e['mean_ll']:.3f} |")
    L += ["", "delta = -1 is exactly Bayesian-PS (asserted).", "",
          "## Observed switches: P(chosen) − P(previous role)", "",
          f"{g['n_switches']} switches; min {g['min']:.3f}; "
          f"{100 * g['frac_below_0']:.1f}% go to a role no more likely than the current one.", "",
          "| percentile | " + " | ".join(g["pct"]) + " |",
          "|---|" + "--:|" * len(g["pct"]),
          "| gap | " + " | ".join(f"{v:.3f}" for v in g["pct"].values()) + " |", "",
          "## Aggregate table (agg-LL-fit δ; other models at paper params)", "",
          f"agg-LL / mean-LL use a λ = {o['lapse']} lapse with no floor; combo-r / marg-r "
          "are on raw predictions. Paper columns = current paper (1e-12 / 1e-20 floors).", "",
          "| Model | combo-r | marg-r | agg-LL | mean-LL | agg-LL (paper) | mean-LL (paper) |",
          "|---|--:|--:|--:|--:|--:|--:|"]
    for m in MODEL_ORDER:
        x, p = o["metrics_by_model"][m], o["paper_metrics_by_model"][m]
        L.append(f"| {dn(m)} | {x['combo_r']:.3f} | {x['marg_r']:.3f} | {x['agg_ll']:.3f} | "
                 f"{x['mean_ll']:.3f} | {p['agg_ll']:.3f} | {p['mean_ll']:.3f} |")
    L += ["", f"## Dominant model per participant (n={o['n_participants']})", "",
          f"| Model | new (λ = {o['lapse']}) | paper (λ = 0.05, old Thresh-PS) |", "|---|--:|--:|"]
    for m in MODEL_ORDER:
        L.append(f"| {dn(m)} | {o['dominant_counts_new'].get(m,0)} | {o['dominant_counts_paper'].get(m,0)} |")
    L += ["", "## delta profile (agg-LL), every 0.1", "", "| delta | agg-LL |", "|--:|--:|"]
    L += [f"| {d} | {v:.4f} |" for d, v in o["fits"]["agg_ll"]["profile"].items()]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")


def dstr(f):
    lo, hi = f["plateau"]
    return f"\\in [{lo:.3f}, {hi:.3f}]" if hi - lo > 0.05 else f"= {f['delta']:.3f}"


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
    L = [f"% main-text Table (aggregate), lambda = {o['lapse']} lapse, no floor: 6 Bayesian rows", *rows, ""]
    L += ["% SI fitted-params rows for Bayesian-Threshold-PS (tolerant version)"]
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
