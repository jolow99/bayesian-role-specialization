"""2026-10-20 — refit every role-selection model under a λ = 0.01 lapse, no floor.

The two 10-20 tolerant-threshold folders fit only Threshold-PS / Threshold-BR
under the lapse; every other model kept its 05-25 parameters, which were fit
under the floored objective (the floor never triggered for any model with
free parameters, but the 1% lapse still shifts their optimum slightly). Here
every model with a free parameter is refit with the same scoring:

  * agg-LL / mean-LL: (1 − λ)·p + λ/27 on the 27-combo joint, no floor
  * combo-r / marg-r: raw predictions
  * individual fit: (1 − λ)·p + λ/3 on each player's marginal, no floor

Scoring, the tolerant threshold models and their fitting routines are
imported from the two 2026-10-20 tolerant-threshold folders. BR and Walk-BR
are evaluated from precomputed expected values (asserted identical to the
05-25 model code) so the fits are fast.

Free parameters and search:
  Bayesian-BR        τ_v                 grid (20, log [0.5, 50]) → bounded 1-D [0.01, 50]
  Bayesian-Walk-BR   τ_v, ε_switch       grid (15 × 21) → L-BFGS-B
  Bayesian-Walk-PS   ε_switch            grid (21) → bounded 1-D [0, 1]
  Threshold-PS       δ                   tolerant-threshold-ps fit_delta
  Threshold-BR       τ_v, δ              tolerant-threshold-br fit_tau_delta
  Random Walk        ε (switch prob.)    grid (21) → bounded 1-D [0, 1]
  Top-k              k                   exhaustive, 1..27
ε_switch is the code's re-decide probability; the paper's ε_s = 1 − ε_switch.

Run from analysis/ (needs both tolerant-threshold folders):
    uv run python experiments/2026-10-20-lapse-refit-all-models/refit_all.py
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
from scipy.optimize import minimize, minimize_scalar

SCRIPT_DIR = Path(__file__).resolve().parent
EXP = SCRIPT_DIR.parent
for d in ("2026-05-28-paper-figures", "2026-05-25-full-pipeline",
          "2026-10-20-tolerant-threshold-ps", "2026-10-20-tolerant-threshold-br"):
    sys.path.insert(0, str(EXP / d))

import paper_figures as pf  # noqa: E402
import tolerant_threshold_br as tbr  # noqa: E402
import tolerant_threshold_ps as tps  # noqa: E402
from models import walk_ps_factory  # noqa: E402
from pipeline import build_joint_dist, precompute_trajectories, strategy_from_params  # noqa: E402

OUT = SCRIPT_DIR
FIG_DIR = OUT / "stuff to incorporate"
FIG_DIR.mkdir(exist_ok=True)
OBJECTIVES = tps.OBJECTIVES
LAPSE = tps.LAPSE
MODEL_ORDER, ALL_COMBOS = tps.MODEL_ORDER, tps.ALL_COMBOS
TABLE_ORDER = ["Bayesian-Value", "Bayesian Walk", "Bayesian Threshold",
               "Bayesian-Belief", "Bayesian Walk-PS", "Bayesian Thresh-PS",
               "Optimal", "Top-7", "Random-to-Optimal", "Copy Others",
               "Contradict Others", "Random Walk", "Random"]  # paper Table 2 order

# ── BR / Walk-BR from precomputed expected values ────────────────────

def _br_predict(trajectory, evs, tau, eps_switch):
    out = []
    for st, ev in zip(trajectory, evs):
        prev = st["prev_roles"]
        per_agent = []
        for i in range(3):
            sm = tbr._softmax(ev[i], tau)
            if prev is None or eps_switch is None:
                per_agent.append(sm)
            else:
                stick = np.zeros(3); stick[prev[i]] = 1.0
                per_agent.append((1.0 - eps_switch) * stick + eps_switch * sm)
        out.append({"predicted_dist": build_joint_dist(per_agent),
                    "human_combo": st["human_combo"],
                    "model_marginal": np.mean(per_agent, axis=0)})
    return out


def br_factory(tau, evs_all, eps_switch=None):
    """eps_switch=None → Bayesian-BR; otherwise Bayesian-Walk-BR."""
    def factory(records_subset, traj_subset):
        idx = {id(r): i for i, r in enumerate(records_subset)}
        return lambda rec: _br_predict(traj_subset[idx[id(rec)]], evs_all[idx[id(rec)]],
                                       tau, eps_switch)
    return factory


def _same_predictions(records, fa, fb, tol=1e-12):
    return all(abs(a["predicted_dist"][c] - b["predicted_dist"].get(c, 0.0)) < tol
               for rec in records for a, b in zip(fa(rec), fb(rec)) for c in ALL_COMBOS)

# ── generic fitters (maximise each objective) ────────────────────────

def fit_1d(records, make_fn, grid, lo, hi):
    cache = {}
    def at(x):
        x = round(float(x), 6)
        if x not in cache:
            cache[x] = tps.lapse_metrics(records, make_fn(x))
        return cache[x]
    for x in grid:
        at(x)
    fits = {}
    for obj in OBJECTIVES:
        xs = sorted(cache)
        i = max(range(len(xs)), key=lambda j: cache[xs[j]][obj])
        a, b = xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]
        if b > a:
            minimize_scalar(lambda x: -at(x)[obj], bounds=(max(lo, a), min(hi, b)),
                            method="bounded", options={"xatol": 1e-4})
        x_best = max(cache, key=lambda x: cache[x][obj])
        fits[obj] = {"x": x_best, "objective_value": cache[x_best][obj]}
    return fits


def fit_walk_br(records, trajs, evs):
    cache = {}
    def at(tau, eps):
        key = (round(float(tau), 6), round(float(eps), 6))
        if key not in cache:
            cache[key] = tps.lapse_metrics(records, br_factory(key[0], evs, key[1])(records, trajs))
        return cache[key]
    for t in np.geomspace(0.5, 50.0, 15):
        for e in np.linspace(0.0, 1.0, 21):
            at(t, e)
    fits = {}
    for obj in OBJECTIVES:
        t0, e0 = max(cache, key=lambda k: cache[k][obj])
        minimize(lambda v: -at(v[0], v[1])[obj], x0=[t0, e0], method="L-BFGS-B",
                 bounds=[(0.01, 50.0), (0.0, 1.0)], options={"eps": 1e-4})
        tb, eb = max(cache, key=lambda k: cache[k][obj])
        fits[obj] = {"tau_softmax": tb, "epsilon_switch": eb, "objective_value": cache[(tb, eb)][obj]}
    return fits


def fit_top_k(records):
    evals = {k: tps.lapse_metrics(records, pf._wrap_baseline(pf.make_predict_top_k(k)))
             for k in range(1, 28)}
    return {obj: {"k": max(evals, key=lambda k: evals[k][obj]),
                  "objective_value": max(v[obj] for v in evals.values())} for obj in OBJECTIVES}, evals

# ── R4 plot (as in the tolerant-threshold folders, Top-k label follows fitted k) ──

def plot_r4(post, label):
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
        ax.bar(x, vals, bottom=bottoms, width=1.0, color=tps.PALETTE[m],
               edgecolor="white", linewidth=0.35)
        bottoms += vals
    assert np.allclose(bottoms, 1.0, atol=1e-9)
    ax.set_xlim(-0.5, len(pids) - 0.5); ax.set_ylim(0, 1)
    ax.set_xlabel(f"Participants (n = {len(pids)}, sorted by best-fitting model)")
    ax.set_ylabel("P(model | participant's choices)")
    ax.set_xticks([]); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.spines[["top", "right"]].set_visible(False)
    proxy = lambda m: mpatches.Patch(facecolor=tps.PALETTE[m], edgecolor="white", linewidth=0.35)
    fig.tight_layout(pad=0.3, rect=(0, 0.23, 1, 1))
    kw = dict(loc="upper center", ncols=2, fontsize=5, title_fontsize=5.5, frameon=False,
              handlelength=1.2, handleheight=1.1, columnspacing=0.5, labelspacing=0.35,
              borderpad=0.2)
    l1 = fig.legend([proxy(m) for m in tps.BAYES], [label(m) for m in tps.BAYES],
                    title="Bayesian models", bbox_to_anchor=(0.25, 0.22), **kw)
    l2 = fig.legend([proxy(m) for m in tps.BASE], [label(m) for m in tps.BASE],
                    title="Non-Bayesian baselines", bbox_to_anchor=(0.79, 0.22), **kw)
    for l in (l1, l2):
        l._legend_box.align = "left"
    for ext, k in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(FIG_DIR / f"R4_individual_fitting.{ext}", bbox_inches="tight",
                    facecolor="white", **k)
    plt.close(fig)
    return dom

# ── main ─────────────────────────────────────────────────────────────

def main():
    s1 = json.load(open(pf.STAGE1_PARAMS))
    strat = strategy_from_params(s1["memory_strategy"], s1.get("window"), s1.get("drift_delta", 0.0))
    cells = json.load(open(pf.PIPELINE_RESULTS))["cells"]
    old = json.load(open(pf.RESULTS_PATH))
    paper = json.load(open(EXP / "2026-10-15-pure-threshold-ps" / "results.json"))
    br_run = json.load(open(EXP / "2026-10-20-tolerant-threshold-br" / "results.json"))
    records = pf.load_records_with_pids(verbose=False)
    trajs = precompute_trajectories(records, s1["tau_prior"], s1["epsilon"], memory_strategy=strat)
    evs = tbr.precompute_evs(records, trajs)

    # 0. sanity: paper models reproduce the paper (floored) numbers; the fast
    #    EV-based BR / Walk-BR match the 05-25 model code exactly
    fns0, params0 = pf.build_predict_fns(records, trajs, cells)
    for m in ("Mixture-PS", "Bayesian Thresh-PS", "Bayesian Threshold"):
        fns0.pop(m); params0.pop(m)
    floored = pf.evaluate_all(records, fns0)
    for m in fns0:
        for k in ("combo_r", "marg_r", "agg_ll", "mean_ll"):
            assert abs(floored[m][k] - old["aggregate"]["metrics_by_model"][m][k]) < 1e-9, (m, k)
    vp, wp = params0["Bayesian-Value"], params0["Bayesian Walk"]
    assert _same_predictions(records, br_factory(vp["tau_softmax"], evs)(records, trajs),
                             fns0["Bayesian-Value"])
    assert _same_predictions(records, br_factory(wp["tau_softmax"], evs, wp["epsilon_switch"])(records, trajs),
                             fns0["Bayesian Walk"])
    print("paper numbers reproduced; EV-based BR / Walk-BR match the model code")

    # 1. refit every model with free parameters
    fits = {}
    f = fit_1d(records, lambda t: br_factory(t, evs)(records, trajs),
               np.geomspace(0.5, 50.0, 20), 0.01, 50.0)
    fits["Bayesian-Value"] = {o: {"tau_softmax": v["x"], "objective_value": v["objective_value"]}
                              for o, v in f.items()}
    print("fit Bayesian-BR"); fits["Bayesian Walk"] = fit_walk_br(records, trajs, evs)
    print("fit Bayesian-Walk-BR")
    f = fit_1d(records, lambda e: walk_ps_factory(e)(records, trajs), np.linspace(0, 1, 21), 0.0, 1.0)
    fits["Bayesian Walk-PS"] = {o: {"epsilon_switch": v["x"], "objective_value": v["objective_value"]}
                                for o, v in f.items()}
    f = fit_1d(records, lambda e: pf._wrap_baseline(pf.make_predict_random_walk(e)),
               np.linspace(0, 1, 21), 0.0, 1.0)
    fits["Random Walk"] = {o: {"eps": v["x"], "objective_value": v["objective_value"]}
                           for o, v in f.items()}
    fits["Top-7"], topk_profile = fit_top_k(records)
    print("fit Walk-PS, Random Walk, Top-k")
    f = tps.fit_delta(records, trajs)
    fits["Bayesian Thresh-PS"] = {o: {"delta": v["delta"], "objective_value": v["objective_value"],
                                      "delta_plateau": v["plateau"]} for o, v in f.items()}
    f = tbr.fit_tau_delta(records, trajs, evs)
    fits["Bayesian Threshold"] = {o: {"tau_softmax": v["tau_v"], "delta": v["delta"],
                                      "objective_value": v["objective_value"],
                                      "delta_plateau": v["delta_plateau"]} for o, v in f.items()}

    def build(m, p):
        if m == "Bayesian-Value":
            return br_factory(p["tau_softmax"], evs)(records, trajs)
        if m == "Bayesian Walk":
            return br_factory(p["tau_softmax"], evs, p["epsilon_switch"])(records, trajs)
        if m == "Bayesian Walk-PS":
            return walk_ps_factory(p["epsilon_switch"])(records, trajs)
        if m == "Random Walk":
            return pf._wrap_baseline(pf.make_predict_random_walk(p["eps"]))
        if m == "Top-7":
            return pf._wrap_baseline(pf.make_predict_top_k(p["k"]))
        if m == "Bayesian Thresh-PS":
            return tps.tolerant_factory(p["delta"])(records, trajs)
        if m == "Bayesian Threshold":
            return tbr.tolerant_br_factory(p["tau_softmax"], p["delta"], evs)(records, trajs)
        raise KeyError(m)

    per_obj_eval = {m: {o: tps.lapse_metrics(records, build(m, fits[m][o])) for o in OBJECTIVES}
                    for m in fits}

    # refits must do at least as well as the paper params under the same scoring
    old_params = {"Bayesian-Value": vp, "Bayesian Walk": wp,
                  "Bayesian Walk-PS": params0["Bayesian Walk-PS"],
                  "Random Walk": params0["Random Walk"], "Top-7": {"k": 7}}
    at_old = {m: tps.lapse_metrics(records, build(m, p)) for m, p in old_params.items()}
    for m in old_params:
        for o in OBJECTIVES:
            assert per_obj_eval[m][o][o] >= at_old[m][o] - 1e-9, (m, o)

    # 2. all 13 models at their agg-LL fits
    fns = {}
    params = {}
    for m in MODEL_ORDER:
        if m in fits:
            params[m] = {k: v for k, v in fits[m]["agg_ll"].items()
                         if k not in ("objective_value", "delta_plateau")}
            fns[m] = build(m, params[m])
        else:
            params[m] = {}
            fns[m] = fns0[m]
    metrics = {m: tps.lapse_metrics(records, fn) for m, fn in fns.items()}
    k_fit = params["Top-7"]["k"]
    label = lambda m: f"Top-{k_fit}" if m == "Top-7" else tps.dn(m)

    # 3. individual fit
    margs = pf.precompute_player_marginals(records, fns)
    lls = tps.individual_ll(records, margs)
    post = tps.posteriors_from_ll(lls, MODEL_ORDER)
    dom = plot_r4(post, label)
    counts = Counter(dom.values())
    paper_post = tps.posteriors_from_ll(paper["individual_ll"], MODEL_ORDER)
    paper_counts = Counter(max(MODEL_ORDER, key=lambda m: v[m]) for v in paper_post.values())

    # 4. outputs
    out = {"lapse": LAPSE, "fits": fits, "per_objective_eval": per_obj_eval,
           "metrics_at_paper_params_lapse": at_old, "params_by_model": params,
           "metrics_by_model": metrics, "br_folder_metrics_by_model": br_run["metrics_by_model"],
           "paper_metrics_by_model": paper["metrics_by_model"],
           "top_k_profile": {str(k): v for k, v in topk_profile.items()},
           "dominant_counts_new": dict(counts),
           "dominant_counts_br_folder": br_run["dominant_counts_new"],
           "dominant_counts_paper": dict(paper_counts),
           "individual_ll": lls, "n_participants": len(post)}
    json.dump(out, open(OUT / "results.json", "w"), indent=2)
    write_md(out, label)
    write_tex(out, label)
    for m in TABLE_ORDER:
        x = metrics[m]
        print(f"  {label(m):24s} {params[m]}  agg-LL {x['agg_ll']:.3f}  "
              f"dominant {counts.get(m, 0)} (paper {paper_counts.get(m, 0)})")


def pstr(m, p, tex=False):
    """Fitted params in paper notation (ε_s = 1 − ε_switch = stick probability)."""
    sub = "BR" if m == "Bayesian Threshold" else "PS"
    items = []  # (tex symbol, plain symbol, value)
    if "tau_softmax" in p:
        items.append((r"\tau_v", "τ_v", f"{p['tau_softmax']:.2f}"))
    if "epsilon_switch" in p:
        items.append((r"\varepsilon_s", "ε_s", f"{1 - p['epsilon_switch']:.2f}"))
    if "delta" in p:
        items.append((rf"\delta_{{\mathrm{{{sub}}}}}", f"δ_{sub}", f"{p['delta']:.3f}"))
    if "eps" in p:
        items.append((r"\varepsilon", "ε", f"{p['eps']:.3f}"))
    if "k" in p:
        items.append(("k", "k", str(p["k"])))
    if not items:
        return "none"
    if tex:
        return ", ".join(f"${s}{{=}}{v}$" for s, _, v in items)
    return ", ".join(f"{u}={v}" for _, u, v in items)


def write_md(o, label):
    L = [f"# Refit of every model under a λ = {o['lapse']} lapse, no floor", "",
         "## Fitted parameters per objective", "",
         "ε_s = probability of sticking with the previous role (= 1 − code's ε_switch). "
         "Random Walk ε = probability of switching.", "",
         "| Model | Fit on | Params | combo-r | marg-r | agg-LL | mean-LL |",
         "|---|---|---|--:|--:|--:|--:|"]
    for m in TABLE_ORDER:
        if m not in o["fits"]:
            continue
        for obj in OBJECTIVES:
            e = o["per_objective_eval"][m][obj]
            p = {k: v for k, v in o["fits"][m][obj].items()
                 if k not in ("objective_value", "delta_plateau")}
            L.append(f"| {label(m) if obj == OBJECTIVES[0] else ''} | {obj} | {pstr(m, p)} | "
                     f"{e['combo_r']:.3f} | {e['marg_r']:.3f} | {e['agg_ll']:.3f} | {e['mean_ll']:.3f} |")
    L += ["", "## Aggregate table (agg-LL fits)", "",
          f"New columns: every model fit and scored with λ = {o['lapse']}, no floor. "
          "`BR folder` = same scoring, paper params for non-threshold models. "
          "`paper` = current paper (floored).", "",
          "| Model | Params | combo-r | marg-r | agg-LL | mean-LL | agg-LL (BR folder) | agg-LL (paper) |",
          "|---|---|--:|--:|--:|--:|--:|--:|"]
    for m in TABLE_ORDER:
        x = o["metrics_by_model"][m]
        L.append(f"| {label(m)} | {pstr(m, o['params_by_model'][m])} | {x['combo_r']:.3f} | "
                 f"{x['marg_r']:.3f} | {x['agg_ll']:.3f} | {x['mean_ll']:.3f} | "
                 f"{o['br_folder_metrics_by_model'][m]['agg_ll']:.3f} | "
                 f"{o['paper_metrics_by_model'][m]['agg_ll']:.3f} |")
    L += ["", f"## Dominant model per participant (n={o['n_participants']})", "",
          "| Model | new (all refit) | BR folder | paper (λ = 0.05) |", "|---|--:|--:|--:|"]
    for m in TABLE_ORDER:
        L.append(f"| {label(m)} | {o['dominant_counts_new'].get(m, 0)} | "
                 f"{o['dominant_counts_br_folder'].get(m, 0)} | {o['dominant_counts_paper'].get(m, 0)} |")
    L += ["", "## Top-k profile (agg-LL)", "", "| k | agg-LL | combo-r |", "|--:|--:|--:|"]
    L += [f"| {k} | {v['agg_ll']:.4f} | {v['combo_r']:.3f} |" for k, v in o["top_k_profile"].items()]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")


def write_tex(o, label):
    keys = ["combo_r", "marg_r", "agg_ll", "mean_ll"]
    allm = o["metrics_by_model"]
    best = {k: max(v[k] for v in allm.values()) for k in keys}
    fm = lambda v: f"{v:.3f}".replace("-", "$-$")
    def fmt(k, v):
        s = fm(v)
        if abs(v - best[k]) < 5e-4:
            s = r"$\mathbf{" + f"{v:.3f}" + "}$" if v < 0 else r"\textbf{" + s + "}"
        return s
    tex_name = lambda m: (f"Top-$k$ ($k={o['params_by_model']['Top-7']['k']}$)" if m == "Top-7"
                          else label(m))
    L = [f"% main-text aggregate table, all models fit and scored with lambda = {o['lapse']} lapse, no floor"]
    for i, m in enumerate(TABLE_ORDER):
        if i in (3, 6):
            L.append(r"\midrule" if i == 6 else "% ---")
        L.append(f"{tex_name(m):<22s}& " + " & ".join(fmt(k, allm[m][k]) for k in keys) + r" \\")
    L += ["", "% SI fitted-params table rows (model & objective & params & 4 metrics)"]
    for m in TABLE_ORDER:
        if m not in o["fits"]:
            continue
        for i, obj in enumerate(OBJECTIVES):
            e = o["per_objective_eval"][m][obj]
            p = {k: v for k, v in o["fits"][m][obj].items()
                 if k not in ("objective_value", "delta_plateau")}
            lab = {"combo_r": "combo-$r$", "agg_ll": "agg-LL", "mean_ll": "mean-LL"}[obj]
            L.append((tex_name(m) if i == 0 else "") + f" & {lab} & {pstr(m, p, tex=True)} & "
                     f"{e['combo_r']:.3f} & {e['marg_r']:.3f} & {fm(e['agg_ll'])} & {fm(e['mean_ll'])} \\\\")
    (OUT / "tables_snippets.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
