"""2026-10-20 — tolerant Threshold-BR + lapse (λ = 0.01, no floor) refit.

Companion to 2026-10-20-tolerant-threshold-ps. The old Threshold-BR never
included the current role in its candidate set, so it forced a switch
whenever any role's expected value beat the current one by δ and gave 0 to
staying and to every other switch. Same fix as Threshold-PS (Xuan's option 2):

    allowed(δ) = {prev} ∪ {r ≠ prev : V(r) − V(prev) > δ}
    pi(r)      ∝ exp(V(r) / τ_v)  over allowed(δ)

δ is in units of expected team value (V ranges 0–90 here, gaps −68 to +61),
so δ ∈ [−70, 70]; δ = −70 lets every role through → exactly Bayesian-BR.

Scoring is identical to the PS folder (imported from it): λ = 0.01 lapse with
no floor for agg-LL / mean-LL and the individual fit, raw predictions for
combo-r / marg-r. The final 13-model table uses the tolerant Threshold-PS at
its agg-LL fit from that folder; the other 11 models keep their paper params.

(τ_v, δ) fit: coarse grid (δ step 1 × 20 log-spaced τ), then δ step 0.25
within ±2 of each objective's best with τ optimised (bounded) at every δ.
The objective is piecewise constant in δ, so L-BFGS-B is not used for it.

Run from analysis/ (needs the PS folder's results.json):
    uv run python experiments/2026-10-20-tolerant-threshold-br/tolerant_threshold_br.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

SCRIPT_DIR = Path(__file__).resolve().parent
EXP = SCRIPT_DIR.parent
PS_DIR = EXP / "2026-10-20-tolerant-threshold-ps"
sys.path.insert(0, str(EXP / "2026-05-28-paper-figures"))
sys.path.insert(0, str(EXP / "2026-05-25-full-pipeline"))
sys.path.insert(0, str(PS_DIR))

import paper_figures as pf  # noqa: E402
import tolerant_threshold_ps as tps  # noqa: E402  (lapse metrics, R4 plot, PS model)
from models import _expected_values_per_role  # noqa: E402
from pipeline import build_joint_dist, precompute_trajectories, strategy_from_params  # noqa: E402

OUT = SCRIPT_DIR
FIG_DIR = OUT / "stuff to incorporate"
FIG_DIR.mkdir(exist_ok=True)
tps.FIG_DIR = FIG_DIR  # plot_r4 writes to its module's FIG_DIR
OBJECTIVES = tps.OBJECTIVES
LAPSE = tps.LAPSE
DMIN, DMAX = -70.0, 70.0
TAU_LO, TAU_HI = 0.01, 50.0
dn, MODEL_ORDER, ALL_COMBOS, ROLE_IDX = tps.dn, tps.MODEL_ORDER, tps.ALL_COMBOS, tps.ROLE_IDX

# ── tolerant Threshold-BR ────────────────────────────────────────────

def _softmax(ev, tau):
    z = ev / tau
    p = np.exp(z - z.max())
    return p / p.sum()


def _tolerant_br_dist(ev, prev, delta, tau):
    allowed = [prev] + [r for r in range(3) if r != prev and (ev[r] - ev[prev]) > delta]
    d = np.zeros(3)
    d[allowed] = _softmax(ev[allowed], tau)
    return d


def precompute_evs(records, trajs):
    """evs[ri][s] = (3 agents × 3 roles) expected values; independent of τ, δ."""
    return [[np.array([_expected_values_per_role(i, st["intent"], st["thp"], st["ehp"],
                                                 st["prior"], rec["env_config"]["values"])
                       for i in range(3)]) for st in traj]
            for rec, traj in zip(records, trajs)]


def _predict(trajectory, evs, tau, delta):
    out = []
    for st, ev in zip(trajectory, evs):
        prev = st["prev_roles"]
        per_agent = [_softmax(ev[i], tau) if prev is None
                     else _tolerant_br_dist(ev[i], prev[i], delta, tau) for i in range(3)]
        out.append({"predicted_dist": build_joint_dist(per_agent),
                    "human_combo": st["human_combo"],
                    "model_marginal": np.mean(per_agent, axis=0)})
    return out


def tolerant_br_factory(tau, delta, evs_all):
    def factory(records_subset, traj_subset):
        idx = {id(r): i for i, r in enumerate(records_subset)}
        return lambda rec: _predict(traj_subset[idx[id(rec)]], evs_all[idx[id(rec)]], tau, delta)
    return factory

# ── fitting ──────────────────────────────────────────────────────────

def fit_tau_delta(records, trajs, evs):
    cache = {}
    def at(tau, delta):
        key = (round(float(tau), 6), round(float(delta), 4))
        if key not in cache:
            cache[key] = tps.lapse_metrics(records, tolerant_br_factory(*key, evs)(records, trajs))
        return cache[key]

    taus = np.geomspace(0.5, 50.0, 20)
    deltas = np.round(np.arange(DMIN, DMAX + 1e-9, 1.0), 4)
    coarse = {(t, d): at(t, d) for d in deltas for t in taus}
    print(f"coarse grid done ({len(coarse)} points)")

    fits = {}
    for obj in OBJECTIVES:
        (t0, d0), _ = max(coarse.items(), key=lambda kv: kv[1][obj])
        rows = []
        for d in np.round(np.arange(max(DMIN, d0 - 2), min(DMAX, d0 + 2) + 1e-9, 0.25), 4):
            res = minimize_scalar(lambda t: -at(t, d)[obj], bounds=(TAU_LO, TAU_HI),
                                  method="bounded", options={"xatol": 1e-3})
            rows.append((float(d), float(res.x), -float(res.fun)))
        rows.append((float(d0), float(t0), coarse[(t0, d0)][obj]))
        best = max(r[2] for r in rows)
        top = sorted(r for r in rows if r[2] >= best - 1e-6)
        d_best, t_best, _ = top[0]
        prof_tau = t_best
        fits[obj] = {"delta": d_best, "tau_v": t_best, "objective_value": float(best),
                     "delta_plateau": [top[0][0], top[-1][0]],
                     "profile_at_tau": prof_tau,
                     "profile": {f"{d:.0f}": at(prof_tau, d)[obj] for d in deltas[::10]}}
        print(f"tolerant Thresh-BR fit on {obj:8s}: tau={t_best:.3f} delta={d_best:.2f} "
              f"(delta plateau {fits[obj]['delta_plateau']}) value={best:.4f}")
    return fits


def switch_gaps(records, trajs, evs):
    """V(chosen) − V(prev) for every observed switch."""
    gaps = []
    for traj, ev_r in zip(trajs, evs):
        for st, ev in zip(traj, ev_r):
            prev = st["prev_roles"]
            if prev is None:
                continue
            for i in range(3):
                a = ROLE_IDX[st["human_combo"][i]]
                if a != prev[i]:
                    gaps.append(float(ev[i][a] - ev[i][prev[i]]))
    g = np.array(gaps)
    return {"n_switches": int(g.size), "min": float(g.min()), "max": float(g.max()),
            "pct": {str(q): float(np.percentile(g, q)) for q in (1, 5, 25, 50, 75, 95)},
            "frac_below_0": float((g <= 0).mean())}

# ── main ─────────────────────────────────────────────────────────────

def main():
    s1 = json.load(open(pf.STAGE1_PARAMS))
    strat = strategy_from_params(s1["memory_strategy"], s1.get("window"), s1.get("drift_delta", 0.0))
    cells = json.load(open(pf.PIPELINE_RESULTS))["cells"]
    old = json.load(open(pf.RESULTS_PATH))
    paper = json.load(open(EXP / "2026-10-15-pure-threshold-ps" / "results.json"))
    ps_run = json.load(open(PS_DIR / "results.json"))
    records = pf.load_records_with_pids(verbose=False)
    trajs = precompute_trajectories(records, s1["tau_prior"], s1["epsilon"], memory_strategy=strat)
    evs = precompute_evs(records, trajs)

    # 0. sanity: untouched models reproduce the paper (floored) numbers
    fns, params = pf.build_predict_fns(records, trajs, cells)
    for m in ("Mixture-PS", "Bayesian Thresh-PS", "Bayesian Threshold"):
        fns.pop(m); params.pop(m)
    floored = pf.evaluate_all(records, fns)
    for m in fns:
        for k in ("combo_r", "marg_r", "agg_ll", "mean_ll"):
            assert abs(floored[m][k] - old["aggregate"]["metrics_by_model"][m][k]) < 1e-9, (m, k)
    print("untouched models reproduce the paper's (floored) metrics exactly")

    # 1. δ = −70 must be exactly Bayesian-BR (at BR's fitted τ)
    tau_br = params["Bayesian-Value"]["tau_softmax"]
    assert DMIN < min(float((ev[i] - ev[i][j]).min()) for ev_r in evs for ev in ev_r
                      for i in range(3) for j in range(3))
    tol = tolerant_br_factory(tau_br, DMIN, evs)(records, trajs)
    for rec in records:
        for a, b in zip(tol(rec), fns["Bayesian-Value"](rec)):
            assert all(abs(a["predicted_dist"][c] - b["predicted_dist"].get(c, 0.0)) < 1e-12
                       for c in ALL_COMBOS)
    print(f"tolerant Threshold-BR at delta={DMIN:.0f} reproduces Bayesian-BR exactly")

    # 2. fit (τ_v, δ) under each objective
    fits = fit_tau_delta(records, trajs, evs)
    per_obj_eval = {obj: tps.lapse_metrics(records, tolerant_br_factory(
        fits[obj]["tau_v"], fits[obj]["delta"], evs)(records, trajs)) for obj in OBJECTIVES}
    gaps = switch_gaps(records, trajs, evs)

    # 3. all 13 models under the lapse metrics: both tolerant threshold models
    d_ps = ps_run["fits"]["agg_ll"]["delta"]
    fns["Bayesian Thresh-PS"] = tps.tolerant_factory(d_ps)(records, trajs)
    params["Bayesian Thresh-PS"] = {"delta": d_ps}
    fa = fits["agg_ll"]
    fns["Bayesian Threshold"] = tolerant_br_factory(fa["tau_v"], fa["delta"], evs)(records, trajs)
    params["Bayesian Threshold"] = {"tau_softmax": fa["tau_v"], "delta": fa["delta"]}
    fns = {m: fns[m] for m in MODEL_ORDER}
    metrics = {m: tps.lapse_metrics(records, fn) for m, fn in fns.items()}
    assert metrics["Bayesian Threshold"]["agg_ll"] >= metrics["Bayesian-Value"]["agg_ll"] - 1e-9
    assert abs(metrics["Bayesian Thresh-PS"]["agg_ll"]
               - ps_run["metrics_by_model"]["Bayesian Thresh-PS"]["agg_ll"]) < 1e-9

    # 4. individual fit (pooled params, λ lapse on marginals, no floor)
    margs = pf.precompute_player_marginals(records, fns)
    lls = tps.individual_ll(records, margs)
    post = tps.posteriors_from_ll(lls, MODEL_ORDER)
    dom = tps.plot_r4(post)
    counts = Counter(dom.values())
    paper_post = tps.posteriors_from_ll(paper["individual_ll"], MODEL_ORDER)
    paper_counts = Counter(max(MODEL_ORDER, key=lambda m: v[m]) for v in paper_post.values())

    # 5. write outputs
    out = {"lapse": LAPSE, "fits": fits, "per_objective_eval": per_obj_eval,
           "switch_gaps": gaps, "metrics_by_model": metrics,
           "ps_folder_metrics_by_model": ps_run["metrics_by_model"],
           "paper_metrics_by_model": paper["metrics_by_model"],
           "params_by_model": params, "dominant_counts_new": dict(counts),
           "dominant_counts_ps_folder": ps_run["dominant_counts_new"],
           "dominant_counts_paper": dict(paper_counts),
           "individual_ll": lls, "n_participants": len(post)}
    json.dump(out, open(OUT / "results.json", "w"), indent=2)
    write_md(out)
    write_tex(out)
    print("\nDominant counts (new vs PS folder vs paper):")
    for m in MODEL_ORDER:
        print(f"  {dn(m):24s} {counts.get(m,0):3d}  (PS folder {out['dominant_counts_ps_folder'].get(m,0)},"
              f" paper {paper_counts.get(m,0)})")


def write_md(o):
    g = o["switch_gaps"]
    L = [f"# Tolerant Threshold-BR refit (λ = {o['lapse']}, no floor)", "",
         "## Fitted (τ_v, δ)", "",
         "| Fit on | τ_v | δ | δ plateau (equal-best) | combo-r | marg-r | agg-LL | mean-LL |",
         "|---|--:|--:|---|--:|--:|--:|--:|"]
    for obj in OBJECTIVES:
        f, e = o["fits"][obj], o["per_objective_eval"][obj]
        L.append(f"| {obj} | {f['tau_v']:.2f} | {f['delta']:.2f} | {f['delta_plateau'][0]:.2f} to "
                 f"{f['delta_plateau'][1]:.2f} | {e['combo_r']:.3f} | {e['marg_r']:.3f} | "
                 f"{e['agg_ll']:.3f} | {e['mean_ll']:.3f} |")
    L += ["", f"δ = {DMIN:.0f} is exactly Bayesian-BR (asserted).", "",
          "## Observed switches: V(chosen) − V(previous role)", "",
          f"{g['n_switches']} switches; min {g['min']:.2f}, max {g['max']:.2f}; "
          f"{100 * g['frac_below_0']:.1f}% go to a role of no higher value than the current one.", "",
          "| percentile | " + " | ".join(g["pct"]) + " |",
          "|---|" + "--:|" * len(g["pct"]),
          "| gap | " + " | ".join(f"{v:.2f}" for v in g["pct"].values()) + " |", "",
          "## Aggregate table (both tolerant threshold models at their agg-LL fits; others at paper params)", "",
          f"agg-LL / mean-LL: λ = {o['lapse']} lapse, no floor. combo-r / marg-r: raw predictions. "
          "Paper columns = current paper (1e-12 / 1e-20 floors).", "",
          "| Model | combo-r | marg-r | agg-LL | mean-LL | agg-LL (paper) | mean-LL (paper) |",
          "|---|--:|--:|--:|--:|--:|--:|"]
    for m in MODEL_ORDER:
        x, p = o["metrics_by_model"][m], o["paper_metrics_by_model"][m]
        L.append(f"| {dn(m)} | {x['combo_r']:.3f} | {x['marg_r']:.3f} | {x['agg_ll']:.3f} | "
                 f"{x['mean_ll']:.3f} | {p['agg_ll']:.3f} | {p['mean_ll']:.3f} |")
    L += ["", f"## Dominant model per participant (n={o['n_participants']})", "",
          "| Model | new (both tolerant) | PS folder (old Thresh-BR) | paper (λ = 0.05) |",
          "|---|--:|--:|--:|"]
    for m in MODEL_ORDER:
        L.append(f"| {dn(m)} | {o['dominant_counts_new'].get(m,0)} | "
                 f"{o['dominant_counts_ps_folder'].get(m,0)} | {o['dominant_counts_paper'].get(m,0)} |")
    fa = o["fits"]["agg_ll"]
    L += ["", f"## δ profile (agg-LL) at τ_v = {fa['profile_at_tau']:.2f}, every 10", "",
          "| δ | agg-LL |", "|--:|--:|"]
    L += [f"| {d} | {v:.4f} |" for d, v in fa["profile"].items()]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")


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
    L = [f"% main-text Table (aggregate), lambda = {o['lapse']} lapse, no floor, both tolerant "
         "threshold models: 6 Bayesian rows", *rows, ""]
    L += ["% SI fitted-params rows for Bayesian-Threshold-BR (tolerant version)"]
    fm = lambda v: f"{v:.3f}".replace("-", "$-$")
    for i, obj in enumerate(OBJECTIVES):
        f, e = o["fits"][obj], o["per_objective_eval"][obj]
        lab = {"combo_r": "combo-$r$", "agg_ll": "agg-LL", "mean_ll": "mean-LL"}[obj]
        d = f"{f['delta']:.2f}".replace("-", "$-$")
        L.append(("Bayesian-Threshold-BR" if i == 0 else "") + f" & {lab} & "
                 f"$\\tau_v{{=}}{f['tau_v']:.2f}$, $\\delta_{{\\mathrm{{BR}}}}{{=}}${d} & "
                 f"{e['combo_r']:.3f} & {e['marg_r']:.3f} & {fm(e['agg_ll'])} & {fm(e['mean_ll'])} \\\\")
    (OUT / "tables_snippets.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
