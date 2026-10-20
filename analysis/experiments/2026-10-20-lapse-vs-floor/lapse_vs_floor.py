"""2026-10-20 — three ways to treat zero-probability predictions.

Xuan (2026-10-19/20): aggregate table floors predictions (1e-12 agg-LL, 1e-20
mean-LL) while the individual fits mix in a 5% uniform lapse. Compare ONE rule
applied to BOTH analyses:

  none  : no lapse, no floor. A model with p=0 on an observed choice gets -inf.
  lapse : p' = (1-L)p + L/K, L = 0.05. K = 27 combos (aggregate), 3 roles (individual).
  floor : p' = max(p, 1e-12) at the scoring step, same value in both analyses.

All 14 models (13 from 2026-10-15 + Always Stick). Other models' parameters are
the existing pooled fits (not refit). Threshold-PS delta is refit per variant on
agg-LL where that is defined (lapse: refit; floor: 0.435 as in 10-15; none: agg-LL is
-inf at every delta, so the combo-r fit 0.255 is shown).

Run from analysis/:
    uv run python experiments/2026-10-20-lapse-vs-floor/lapse_vs_floor.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXP = SCRIPT_DIR.parent
sys.path.insert(0, str(EXP / "2026-10-15-pure-threshold-ps"))
sys.path.insert(0, str(EXP / "2026-10-19-always-stick"))
import pure_threshold_ps as pt  # noqa: E402
import always_stick as ak  # noqa: E402
from pure_threshold_ps import pf, posteriors_from_ll  # noqa: E402
from pipeline import precompute_trajectories, strategy_from_params  # noqa: E402
from shared.evaluation import compute_pearson, run_predictions  # noqa: E402

OUT = SCRIPT_DIR
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
pt.FIG_DIR = FIG
MODEL_ORDER = pt.MODEL_ORDER  # already includes Always Stick (set by always_stick import)
dn = pt.dn
LAMBDA, FLOOR = 0.05, 1e-12
VARIANTS = {"none": "No lapse, no floor", "lapse": f"Lapse (λ = {LAMBDA})", "floor": f"Floor ({FLOOR:g})"}


# ── lapse wrapper on predict_fns ─────────────────────────────────────
def with_lapse(fn):
    def g(rec):
        out = []
        for p in fn(rec):
            d = {c: (1 - LAMBDA) * v + LAMBDA / 27.0 for c, v in p["predicted_dist"].items()}
            out.append({**p, "predicted_dist": d,
                        "model_marginal": (1 - LAMBDA) * np.asarray(p["model_marginal"]) + LAMBDA / 3.0})
        return out
    return g


# ── scoring with explicit floor (0 → true -inf) ──────────────────────
def _log(p, floor):
    with np.errstate(divide="ignore"):
        return np.log(max(p, floor)) if floor > 0 else (np.log(p) if p > 0 else -np.inf)


def mean_ll(full, floor):
    per_env = []
    for d in full.values():
        ls = [_log(pred["predicted_dist"].get(pred["human_combo"], 0.0), floor)
              for team in d["team_predictions"] for pred in team]
        if ls:
            per_env.append(np.mean(ls))
    return float(np.mean(per_env))


def agg_ll(full, floor):
    tot_ce = tot_w = 0.0
    for d in full.values():
        for s in range(d["max_stages"]):
            pred = d["stage_predicted"].get(s)
            hc = d["stage_human"].get(s, {})
            n = d["stage_counts"].get(s, 0)
            if pred is None or n == 0:
                continue
            for cc in d["canonical_combos"]:
                ph = hc.get(cc, 0) / n
                if ph <= 0:
                    continue
                tot_ce += n * ph * _log(pred.get(cc, 0.0), floor)
            tot_w += n
    return float(tot_ce / tot_w)


def fit_delta_lapse(records, trajs):
    best = (-np.inf, None)
    prof = {}
    for d in np.round(np.arange(0, 1.0001, 0.01), 2):
        fn = with_lapse(pt.thresh_ps_pure_factory(float(d))(records, trajs))
        v = agg_ll(run_predictions(records, fn), 0.0)
        prof[float(d)] = v
        if v > best[0] + 1e-12:
            best = (v, float(d))
    plateau = [d for d, v in prof.items() if v >= best[0] - 1e-9]
    return best[1], best[0], (min(plateau), max(plateau)), prof


# ── individual fit under a variant ───────────────────────────────────
def individual_post(records, margs, variant):
    names = list(margs)
    ll = {n: defaultdict(float) for n in names}
    pids = set()
    for ri, rec in enumerate(records):
        for s, hc in enumerate(rec["stage_roles"]):
            for pos in range(3):
                pid = rec["participant_ids"][pos]
                if pid is None:
                    continue
                pids.add(pid)
                a = pf.ROLE_CHAR_TO_IDX[hc[pos]]
                for n in names:
                    st = margs[n][ri]
                    m = st[s][pos] if s < len(st) else np.ones(3) / 3
                    if variant == "lapse":
                        p = (1 - LAMBDA) * m[a] + LAMBDA / 3
                    else:
                        p = m[a]
                    ll[n][pid] += _log(p, FLOOR if variant == "floor" else 0.0)
    post = {}
    for pid in pids:
        v = np.array([ll[n][pid] for n in names])
        with np.errstate(invalid="ignore"):
            e = np.exp(v - v.max())
        post[pid] = dict(zip(names, e / e.sum()))
    return post, {n: dict(d) for n, d in ll.items()}


# ── figures ──────────────────────────────────────────────────────────
def fmt(v, k):
    return "−∞" if not np.isfinite(v) else f"{v:.3f}".replace("-", "−")


def table_fig(metrics, deltas, variant, ax):
    ax.axis("off")
    rows = [[dn(m) + (f"  (δ={deltas[variant]:.2f})" if m == "Bayesian Thresh-PS" else "")] +
            [fmt(metrics[m][k], k) for k in ("combo_r", "marg_r", "agg_ll", "mean_ll")] for m in MODEL_ORDER]
    t = ax.table(cellText=rows, colLabels=["Model", "combo-r", "marg-r", "agg-LL", "mean-LL"],
                 cellLoc="right", colLoc="right", loc="center", colWidths=[0.46, 0.13, 0.13, 0.14, 0.14])
    t.auto_set_font_size(False); t.set_fontsize(7.5); t.scale(1, 1.25)
    for (r, c), cell in t.get_celld().items():
        cell.set_linewidth(0.3)
        if c == 0:
            cell.set_text_props(ha="left"); cell._loc = "left"
        if r > 0 and c >= 3 and cell.get_text().get_text() == "−∞":
            cell.set_text_props(color="#b2182b")
    ax.set_title(VARIANTS[variant], fontsize=10, fontweight="bold")


def main():
    s1 = json.load(open(pf.STAGE1_PARAMS))
    strat = strategy_from_params(s1["memory_strategy"], s1.get("window"), s1.get("drift_delta", 0.0))
    cells = json.load(open(pf.PIPELINE_RESULTS))["cells"]
    records = pf.load_records_with_pids(verbose=False)
    trajs = precompute_trajectories(records, s1["tau_prior"], s1["epsilon"], memory_strategy=strat)
    prev = json.load(open(EXP / "2026-10-15-pure-threshold-ps" / "results.json"))

    base, _ = pf.build_predict_fns(records, trajs, cells)
    base.pop("Mixture-PS")
    base[ak.NAME] = pf._wrap_baseline(ak.predict_always_stick)

    d_lapse, v_lapse, plat, prof = fit_delta_lapse(records, trajs)
    deltas = {"none": prev["fits"]["combo_r"]["delta"], "lapse": d_lapse, "floor": prev["fits"]["agg_ll"]["delta"]}
    print(f"Threshold-PS delta: none(combo-r)={deltas['none']}  lapse(agg-LL)={d_lapse} plateau={plat} "
          f"value={v_lapse:.4f}  floor={deltas['floor']}")

    def fns_for(variant):
        f = dict(base)
        f["Bayesian Thresh-PS"] = pt.thresh_ps_pure_factory(deltas[variant])(records, trajs)
        f = {m: f[m] for m in MODEL_ORDER}
        return {m: (with_lapse(fn) if variant == "lapse" else fn) for m, fn in f.items()}

    results = {}
    fig_t, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, v in zip(axes, VARIANTS):
        fns = fns_for(v)
        metrics = {}
        for m, fn in fns.items():
            full = run_predictions(records, fn)
            g = compute_pearson(full).get("__global__", {})
            fl = FLOOR if v == "floor" else 0.0
            metrics[m] = {"combo_r": g["combo"]["r"], "marg_r": g["marginal"]["r"],
                          "agg_ll": agg_ll(full, fl), "mean_ll": mean_ll(full, fl)}
        margs = pf.precompute_player_marginals(records, fns)
        # marginals from the lapse-wrapped joint already include the lapse; score those WITHOUT a second lapse
        post, ll = individual_post(records, margs, "none" if v == "lapse" else v)
        dom = {p: max(MODEL_ORDER, key=lambda m: pp[m]) for p, pp in post.items()}
        results[v] = {"metrics": metrics, "delta": deltas[v], "dominant_counts": dict(Counter(dom.values())),
                      "n_participants": len(post)}
        # per-variant R4 (uses pt.plot_r4 with this variant's posteriors)
        pt.plot_r4(post, name=f"R4_individual_{v}")
        # single-variant table
        f1, a1 = plt.subplots(figsize=(6, 5.2)); table_fig(metrics, deltas, v, a1)
        f1.savefig(FIG / f"aggregate_table_{v}.png", dpi=250, bbox_inches="tight", facecolor="white"); plt.close(f1)
        table_fig(metrics, deltas, v, ax)
        print(v, "dominant:", {dn(m): c for m, c in sorted(Counter(dom.values()).items(), key=lambda x: -x[1])})
    fig_t.tight_layout()
    fig_t.savefig(FIG / "aggregate_tables_side_by_side.png", dpi=220, facecolor="white")
    fig_t.savefig(FIG / "aggregate_tables_side_by_side.pdf", facecolor="white")

    fig_i, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, v in zip(axes, VARIANTS):
        ax.imshow(mpimg.imread(FIG / f"R4_individual_{v}.png")); ax.axis("off"); ax.set_title(VARIANTS[v], fontweight="bold")
    fig_i.tight_layout()
    fig_i.savefig(FIG / "individual_fits_side_by_side.png", dpi=220, facecolor="white")

    results["delta_profile_lapse"] = prof
    json.dump(results, open(OUT / "results.json", "w"), indent=2, default=float)

    L = ["# Lapse vs floor vs none", "", f"Threshold-PS delta: none={deltas['none']:.3f} (combo-r fit; agg-LL undefined), "
         f"lapse={deltas['lapse']:.3f} (plateau {plat[0]:.2f}–{plat[1]:.2f}), floor={deltas['floor']:.3f}", ""]
    for v in VARIANTS:
        L += [f"## {VARIANTS[v]}", "", "| Model | combo-r | marg-r | agg-LL | mean-LL | # dominant |", "|---|--:|--:|--:|--:|--:|"]
        for m in MODEL_ORDER:
            x = results[v]["metrics"][m]
            L.append(f"| {dn(m)} | {fmt(x['combo_r'],0)} | {fmt(x['marg_r'],0)} | {fmt(x['agg_ll'],0)} | {fmt(x['mean_ll'],0)} | {results[v]['dominant_counts'].get(m, 0)} |")
        L.append("")
    (OUT / "summary.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
