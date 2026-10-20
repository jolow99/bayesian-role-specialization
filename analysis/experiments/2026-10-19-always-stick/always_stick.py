"""2026-10-19 — Always-Stick baseline + regenerate model-comparison outputs.

Why: the pure Threshold-PS refit (2026-10-15) fits delta on a plateau that
runs to delta = 1.0, where the gate can never open — i.e. it IS an
always-stick model after stage 1 (predictions identical at delta = 0.435 and
1.0 on all 721 stage predictions). Always-Stick makes that explicit:

    stage 1 : uniform over the 27 combos (same as Random Walk)
    stage>1 : all mass on the previous combo (parameter-free, = Random Walk eps -> 0)

Added as a 14th model (7 Bayesian-family rows unchanged + 8 baselines);
everything else is identical to 2026-10-15 (asserted against its results.json).

Run from analysis/:
    uv run python experiments/2026-10-19-always-stick/always_stick.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXP = SCRIPT_DIR.parent
sys.path.insert(0, str(EXP / "2026-10-15-pure-threshold-ps"))
import pure_threshold_ps as pt  # noqa: E402  (puts pf / pipeline on path)
from pure_threshold_ps import pf, posteriors_from_ll  # noqa: E402
from pipeline import precompute_trajectories, strategy_from_params  # noqa: E402

OUT = SCRIPT_DIR
FIG_DIR = OUT / "stuff to incorporate"
FIG_DIR.mkdir(exist_ok=True)
pt.FIG_DIR = FIG_DIR

NAME = "Always Stick"


def predict_always_stick(record, stage_idx, prev_combo, env_config):
    if prev_combo is None:
        return {c: 1.0 / 27 for c in pf.ALL_ROLE_COMBOS}
    return {c: float(c == prev_combo) for c in pf.ALL_ROLE_COMBOS}


# extend R4's model list / legend / palette
pt.MODEL_ORDER = pt.MODEL_ORDER + [NAME]
pt.BASE = pt.BASE + [NAME]
pt.PALETTE[NAME] = "#525252"
MODEL_ORDER = pt.MODEL_ORDER
dn = pt.dn


def main():
    prev = json.load(open(EXP / "2026-10-15-pure-threshold-ps" / "results.json"))
    delta = prev["fits"]["agg_ll"]["delta"]

    s1 = json.load(open(pf.STAGE1_PARAMS))
    strat = strategy_from_params(s1["memory_strategy"], s1.get("window"), s1.get("drift_delta", 0.0))
    cells = json.load(open(pf.PIPELINE_RESULTS))["cells"]
    records = pf.load_records_with_pids(verbose=False)
    trajs = precompute_trajectories(records, s1["tau_prior"], s1["epsilon"], memory_strategy=strat)

    fns, params = pf.build_predict_fns(records, trajs, cells)
    fns.pop("Mixture-PS"); params.pop("Mixture-PS")
    fns["Bayesian Thresh-PS"] = pt.thresh_ps_pure_factory(delta)(records, trajs)
    params["Bayesian Thresh-PS"] = {"delta": delta}
    fns[NAME] = pf._wrap_baseline(predict_always_stick)
    params[NAME] = {}
    fns = {m: fns[m] for m in MODEL_ORDER}
    metrics = pf.evaluate_all(records, fns)

    for m in prev["metrics_by_model"]:
        for k in ("combo_r", "marg_r", "agg_ll", "mean_ll"):
            assert abs(metrics[m][k] - prev["metrics_by_model"][m][k]) < 1e-9, (m, k)
    print("13 existing models reproduce 2026-10-15 metrics exactly")

    # how much of Always-Stick's score is the stage-1 prior? (Thresh-PS delta=1 uses the Bayesian prior there)
    margs = pf.precompute_player_marginals(records, fns)
    _, player_ll, _ = pf.individual_posteriors(records, margs)
    lls = {m: dict(d) for m, d in player_ll.items()}
    post = posteriors_from_ll(lls, MODEL_ORDER)
    dom = pt.plot_r4(post)
    counts = Counter(dom.values())
    old_counts = prev["dominant_counts_new"]
    moved = Counter((max(prev["individual_ll"], key=lambda m: prev["individual_ll"][m][p]), dom[p])
                    for p in post
                    if max(MODEL_ORDER[:-1], key=lambda m: prev["individual_ll"][m][p]) != dom[p])
    # simpler/explicit reassignment table from 10-15 dominant model (posterior argmax over 13)
    old_post = posteriors_from_ll(prev["individual_ll"], MODEL_ORDER[:-1])
    old_dom = {p: max(MODEL_ORDER[:-1], key=lambda m: old_post[p][m]) for p in old_post}
    moved = Counter((old_dom[p], dom[p]) for p in post if old_dom[p] != dom[p])

    out = {"delta_thresh_ps": delta, "metrics_by_model": metrics, "params_by_model": params,
           "dominant_counts_new": dict(counts), "dominant_counts_1015": old_counts,
           "reassigned": {f"{a} -> {b}": n for (a, b), n in moved.items()},
           "individual_ll": lls, "n_participants": len(post), "lapse": pf.LAPSE}
    json.dump(out, open(OUT / "results.json", "w"), indent=2)
    write_md(out)
    write_tex(out)
    print("\nDominant counts (new vs 10-15):")
    for m in MODEL_ORDER:
        print(f"  {dn(m):24s} {counts.get(m, 0):3d}  (was {old_counts.get(m, 0)})")
    print("reassigned:", out["reassigned"])


def write_md(o):
    L = [f"# Always-Stick baseline added (14 models)", "",
         "## Aggregate table (agg_ll-fit params; Threshold-PS delta = "
         f"{o['delta_thresh_ps']:.3f})", "",
         "| Model | combo-r | marg-r | agg-LL | mean-LL |", "|---|--:|--:|--:|--:|"]
    for m in MODEL_ORDER:
        x = o["metrics_by_model"][m]
        L.append(f"| {dn(m)} | {x['combo_r']:.3f} | {x['marg_r']:.3f} | {x['agg_ll']:.3f} | {x['mean_ll']:.3f} |")
    L += ["", "## Dominant model per participant (n=102)", "", "| Model | with Always-Stick | 10-15 (13-model) |",
          "|---|--:|--:|"]
    for m in MODEL_ORDER:
        L.append(f"| {dn(m)} | {o['dominant_counts_new'].get(m, 0)} | {o['dominant_counts_1015'].get(m, 0)} |")
    L += ["", "Reassigned participants: " + (", ".join(f"{k}: {v}" for k, v in o["reassigned"].items()) or "none"), ""]
    (OUT / "summary.md").write_text("\n".join(L))


def write_tex(o):
    keys = ["combo_r", "marg_r", "agg_ll", "mean_ll"]
    allm = o["metrics_by_model"]
    best = {k: max(v[k] for v in allm.values()) for k in keys}
    def fmt(k, v):
        s = f"{v:.3f}".replace("-", "$-$") if v < 0 else f"{v:.3f}"
        if abs(v - best[k]) < 5e-4:
            s = r"$\mathbf{" + f"{v:.3f}" + "}$" if v < 0 else r"\textbf{" + s + "}"
        return s
    L = ["% aggregate table rows incl. Always Stick (bold = best in column)"]
    L += [f"{dn(m):<22s}& " + " & ".join(fmt(k, allm[m][k]) for k in keys) + r" \\" for m in MODEL_ORDER]
    (OUT / "tables_snippets.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
