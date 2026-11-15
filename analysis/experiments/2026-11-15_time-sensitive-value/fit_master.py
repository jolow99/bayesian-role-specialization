"""Master table: every combination of the three 2026-11-10 changes, each refit.

Axes
  value   paper V(HORIZON) at every stage  |  time-aware V(HORIZON - 2·stage)
  belief  mixed (1-β)P_N + βP_0 (paper)    |  unmixed P_N
  group / belief source
    U            50 updaters on their own stat-prior chain (refit)
    NU-FTM       52 non-updaters on the paper's fixed FTM prior (refit)
    NU-as-U      52 non-updaters on the updaters' chain (refit)
    NU-as-U-nr   same beliefs AND the matching updater variant's fitted params, nothing refit
All Bayesian response models are fit by agg-LL (2026-11-02 fit_bayes, unchanged), inference params fixed.
NU-FTM does not depend on mixed/unmixed (FTM prior is fixed), so it has one row-set per value function.
PS-type models never use V, so their fits are identical across the value axis (shown anyway).

combo-r conventions follow 2026-11-12 make_tables.py: U and NU-FTM keep the paper beliefs for the
other group's players (non-updaters on FTM); NU-as-U puts every player on the stat-prior chain.

Verification (asserted): with paper V every variant reproduces the stored 2026-11-12 tables
(U-mixed, U-unmixed, NU-paper, NU-mixed, NU-unmixed; NU-as-U-nr mixed vs nonupd-as-updaters).

Run from analysis/:
    uv run python experiments/2026-11-15_time-sensitive-value/fit_master.py [numpy|memo]
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
BASE_DIR = EXP / "2026-11-02_ftm-prior-fitting"
VAR_DIR = EXP / "2026-11-12_response-refit-variants"
sys.path.insert(0, str(BASE_DIR / "response-fitting"))
sys.path.insert(0, str(VAR_DIR))
sys.path.insert(0, str(HERE))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mt = _load("make_tables", VAR_DIR / "make_tables.py")
ts = _load("fit_time_sensitive", HERE / "fit_time_sensitive.py")

from beliefs import all_states  # noqa: E402
from models.data import Obs, load_records  # noqa: E402
from models.predict import BAYES  # noqa: E402

COLS = ("agg_ll", "mean_ll", "marg_r", "combo_r")


def close_fit(a, b, ptol=1e-4, mtol=1e-6):
    """Same fit up to float32 round-off in the value tables (~1e-5): params to ptol, metrics to mtol."""
    for m in b:
        pa, pb = a[m]["params"], b[m]["params"]
        if set(pa) != set(pb) or any(abs(pa[k] - pb[k]) > ptol for k in pa):
            return False
        if any(abs(a[m][k] - b[m][k]) > mtol for k in COLS):
            return False
        if any(abs(a[m]["participant_ll"][p] - b[m]["participant_ll"][p]) > 1e-4 for p in b[m]["participant_ll"]):
            return False
    return set(a) == set(b)


def main(source="numpy"):
    npz = HERE / ("value_by_turns.npz" if source == "numpy" else "value_by_turns_memo.npz")
    params = json.load(open(BASE_DIR / "inference-fitting" / "params.json"))
    with open(BASE_DIR / "inference-fitting" / "participant_types.csv") as f:
        p_nu = {r["participant_id"]: float(r["p_non_updater"]) for r in csv.DictReader(f)}
    records = load_records()
    ob = Obs(records, p_nu)
    nu = ob.non_updater
    none = np.zeros(ob.n, bool)
    states = all_states(records, params)
    vt = np.load(npz)
    n_p = {"U": len(np.unique(ob.pid[~nu])), "NU": len(np.unique(ob.pid[nu]))}
    print(f"value tables: {npz.name}")

    R = {}  # (group, value, belief) -> {model: result}
    for value in ("paper", "time-aware"):
        tsv = value == "time-aware"
        for belief in ("mixed", "unmixed"):
            U_b = ts.tbl_beliefs(records, ob, params, states, vt, belief, nu, tsv)
            NU_b = ts.tbl_beliefs(records, ob, params, states, vt, belief, none, tsv)
            print(f"[{value} / {belief}] U ...", flush=True)
            R[("U", value, belief)] = mt.fit_table(ob, ~nu, *U_b, records)
            print(f"[{value} / {belief}] NU-as-U ...", flush=True)
            R[("NU-as-U", value, belief)] = mt.fit_table(ob, nu, *NU_b, records)
            P_up = {m: R[("U", value, belief)][m]["params"] for m in BAYES}
            R[("NU-as-U-nr", value, belief)] = ts.score(ob, nu, *NU_b, records, P_up)
            if belief == "mixed":  # FTM prior is fixed: no mixed/unmixed distinction
                print(f"[{value}] NU-FTM ...", flush=True)
                R[("NU-FTM", value, "n/a")] = mt.fit_table(ob, nu, *U_b, records)
            if tsv:  # frozen: time-aware V scored at the PAPER-V fitted params (no refit)
                fz = "time-aware (frozen)"
                P = lambda g, b: {m: R[(g, "paper", b)][m]["params"] for m in BAYES}
                R[("U", fz, belief)] = ts.score(ob, ~nu, *U_b, records, P("U", belief))
                R[("NU-as-U", fz, belief)] = ts.score(ob, nu, *NU_b, records, P("NU-as-U", belief))
                R[("NU-as-U-nr", fz, belief)] = ts.score(ob, nu, *NU_b, records, P("U", belief))
                if belief == "mixed":
                    R[("NU-FTM", fz, "n/a")] = ts.score(ob, nu, *U_b, records, P("NU-FTM", "n/a"))

    # ---- verification against the stored 2026-11-12 tables (paper V) ----
    stored = VAR_DIR / "tables" / "fit-results"
    ld = lambda n: json.load(open(stored / f"{n}.json"))
    for key, name in ((("U", "paper", "mixed"), "U-mixed"), (("U", "paper", "unmixed"), "U-unmixed"),
                      (("NU-FTM", "paper", "n/a"), "NU-paper"), (("NU-as-U", "paper", "mixed"), "NU-mixed"),
                      (("NU-as-U", "paper", "unmixed"), "NU-unmixed")):
        assert close_fit(R[key], ld(name)), f"{key} does not reproduce {name}"
        print(f"[PASS] paper V {key[0]}/{key[2]} reproduces stored {name}")
    nr = json.load(open(VAR_DIR / "nonupd-as-updaters" / "bayes-models.json"))
    assert all(abs(R[("NU-as-U-nr", "paper", "mixed")][m][k] - nr[m][k]) < 1e-6 for m in BAYES for k in COLS)
    print("[PASS] paper V NU-as-U-nr/mixed reproduces nonupd-as-updaters")
    for g in ("U", "NU-as-U", "NU-as-U-nr", "NU-FTM"):  # PS models do not use V
        for belief in ("mixed", "unmixed", "n/a"):
            a, b = R.get((g, "paper", belief)), R.get((g, "time-aware", belief))
            if a and b and g != "NU-as-U-nr":
                for m in ("Bayesian-PS", "Bayesian-Walk-PS", "Bayesian-Threshold-PS"):
                    assert abs(a[m]["agg_ll"] - b[m]["agg_ll"]) < 1e-9, (g, belief, m)
    print("[PASS] PS-type fits identical across value functions (they don't use V)")

    # ---- outputs ----
    out = HERE / f"master_{source}"
    out.mkdir(exist_ok=True)
    flat = {"|".join(k): v for k, v in R.items()}
    json.dump(flat, open(out / "all_fits.json", "w"), indent=1)
    names = {"U": f"Updaters (n = {n_p['U']}), own beliefs, refit",
             "NU-FTM": f"Non-updaters (n = {n_p['NU']}), paper fixed FTM prior, refit",
             "NU-as-U": f"Non-updaters (n = {n_p['NU']}) on updater beliefs, refit",
             "NU-as-U-nr": f"Non-updaters (n = {n_p['NU']}) on updater beliefs AND updater params, no refit"}
    with open(out / "master.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "value_fn", "belief", "model", "params", *COLS, "sum_participant_ll"])
        for (g, v, b), res in R.items():
            for m in mt.ORDER:
                r = res[m]
                w.writerow([g, v, b, m, mt.fmt_params(r["params"]), *[f"{r[c]:.4f}" for c in COLS],
                            f"{sum(r['participant_ll'].values()):.1f}"])
    md = ["# Master table: value function × belief × group (every variant refit)", "",
          f"Value tables: `{npz.name}`. Inference params fixed (2026-11-02). Fit objective: agg-LL. Best agg-LL per block in bold. "
          "Value = paper V(10) at every stage vs time-aware V(10 − 2·stage) (refit); 'time-aware (frozen)' = time-aware V scored at "
          "the paper-V fitted params, nothing refit; belief = mixed (paper) vs unmixed P_N.", "",
          "## Summary: Bayesian-Walk-BR (winner) in every variant", "",
          "| Group | Value fn | Belief | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |",
          "|---|---|---|---|---|---|---|---|---|"]
    for (g, v, b), res in R.items():
        r = res["Bayesian-Walk-BR"]
        md.append(f"| {g} | {v} | {b} | {mt.fmt_params(r['params'])} | {r['agg_ll']:.4f} | {r['mean_ll']:.4f} | "
                  f"{r['marg_r']:.3f} | {r['combo_r']:.3f} | {sum(r['participant_ll'].values()):.1f} |")
    md += ["", "## Best model by agg-LL in every variant", "", "| Group | Value fn | Belief | Best model |", "|---|---|---|---|"]
    for (g, v, b), res in R.items():
        md.append(f"| {g} | {v} | {b} | {max(res, key=lambda m: res[m]['agg_ll'])} |")
    md += ["", "## Full tables", ""]
    for g in names:
        md += [f"### {names[g]}", ""]
        for (gg, v, b), res in R.items():
            if gg != g:
                continue
            best = max(res, key=lambda m: res[m]["agg_ll"])
            md += [f"**value = {v}, belief = {b}**", "",
                   "| Model | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |", "|---|---|---|---|---|---|---|"]
            for m in mt.ORDER:
                r = res[m]
                bf = (lambda x: f"**{x}**") if m == best else (lambda x: x)
                md.append(f"| {bf(m)} | {mt.fmt_params(r['params'])} | {bf('%.4f' % r['agg_ll'])} | {r['mean_ll']:.4f} | "
                          f"{r['marg_r']:.3f} | {r['combo_r']:.3f} | {sum(r['participant_ll'].values()):.1f} |")
            md.append("")
    (out / "master.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out}/master.md, master.csv, all_fits.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "numpy")
