"""Response-model refit with a time-sensitive value function (2026-11-10 meeting, Xuan).

The paper's value function V(HORIZON) assumes a full round of 10 turns remains at every stage. Here a
stage-s choice (s = 0..4, 2 turns per stage) uses V(HORIZON - 2 s) from value_by_turns.npz, i.e. the
value of the role assignment given the turns actually left.

Everything else is unchanged from 2026-11-12 make_tables.py: fixed inference params, same beliefs,
same agg-LL objective and fitting routine. Compared against the shipped-value fits:
  U-mixed   updaters, paper beliefs (mixed posterior; non-updater teammates on the FTM prior for combo-r)
  NU-mixed  non-updaters on updater (mixed) beliefs, refit
  NU-as-U   non-updaters on updater (mixed) beliefs AND updater parameters, nothing refit
            (the analysis Xuan chose)

Verification (asserted): with the time index forced to t = HORIZON the pipeline reproduces the
2026-11-12 U-mixed / NU-mixed fits exactly; the time-indexed table matches values.npy at t = HORIZON.

Run from analysis/:
    uv run python experiments/2026-11-15_time-sensitive-value/fit_time_sensitive.py
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

spec = importlib.util.spec_from_file_location("make_tables", VAR_DIR / "make_tables.py")
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)

from beliefs import all_states  # noqa: E402
from models.beliefs import expected_values, own_marginal  # noqa: E402
from models.data import Obs, load_records  # noqa: E402
from models.predict import BAYES, bayes_marginals  # noqa: E402
from models.scoring import combo_r, metrics, participant_lls  # noqa: E402

HORIZON, TURNS_PER_STAGE = 10, 2
LOG = []


def log(msg):
    print(msg)
    LOG.append(msg)


def tbl_beliefs(records, ob, params, states, vt, kind, ftm_rows, time_sensitive=True):
    """(PM, EV). Rows in ftm_rows use the FTM prior; others the stat-prior chain's `kind` belief.
    EV uses V(HORIZON - 2 s) if time_sensitive else V(HORIZON)."""
    p = np.array([params["p_F"], params["p_T"], params["p_M"]])
    ftm = np.einsum("i,j,k->ijk", p, p, p)
    PM, EV = np.empty((ob.n, 3)), np.empty((ob.n, 3))
    for q, (ri, s, pos) in enumerate(zip(ob.ri, ob.s, ob.pos)):
        st, rec = states[ri][s], records[ri]
        t = HORIZON - TURNS_PER_STAGE * s if time_sensitive else HORIZON
        v = vt[f"{rec['stat_profile']}__{rec['optimal_roles']}"][t][:, st["intent"], st["thp"], st["ehp"]]
        b = ftm if ftm_rows[q] else st[kind]
        PM[q], EV[q] = own_marginal(b, pos), expected_values(b, pos, v)
    return PM, EV


def score(ob, mask, PM, EV, records, P):
    out = {}
    for m in BAYES:
        M = bayes_marginals(ob, PM, EV, m, P[m])
        out[m] = {"params": P[m], **metrics(M, ob, mask), "combo_r": combo_r(M, ob, records),
                  "participant_ll": participant_lls(M, ob, mask)}
    return out


def row(m, r, ref):
    d = r["agg_ll"] - ref[m]["agg_ll"]
    return (f"| {m} | {mt.fmt_params(r['params'])} | {r['agg_ll']:.4f} | {d:+.4f} | {r['mean_ll']:.4f} | "
            f"{r['marg_r']:.3f} | {r['combo_r']:.3f} | {sum(r['participant_ll'].values()):.1f} |")


def main():
    params = json.load(open(BASE_DIR / "inference-fitting" / "params.json"))
    with open(BASE_DIR / "inference-fitting" / "participant_types.csv") as f:
        p_nu = {r["participant_id"]: float(r["p_non_updater"]) for r in csv.DictReader(f)}
    records = load_records()
    ob = Obs(records, p_nu)
    nu = ob.non_updater
    states = all_states(records, params)
    vt = np.load(HERE / "value_by_turns.npz")
    for rec in records:  # time-indexed table's last slice = shipped values
        assert np.allclose(vt[f"{rec['stat_profile']}__{rec['optimal_roles']}"][HORIZON],
                           rec["env_config"]["values"], atol=1e-3)
    log("[PASS] V(t = HORIZON) in value_by_turns.npz matches every record's shipped values.npy")

    ref_dir = VAR_DIR / "tables" / "fit-results"
    ref = {k: json.load(open(ref_dir / f"{k}.json")) for k in ("U-mixed", "NU-mixed")}

    # sanity: forcing t = HORIZON reproduces the 2026-11-12 fits exactly
    PM0, EV0 = tbl_beliefs(records, ob, params, states, vt, "mixed", nu, time_sensitive=False)
    chk = mt.fit_table(ob, ~nu, PM0, EV0, records)
    assert mt.same_fit(chk, ref["U-mixed"], tol=1e-6), "t = HORIZON does not reproduce U-mixed"
    log("[PASS] time index forced to t = HORIZON reproduces the 2026-11-12 U-mixed fit exactly")

    none = np.zeros(ob.n, bool)
    U_b = tbl_beliefs(records, ob, params, states, vt, "mixed", nu)     # paper beliefs, time-sensitive V
    NU_b = tbl_beliefs(records, ob, params, states, vt, "mixed", none)  # everyone on stat-prior posterior
    # how much does the time index change the values the models see?
    dEV = np.abs(U_b[1] - EV0).max(axis=1)
    late = ob.s >= 2
    log(f"EV change from time index: median |ΔEV| stages 1-2 = {np.median(dEV[~late]):.2f}, "
        f"stages 3-5 = {np.median(dEV[late]):.2f} (EV scale 0-100)")

    log("fitting U-mixed (time-sensitive) ...")
    U = mt.fit_table(ob, ~nu, *U_b, records)
    log("fitting NU-mixed (time-sensitive) ...")
    NU = mt.fit_table(ob, nu, *NU_b, records)
    P_up = {m: U[m]["params"] for m in BAYES}
    NUasU = score(ob, nu, *NU_b, records, P_up)

    out = {"U-mixed-TS": U, "NU-mixed-TS": NU, "NU-as-U-TS": NUasU}
    (HERE / "fit-results").mkdir(exist_ok=True)
    for k, v in out.items():
        json.dump(v, open(HERE / "fit-results" / f"{k}.json", "w"), indent=2)

    n_p = {"U": len(np.unique(ob.pid[~nu])), "NU": len(np.unique(ob.pid[nu]))}
    hdr = ["| Model | Params | agg-LL | Δ vs shipped V | mean-LL | marg-r | combo-r | Σ participant LL |",
           "|---|---|---|---|---|---|---|---|"]
    md = ["# Time-sensitive value function: V(10 − 2·stage) instead of V(10)", "",
          "Δ vs shipped V = agg-LL minus the same model's agg-LL in the 2026-11-12 table with the paper's "
          "time-insensitive V (positive = time-sensitive fits better). Inference params fixed (2026-11-02).", "",
          f"## Updaters, mixed posterior (n = {n_p['U']}, {int((~nu).sum())} choices), refit", ""] + hdr
    md += [row(m, U[m], ref["U-mixed"]) for m in mt.ORDER]
    md += ["", f"## Non-updaters on updater beliefs, refit (n = {n_p['NU']}, {int(nu.sum())} choices)", ""] + hdr
    md += [row(m, NU[m], ref["NU-mixed"]) for m in mt.ORDER]
    md += ["", "## Non-updaters on updater beliefs AND updater (time-sensitive) parameters, nothing refit", ""] + hdr
    nu_asu_ref = json.load(open(VAR_DIR / "nonupd-as-updaters" / "bayes-models.json"))
    md += [row(m, NUasU[m], nu_asu_ref) for m in mt.ORDER]
    (HERE / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
