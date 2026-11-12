"""Bayesian response-model tables for the 2026-11-10 robustness checks (Xuan).

Updaters (n = 50): mixed vs unmixed posterior.
  U-mixed    updaters on the mixed posterior (paper)
  U-unmixed  updaters on the unmixed posterior P_N
Non-updaters (n = 52): paper vs modelled with updater beliefs.
  NU-paper    fixed FTM prior (paper, Table S6 / Fig S6)
  NU-mixed    updater beliefs: mixed posterior from the stat prior
  NU-unmixed  updater beliefs: unmixed posterior from the stat prior

All posteriors come from the stat prior P_0 at the fixed inference fit (2026-11-02 params.json).
Free parameters maximise agg-LL on the group's rows (2026-11-02 fit_bayes, unchanged).
combo-r is team-level on all teams; teammates hold the paper beliefs (updaters mixed,
non-updaters FTM) except in U-unmixed (updaters unmixed) and NU-mixed / NU-unmixed (every
player on the stat-prior posterior).

Verification (verify.log): beliefs match the 2026-11-02 code; U-mixed and NU-paper reproduce the
2026-11-02 fits exactly; every table reproduces the agg-LL and combo-r values reported on 2026-11-12.

Outputs: tables/{fit-results/<table>.json, tables.md, verify.log}

Run from analysis/:
    uv run python experiments/2026-11-12_response-refit-variants/make_tables.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent / "2026-11-02_ftm-prior-fitting"
sys.path.insert(0, str(BASE_DIR / "response-fitting"))
sys.path.insert(0, str(SCRIPT_DIR))

spec = importlib.util.spec_from_file_location("base_fit", BASE_DIR / "response-fitting" / "fit.py")
base_fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_fit)

from beliefs import all_states, stat_beliefs  # noqa: E402
from models.beliefs import player_beliefs  # noqa: E402
from models.data import Obs, load_records  # noqa: E402
from models.predict import BAYES, bayes_marginals  # noqa: E402
from models.scoring import combo_r, metrics, participant_lls  # noqa: E402

TABLE_DIR = SCRIPT_DIR / "tables"
OUT_DIR = TABLE_DIR / "fit-results"
ORDER = ["Bayesian-BR", "Bayesian-Walk-BR", "Bayesian-Threshold-BR",
         "Bayesian-PS", "Bayesian-Walk-PS", "Bayesian-Threshold-PS"]
TABLES = {  # name: (group, title)
    "U-mixed": ("updaters", "Updaters — mixed posterior (1 − β)·P_N + β·P_0 (paper)"),
    "U-unmixed": ("updaters", "Updaters — unmixed posterior P_N"),
    "NU-paper": ("non-updaters", "Non-updaters — paper: fixed FTM prior (Table S6 / Fig S6)"),
    "NU-mixed": ("non-updaters", "Non-updaters — updater beliefs: mixed posterior (1 − β)·P_N + β·P_0"),
    "NU-unmixed": ("non-updaters", "Non-updaters — updater beliefs: unmixed posterior P_N"),
}
# (agg-LL, combo-r) per model in ORDER, as reported in the 2026-11-12 thread
REPORTED = {
    "U-mixed": [(-0.9832, .408), (-0.9196, .482), (-0.9832, .408), (-0.9847, .394), (-0.9428, .442), (-0.9548, .458)],
    "U-unmixed": [(-0.9755, .410), (-0.9157, .481), (-0.9751, .409), (-0.9650, .458), (-0.9607, .468), (-0.9650, .458)],
    "NU-paper": [(-1.0565, .408), (-1.0339, .481), (-1.0567, .411), (-1.1053, .394), (-1.0412, .443), (-1.1053, .394)],
    "NU-mixed": [(-1.0484, .408), (-1.0283, .489), (-1.0484, .408), (-1.0460, .447), (-1.0378, .459), (-1.0414, .464)],
    "NU-unmixed": [(-1.0483, .400), (-1.0279, .489), (-1.0483, .401), (-1.0566, .393), (-1.0566, .393), (-1.0566, .393)],
}
LOG = []


def log(msg):
    print(msg)
    LOG.append(msg)


def check(cond, msg):
    log(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    assert cond, msg


def fit_table(ob, mask, PM, EV, records):
    P = base_fit.fit_bayes(ob, mask, PM, EV)
    out = {}
    for m in BAYES:
        M = bayes_marginals(ob, PM, EV, m, P[m])
        out[m] = {"params": P[m], **metrics(M, ob, mask), "combo_r": combo_r(M, ob, records),
                  "participant_ll": participant_lls(M, ob, mask)}
    return out


def same_fit(a, b, tol=1e-9):
    for m in b:
        pa, pb = a[m]["params"], b[m]["params"]
        if set(pa) != set(pb) or any(abs(pa[k] - pb[k]) > 1e-6 for k in pa):
            return False
        if any(abs(a[m][k] - b[m][k]) > tol for k in ("agg_ll", "mean_ll", "marg_r", "combo_r")):
            return False
        la, lb = a[m]["participant_ll"], b[m]["participant_ll"]
        if set(la) != set(lb) or any(abs(la[p] - lb[p]) > tol for p in la):
            return False
    return set(a) == set(b)


def fmt_params(p):
    names = {"tau_softmax": "τ_v", "epsilon_switch": "ε_s", "delta": "δ"}
    return ", ".join(f"{names[k]} = {v:.3g}" for k, v in p.items()) or "–"


def table_md(title, res, n_p, n_c):
    best = max(res, key=lambda m: res[m]["agg_ll"])
    L = [f"### {title} (n = {n_p}, {n_c} choices)", "",
         "| Model | Params | agg-LL | mean-LL | marg-r | combo-r | Σ participant LL |",
         "|---|---|---|---|---|---|---|"]
    for m in ORDER:
        r = res[m]
        b = (lambda x: f"**{x}**") if m == best else (lambda x: x)
        agg = "%.4f" % r["agg_ll"]
        L.append(f"| {b(m)} | {fmt_params(r['params'])} | {b(agg)} | {r['mean_ll']:.4f} | "
                 f"{r['marg_r']:.3f} | {r['combo_r']:.3f} | {sum(r['participant_ll'].values()):.1f} |")
    return L + ["", "&nbsp;", ""]


def main():
    params = json.load(open(BASE_DIR / "inference-fitting" / "params.json"))
    with open(BASE_DIR / "inference-fitting" / "participant_types.csv") as f:
        p_nu = {r["participant_id"]: float(r["p_non_updater"]) for r in csv.DictReader(f)}
    records = load_records()
    ob = Obs(records, p_nu)
    nu = ob.non_updater
    masks = {"updaters": ~nu, "non-updaters": nu}

    states = all_states(records, params)
    paper = player_beliefs(records, ob, params, nu)  # 2026-11-02: updaters mixed, non-updaters FTM
    mixed = stat_beliefs(records, ob, "mixed", states)
    unmixed = stat_beliefs(records, ob, "unmixed", states)
    B = {"U-mixed": paper,
         "U-unmixed": tuple(np.where(nu[:, None], p, u) for p, u in zip(paper, unmixed)),
         "NU-paper": paper, "NU-mixed": mixed, "NU-unmixed": unmixed}

    log("Belief checks")
    ref = player_beliefs(records, ob, params, np.zeros(ob.n, bool))
    check(all(np.array_equal(a, b) for a, b in zip(mixed, ref)),
          "mixed = 2026-11-02 updater belief on every row")
    s0 = ob.s == 0
    check(all(np.allclose(unmixed[k][s0], mixed[k][s0]) for k in (0, 1)), "unmixed == mixed at stage 1")
    beta, err = params["beta"], 0.0
    for sl in states:
        for st in sl[1:]:
            pred = (1 - beta) * st["unmixed"] + beta * st["prior"]
            err = max(err, float(np.abs(pred / pred.sum() - st["mixed"]).max()))
    check(err < 1e-12, f"mixed == (1 − β)·unmixed + β·P_0 at stages ≥ 2 (max abs err {err:.1e})")
    d = np.abs(unmixed[0][~s0] - mixed[0][~s0]).max(axis=1)
    check((d > 1e-6).all(), f"unmixed ≠ mixed on all {len(d)} later-stage rows (median |ΔPM| {np.median(d):.3f})")
    n_p = {g: len(np.unique(ob.pid[m])) for g, m in masks.items()}
    check(n_p == {"updaters": 50, "non-updaters": 52}, f"participants per group {n_p}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    R = {}
    for name, (g, _) in TABLES.items():
        log(f"fitting {name} ...")
        R[name] = fit_table(ob, masks[g], *B[name], records)
        with open(OUT_DIR / f"{name}.json", "w") as f:
            json.dump(R[name], f, indent=2)

    log("Fit checks")
    base = BASE_DIR / "response-fitting" / "fit-results"
    check(same_fit(R["U-mixed"], json.load(open(base / "updaters" / "bayes-models.json"))),
          "U-mixed reproduces the 2026-11-02 updater fit exactly")
    check(same_fit(R["NU-paper"], json.load(open(base / "non-updaters" / "bayes-models.json"))),
          "NU-paper reproduces the 2026-11-02 non-updater fit exactly")
    for name, rep in REPORTED.items():
        ok = all(abs(R[name][m]["agg_ll"] - a) < 6e-5 and abs(R[name][m]["combo_r"] - c) < 6e-4
                 for m, (a, c) in zip(ORDER, rep))
        check(ok, f"{name}: agg-LL and combo-r match the values reported in chat")

    md = ["# Bayesian response models: mixed vs unmixed posterior, and non-updaters on updater beliefs", "",
          f"Fixed inference params (2026-11-02 params.json): τ_p = {params['tau_prior']:.3f}, "
          f"ε = {params['epsilon']:.3f}, β = {params['beta']:.3f}. Posteriors start from the stat prior P_0. "
          "Free parameters maximise agg-LL on the group's rows; best agg-LL in bold.", "",
          "## Updaters", ""]
    for name, (g, title) in TABLES.items():
        if name == "NU-paper":
            md += ["## Non-updaters", ""]
        md += table_md(title, R[name], n_p[g], int(masks[g].sum()))
    md += ["In NU-paper, Bayesian-PS is the paper's Prior Sampling (sampling from the FTM prior)."]
    (TABLE_DIR / "tables.md").write_text("\n".join(md) + "\n")
    (TABLE_DIR / "verify.log").write_text("\n".join(LOG) + "\n")
    log("done")


if __name__ == "__main__":
    main()
