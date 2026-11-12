"""Non-updaters scored as if they were exactly updaters (Xuan, 2026-11-12).

Nothing is fit on the non-updaters. Each of the 52 non-updaters gets:
  beliefs     the updater belief: mixed Bayesian posterior from the stat prior (as in NU-mixed),
              at the shared inference fit (τ_p, ε, β)
  parameters  the updaters' fitted response parameters (fit-results/U-mixed.json = 2026-11-02 updater fit)
The six Bayesian models are scored on the non-updaters' choices (agg-LL, mean-LL, marg-r, combo-r,
participant LLs). combo-r: all teams, every player on the stat-prior posterior.

Individual fits (Fig S6 style, 2026-11-02 indiv_fits.py plotting): per non-updater, softmax over
participant LLs of the six Bayesian models at updater params, the non-Bayesian baselines at their
2026-11-02 non-updater fits, and Prior Sampling (Bayesian-PS on the FTM prior).

Checks: updater params equal the 2026-11-02 updater fit; scoring the updaters' own rows at those
params reproduces U-mixed (agg-LL, mean-LL, marg-r, participant LLs).

Outputs (nonupd-as-updaters/): nonupd_as_updaters.md, bayes-models.json, indiv_fits_summary.json,
indiv_fits.{png,pdf} (top: all models; bottom: Bayesian models only), verify.log

Run from analysis/:
    uv run python experiments/2026-11-12_response-refit-variants/nonupd_as_updaters.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import make_tables as mt  # sets up sys.path for the 2026-11-02 models package
from beliefs import all_states, stat_beliefs
from models.data import Obs, load_records
from models.predict import BAYES, bayes_marginals
from models.scoring import combo_r, metrics, participant_lls

OUT = mt.SCRIPT_DIR / "nonupd-as-updaters"
spec = importlib.util.spec_from_file_location(
    "indiv", mt.BASE_DIR / "results" / "indiv-fits" / "indiv_fits.py")
indiv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(indiv)


def score(ob, mask, PM, EV, P, records):
    out = {}
    for m in BAYES:
        M = bayes_marginals(ob, PM, EV, m, P[m])
        out[m] = {"params": P[m], **metrics(M, ob, mask), "combo_r": combo_r(M, ob, records),
                  "participant_ll": participant_lls(M, ob, mask)}
    return out


def main():
    mt.LOG.clear()
    params = json.load(open(mt.BASE_DIR / "inference-fitting" / "params.json"))
    with open(mt.BASE_DIR / "inference-fitting" / "participant_types.csv") as f:
        p_nu = {r["participant_id"]: float(r["p_non_updater"]) for r in csv.DictReader(f)}
    records = load_records()
    ob = Obs(records, p_nu)
    nu = ob.non_updater
    PM, EV = stat_beliefs(records, ob, "mixed", all_states(records, params))

    base = mt.BASE_DIR / "response-fitting" / "fit-results"
    upd = json.load(open(mt.OUT_DIR / "U-mixed.json"))
    ref = json.load(open(base / "updaters" / "bayes-models.json"))
    P = {m: upd[m]["params"] for m in BAYES}

    mt.log("Checks")
    mt.check(all(P[m] == ref[m]["params"] for m in BAYES), "updater params = 2026-11-02 updater fit")
    own = score(ob, ~nu, PM, EV, P, records)
    ok = all(abs(own[m][k] - upd[m][k]) < 1e-9 for m in BAYES for k in ("agg_ll", "mean_ll", "marg_r"))
    ok &= all(abs(own[m]["participant_ll"][p] - v) < 1e-9 for m in BAYES
              for p, v in upd[m]["participant_ll"].items())
    mt.check(ok, "scoring updaters at their params reproduces U-mixed")
    n_p = len(np.unique(ob.pid[nu]))
    mt.check(n_p == 52, f"{n_p} non-updaters, {nu.sum()} choices")

    R = score(ob, nu, PM, EV, P, records)
    OUT.mkdir(exist_ok=True)
    with open(OUT / "bayes-models.json", "w") as f:
        json.dump(R, f, indent=2)

    # individual fits
    nb = json.load(open(base / "non-updaters" / "non-bayes-models.json"))
    nu_paper = json.load(open(base / "non-updaters" / "bayes-models.json"))
    refit = json.load(open(mt.OUT_DIR / "NU-mixed.json"))
    summary = {}
    for key, bayes in (("updater_params", R), ("refit_on_nonupdaters", refit)):
        lls = {m: r["participant_ll"] for m, r in nb.items()}
        lls.update({m: r["participant_ll"] for m, r in bayes.items()})
        lls["Prior Sampling"] = nu_paper["Bayesian-PS"]["participant_ll"]
        if key == "updater_params":
            fig, axes = plt.subplots(2, 1, figsize=(3.42, 3.9), gridspec_kw={"hspace": 0.35})
            used = indiv.panel(axes[0], lls, "Non-updaters (n = {n}): all models, updater parameters",
                               summary, key)
            used += indiv.panel(axes[1], {m: lls[m] for m in BAYES}, "Non-updaters (n = {n}): Bayesian models only, updater parameters",
                                summary, "updater_params_bayesian_only")
            axes[1].set_xlabel("Participants (sorted by best-fitting model)")
            indiv.legends(fig, used, 0.04)
            for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
                fig.savefig(OUT / f"indiv_fits.{ext}", bbox_inches="tight", facecolor="white", **kw)
            plt.close(fig)
        else:
            models = indiv.drop_collapsed(lls)
            post = indiv.posteriors(lls, models)
            dom = Counter(max(models, key=lambda m: v[m]) for v in post.values())
            summary[key] = {"n": len(post), "bayesian_best": sum(c for m, c in dom.items() if m in indiv.BAYES),
                            "counts": dict(dom.most_common())}
    with open(OUT / "indiv_fits_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # table
    md = ["# Non-updaters scored as updaters (updater beliefs and updater parameters, nothing refit)", "",
          f"n = 52 non-updaters, {int(nu.sum())} choices. Beliefs: mixed posterior from the stat prior "
          "(τ_p, ε, β from the shared inference fit). Parameters: the updaters' agg-LL fit (U-mixed). "
          "Refit = NU-mixed (same beliefs, parameters fit on non-updaters).", "",
          "| Model | Updater params | agg-LL | refit agg-LL | mean-LL | marg-r | combo-r | Σ participant LL | refit Σ LL |",
          "|---|---|---|---|---|---|---|---|---|"]
    best = max(R, key=lambda m: R[m]["agg_ll"])
    for m in mt.ORDER:
        r, f_ = R[m], refit[m]
        b = (lambda x: f"**{x}**") if m == best else (lambda x: x)
        agg = "%.4f" % r["agg_ll"]
        md.append(f"| {b(m)} | {mt.fmt_params(r['params'])} | {b(agg)} | {f_['agg_ll']:.4f} | "
                  f"{r['mean_ll']:.4f} | {r['marg_r']:.3f} | {r['combo_r']:.3f} | "
                  f"{sum(r['participant_ll'].values()):.1f} | {sum(f_['participant_ll'].values()):.1f} |")
    md += ["", "Non-Bayesian baselines on non-updaters (2026-11-02 non-updater fits):", "",
           "| Model | agg-LL | Σ participant LL |", "|---|---|---|"]
    for m in ("Random Walk", "Random", "Top-5"):
        md.append(f"| {m} | {nb[m]['agg_ll']:.4f} | {sum(nb[m]['participant_ll'].values()):.1f} |")
    s = summary
    md += ["", f"Individual fits: Bayesian model best for **{s['updater_params']['bayesian_best']}/52** "
           f"at updater params (vs {s['refit_on_nonupdaters']['bayesian_best']}/52 refit on non-updaters). "
           "Figure: indiv_fits.png", "",
           "| Best model | updater params | refit |", "|---|---|---|"]
    cu, cr = Counter(s["updater_params"]["counts"]), Counter(s["refit_on_nonupdaters"]["counts"])
    for m in indiv.ORDER:
        if cu[m] or cr[m]:
            md.append(f"| {m} | {cu[m]} | {cr[m]} |")
    (OUT / "nonupd_as_updaters.md").write_text("\n".join(md) + "\n")
    (OUT / "verify.log").write_text("\n".join(mt.LOG) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
