"""Per-player scoring on a subset of rows; lapse λ = 0.01 on each marginal, (1 − λ)·p + λ/3.

From 2026-10-30 fit_helpers.metrics:
  agg-LL  per (env, stage, stat class) cell, the subset's role frequencies scored
          against the mean predicted marginal.
  mean-LL mean log P(chosen role) per env, averaged over envs.
  marg-r  Pearson r between mean predicted and empirical role frequencies per (env, stage).
combo-r is team-level, so it is always computed on all teams.
"""

from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np

from shared.constants import ALL_ROLE_COMBOS
from shared.evaluation import compute_pearson, extract_metrics, run_predictions

LAM = 0.01


def metrics(M, ob, mask):
    M, a = M[mask], ob.a[mask]
    cell = np.unique(ob.cell[mask], return_inverse=True)[1]
    mcell = np.unique(ob.mcell[mask], return_inverse=True)[1]
    env = np.unique(ob.env[mask], return_inverse=True)[1]
    onehot = np.eye(3)[a]

    nc = np.bincount(cell)
    mean_pred = np.zeros((len(nc), 3)); np.add.at(mean_pred, cell, M)
    counts = np.zeros((len(nc), 3)); np.add.at(counts, cell, onehot)
    agg = float(np.sum(counts * np.log((1 - LAM) * mean_pred / nc[:, None] + LAM / 3)) / len(a))

    lp = np.log((1 - LAM) * M[np.arange(len(a)), a] + LAM / 3)
    mean_ll = float(np.mean(np.bincount(env, lp) / np.bincount(env)))

    nm = np.bincount(mcell)[:, None]
    pm = np.zeros((len(nm), 3)); np.add.at(pm, mcell, M)
    hm = np.zeros((len(nm), 3)); np.add.at(hm, mcell, onehot)
    marg_r = float(np.corrcoef((pm / nm).ravel(), (hm / nm).ravel())[0, 1])
    return {"agg_ll": agg, "mean_ll": mean_ll, "marg_r": marg_r}


def combo_r(M, ob, records, joint=None):
    """Team-level combo-r on all teams (shared.evaluation): predicted vs observed canonical-combo
    frequencies per (env, stage). Team joint = product of the 3 players' marginals, or `joint`
    (n, 27), indexed by any row of the team-stage, for models whose joint doesn't factorise."""
    row = {(ri, s, pos): q for q, (ri, s, pos) in enumerate(zip(ob.ri, ob.s, ob.pos))}

    def predict(ri):
        rec = records[ri]
        preds = []
        for s, combo in enumerate(rec["stage_roles"]):
            qs = [row[(ri, s, pos)] for pos in range(3)]
            j = (joint[qs[0]] if joint is not None
                 else np.einsum("i,j,k->ijk", *M[qs]).ravel())
            preds.append({"predicted_dist": dict(zip(ALL_ROLE_COMBOS, j)), "human_combo": combo,
                          "model_marginal": M[qs].mean(axis=0)})
        return preds

    idx = {id(rec): ri for ri, rec in enumerate(records)}
    with warnings.catch_warnings():  # per-env Pearson on constant inputs; only the global r is used
        warnings.simplefilter("ignore")
        corrs = compute_pearson(run_predictions(records, lambda rec: predict(idx[id(rec)])))
    return float(extract_metrics(corrs)["combo_r"])


def participant_lls(M, ob, mask):
    """{participant: Σ log[(1 − λ)·p(chosen) + λ/3]} over the subset's rows."""
    lp = np.log((1 - LAM) * M[np.arange(ob.n), ob.a] + LAM / 3)
    out = defaultdict(float)
    for pid, x in zip(ob.pid[mask], lp[mask]):
        out[pid] += float(x)
    return dict(out)
