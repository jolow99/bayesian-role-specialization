"""Model-agnostic evaluation: aggregation by environment and Pearson correlation.

Ported from computational_model/analysis/online_model_sim.py.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr

from .constants import ROLE_CHAR_TO_IDX, ROLE_SHORT, TURNS_PER_STAGE
from .parsing import canonical_combo, get_canonical_combos
from .inference import combo_marginal


def run_predictions(records, predict_fn):
    """Run a prediction function on all records, aggregate by environment.

    Args:
        records: List of team-round dicts (from load_team_rounds or equivalent).
            Each must have: env_id, stat_profile, optimal_roles, stage_roles, env_config.
        predict_fn: Callable(record) -> list[dict], where each dict has:
            - predicted_dist: dict[combo_str -> prob]
            - human_combo: str
            - model_marginal: length-3 array

    Returns:
        Dict keyed by env_id with aggregated predictions, human counts,
        marginals, and per-team predictions.
    """
    by_env = defaultdict(list)
    for rec in records:
        by_env[rec["env_id"]].append(rec)

    all_results = {}

    for env_id, env_records in by_env.items():
        stat_profile = env_records[0]["stat_profile"]
        optimal = env_records[0]["optimal_roles"]
        canon_combos = get_canonical_combos(stat_profile)

        stage_predicted = defaultdict(lambda: defaultdict(float))
        stage_human = defaultdict(lambda: defaultdict(int))
        stage_model_marg = defaultdict(lambda: np.zeros(3))
        stage_human_marg = defaultdict(lambda: np.zeros(3))
        stage_counts = defaultdict(int)
        team_predictions = []
        max_stages = 0

        for rec in env_records:
            preds = predict_fn(rec)
            team_predictions.append(preds)

            for s, pred in enumerate(preds):
                stage_counts[s] += 1
                max_stages = max(max_stages, s + 1)

                for combo, prob in pred["predicted_dist"].items():
                    stage_predicted[s][canonical_combo(combo, stat_profile)] += prob
                stage_human[s][canonical_combo(pred["human_combo"], stat_profile)] += 1

                stage_model_marg[s] += pred["model_marginal"]
                stage_human_marg[s] += combo_marginal(pred["human_combo"])

        predicted_avg, model_marg_avg, human_marg_avg = {}, {}, {}
        for s in range(max_stages):
            n = stage_counts[s]
            if n > 0:
                predicted_avg[s] = {cc: stage_predicted[s].get(cc, 0.0) / n for cc in canon_combos}
                model_marg_avg[s] = stage_model_marg[s] / n
                human_marg_avg[s] = stage_human_marg[s] / n

        all_results[env_id] = {
            "stat_profile": stat_profile,
            "optimal": optimal,
            "canonical_optimal": canonical_combo(optimal, stat_profile),
            "canonical_combos": canon_combos,
            "n_teams": len(env_records),
            "max_stages": max_stages,
            "stage_predicted": predicted_avg,
            "stage_human": dict(stage_human),
            "stage_counts": dict(stage_counts),
            "team_predictions": team_predictions,
            "stage_model_marginal": model_marg_avg,
            "stage_human_marginal": human_marg_avg,
        }

    return all_results


def compute_pearson(all_results):
    """Compute Pearson r between predicted and observed distributions.

    Returns dict keyed by env_id (plus "__global__") with:
        - combo: {r, p, n} — correlation over canonical combo distributions
        - marginal: {r, p, n} — correlation over per-player marginals
    """
    correlations = {}
    global_combo_m, global_combo_h = [], []
    global_marg_m, global_marg_h = [], []

    for env_id, data in all_results.items():
        combo_m, combo_h, marg_m, marg_h = [], [], [], []

        for s in range(data["max_stages"]):
            predicted = data["stage_predicted"].get(s)
            human_counts = data["stage_human"].get(s, {})
            n = data["stage_counts"].get(s, 0)
            if predicted is None or n == 0:
                continue

            for cc in data["canonical_combos"]:
                combo_m.append(predicted.get(cc, 0.0))
                combo_h.append(human_counts.get(cc, 0) / n)

            mm = data["stage_model_marginal"].get(s)
            hm = data["stage_human_marginal"].get(s)
            if mm is not None and hm is not None:
                marg_m.extend(mm.tolist())
                marg_h.extend(hm.tolist())

        env_corr = {}
        if len(combo_m) >= 2:
            r, p = pearsonr(combo_m, combo_h)
            env_corr["combo"] = {"r": float(r), "p": float(p), "n": len(combo_m)}
        if len(marg_m) >= 2:
            r, p = pearsonr(marg_m, marg_h)
            env_corr["marginal"] = {"r": float(r), "p": float(p), "n": len(marg_m)}

        correlations[env_id] = env_corr
        global_combo_m.extend(combo_m)
        global_combo_h.extend(combo_h)
        global_marg_m.extend(marg_m)
        global_marg_h.extend(marg_h)

    global_corr = {}
    if len(global_combo_m) >= 2:
        r, p = pearsonr(global_combo_m, global_combo_h)
        global_corr["combo"] = {"r": float(r), "p": float(p), "n": len(global_combo_m)}
    if len(global_marg_m) >= 2:
        r, p = pearsonr(global_marg_m, global_marg_h)
        global_corr["marginal"] = {"r": float(r), "p": float(p), "n": len(global_marg_m)}
    correlations["__global__"] = global_corr

    return correlations


def compute_log_likelihood(all_results):
    """Compute mean log-likelihood of human choices under the model.

    Returns dict keyed by env_id with mean_ll and n.
    """
    ll_by_env = {}
    for env_id, data in all_results.items():
        log_liks = []
        for team_preds in data["team_predictions"]:
            for pred in team_preds:
                prob = pred["predicted_dist"].get(pred["human_combo"], 1e-20)
                log_liks.append(np.log(max(prob, 1e-20)))
        if log_liks:
            ll_by_env[env_id] = {"mean_ll": float(np.mean(log_liks)), "n": len(log_liks)}
    return ll_by_env


def extract_metrics(corrs):
    """Extract combo_r and marg_r from compute_pearson output."""
    g = corrs.get("__global__", {})
    return {
        "combo_r": g.get("combo", {}).get("r", float("nan")),
        "marg_r": g.get("marginal", {}).get("r", float("nan")),
    }
