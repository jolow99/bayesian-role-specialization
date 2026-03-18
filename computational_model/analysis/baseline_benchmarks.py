"""
Baseline benchmarks for comparing against the Bayesian role-specialization model.

Implements all baselines:
  1. Random: Every agent randomly chooses among all 3 roles (uniform 1/27).
  2. Random Walk: Agents stick to previous role (1-eps), switch randomly (eps),
     with eps fitted to human data.
  3. Optimal: Team plays optimal role combo from the start (uniform over ties).
  4. Random Among Top-k: Team randomly plays one of the top-k role combos at each
     stage, with k fitted to human data. No learning over stages.
  5. Random-to-Optimal: Linearly interpolates between uniform distribution and the
     optimal role combo over stages within each round.
  6. Copy Others: Agents try to play the same role as what they inferred the other
     players were playing in the previous stage.
  7. Contradict Others: Agents try to play a different role from what they inferred
     the other players were playing in the previous stage.

Computes combo_r and marg_r for each baseline and compares against the main model.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from online_model_sim import (
    ALL_ROLE_COMBOS,
    ROLE_CHAR_TO_IDX,
    ROLE_NAMES,
    MAX_STAGES,
    canonical_combo,
    combo_marginal,
    get_canonical_combos,
    load_team_rounds,
    run_all_predictions,
    compute_pearson,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def _combo_to_flat_idx(combo_str):
    return ROLE_CHAR_TO_IDX[combo_str[0]] * 9 + ROLE_CHAR_TO_IDX[combo_str[1]] * 3 + ROLE_CHAR_TO_IDX[combo_str[2]]


def _dist_to_marginal(dist):
    marginal = np.zeros(3)
    for combo_str, prob in dist.items():
        if prob > 0:
            marginal += prob * combo_marginal(combo_str)
    return marginal


# ── Baseline: Random ─────────────────────────────────────────────────────────

def random_predictions(record):
    """Uniform 1/27 over all combos at every stage."""
    dist = {c: 1.0 / 27.0 for c in ALL_ROLE_COMBOS}
    marginal = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    return [
        {"predicted_dist": dict(dist), "human_combo": hc, "model_marginal": marginal.copy()}
        for hc in record["stage_roles"]
    ]


# ── Baseline: Random Walk ────────────────────────────────────────────────────

def random_walk_predictions(record, eps):
    """Each agent sticks to previous role with prob (1-eps), switches to random with eps.
    Stage 0: uniform. Stage s>0: based on human roles at stage s-1."""
    results = []
    for s, human_combo in enumerate(record["stage_roles"]):
        if s == 0:
            # No previous stage — uniform
            per_agent = [np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]) for _ in range(3)]
        else:
            prev_combo = record["stage_roles"][s - 1]
            per_agent = []
            for i in range(3):
                prev_role = ROLE_CHAR_TO_IDX[prev_combo[i]]
                probs = np.full(3, eps / 3.0)
                probs[prev_role] += 1.0 - eps
                per_agent.append(probs)

        # Joint dist from independent per-agent
        dist = {}
        for combo_str in ALL_ROLE_COMBOS:
            r0, r1, r2 = ROLE_CHAR_TO_IDX[combo_str[0]], ROLE_CHAR_TO_IDX[combo_str[1]], ROLE_CHAR_TO_IDX[combo_str[2]]
            dist[combo_str] = float(per_agent[0][r0] * per_agent[1][r1] * per_agent[2][r2])

        results.append({
            "predicted_dist": dist,
            "human_combo": human_combo,
            "model_marginal": np.mean(per_agent, axis=0),
        })
    return results


def fit_random_walk(records, eps_values=None):
    """Find eps that maximizes marg_r for random walk baseline."""
    if eps_values is None:
        eps_values = np.concatenate([
            np.linspace(0.001, 0.1, 20),
            np.linspace(0.1, 1.0, 30),
        ])

    best_eps, best_combo_r = 0.5, -np.inf
    results_by_eps = {}

    for eps in eps_values:
        preds = run_baseline_predictions(records, random_walk_predictions, eps=eps)
        corrs = compute_pearson(preds)
        metrics = extract_global_metrics(corrs)
        results_by_eps[float(eps)] = metrics

        if not np.isnan(metrics["combo_r"]) and metrics["combo_r"] > best_combo_r:
            best_eps = float(eps)
            best_combo_r = metrics["combo_r"]

        print(f"  ε={eps:.4f}  combo_r={metrics['combo_r']:.4f}  marg_r={metrics['marg_r']:.4f}")

    return best_eps, results_by_eps


# ── Baseline: Optimal ────────────────────────────────────────────────────────

def optimal_predictions(record):
    """Always predict the optimal combo (uniform over ties). Same every stage."""
    env = record["env_config"]
    values = env["values"]
    team_max_hp, enemy_max_hp = env["team_max_hp"], env["enemy_max_hp"]

    initial_intent = record["lds"][0] if record["lds"] else 0
    dist = _optimal_combo_dist(values, initial_intent, team_max_hp, enemy_max_hp)
    marginal = _dist_to_marginal(dist)

    return [
        {"predicted_dist": dict(dist), "human_combo": hc, "model_marginal": marginal.copy()}
        for hc in record["stage_roles"]
    ]


# ── Baseline: Copy Others ────────────────────────────────────────────────────

def copy_others_predictions(record):
    """Each agent tries to play the same role as what others played last stage.
    Stage 0: uniform. Stage s>0: each agent looks at the other 2 players' roles
    in stage s-1 and picks one of those (uniform over the two, or the shared role
    if both played the same)."""
    results = []
    for s, human_combo in enumerate(record["stage_roles"]):
        if s == 0:
            per_agent = [np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]) for _ in range(3)]
        else:
            prev_combo = record["stage_roles"][s - 1]
            per_agent = []
            for i in range(3):
                others = [ROLE_CHAR_TO_IDX[prev_combo[j]] for j in range(3) if j != i]
                probs = np.zeros(3)
                for r in others:
                    probs[r] += 0.5
                per_agent.append(probs)

        dist = {}
        for combo_str in ALL_ROLE_COMBOS:
            r0, r1, r2 = ROLE_CHAR_TO_IDX[combo_str[0]], ROLE_CHAR_TO_IDX[combo_str[1]], ROLE_CHAR_TO_IDX[combo_str[2]]
            dist[combo_str] = float(per_agent[0][r0] * per_agent[1][r1] * per_agent[2][r2])

        results.append({
            "predicted_dist": dist,
            "human_combo": human_combo,
            "model_marginal": np.mean(per_agent, axis=0),
        })
    return results


# ── Baseline: Contradict Others ──────────────────────────────────────────────

def contradict_others_predictions(record):
    """Each agent tries to play a different role from what others played last stage.
    Stage 0: uniform. Stage s>0: each agent looks at the other 2 players' roles
    in stage s-1 and picks uniformly among roles NOT played by the others."""
    results = []
    for s, human_combo in enumerate(record["stage_roles"]):
        if s == 0:
            per_agent = [np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]) for _ in range(3)]
        else:
            prev_combo = record["stage_roles"][s - 1]
            per_agent = []
            for i in range(3):
                others_roles = set(ROLE_CHAR_TO_IDX[prev_combo[j]] for j in range(3) if j != i)
                different_roles = [r for r in range(3) if r not in others_roles]
                if not different_roles:
                    # Both others played same role — 2 roles are different
                    different_roles = [r for r in range(3) if r not in others_roles]
                if not different_roles:
                    # All 3 roles taken by others (impossible with 2 others, 3 roles)
                    different_roles = [0, 1, 2]
                probs = np.zeros(3)
                for r in different_roles:
                    probs[r] = 1.0 / len(different_roles)
                per_agent.append(probs)

        dist = {}
        for combo_str in ALL_ROLE_COMBOS:
            r0, r1, r2 = ROLE_CHAR_TO_IDX[combo_str[0]], ROLE_CHAR_TO_IDX[combo_str[1]], ROLE_CHAR_TO_IDX[combo_str[2]]
            dist[combo_str] = float(per_agent[0][r0] * per_agent[1][r1] * per_agent[2][r2])

        results.append({
            "predicted_dist": dist,
            "human_combo": human_combo,
            "model_marginal": np.mean(per_agent, axis=0),
        })
    return results


# ── Baseline: Random Among Top-k ──────────────────────────────────────────

def _get_topk_combos(values, intent, team_hp, enemy_hp, k):
    """Return the top-k role combos by value at a given state, uniform over them."""
    vals = values[:, intent, team_hp, enemy_hp]
    top_indices = np.argsort(vals)[::-1][:k]
    dist = {}
    for combo_str in ALL_ROLE_COMBOS:
        flat_idx = ROLE_CHAR_TO_IDX[combo_str[0]] * 9 + ROLE_CHAR_TO_IDX[combo_str[1]] * 3 + ROLE_CHAR_TO_IDX[combo_str[2]]
        dist[combo_str] = 1.0 / k if flat_idx in top_indices else 0.0
    return dist


def topk_predictions(record, k):
    """Predict uniform over top-k combos at initial state. Same dist every stage."""
    env = record["env_config"]
    values = env["values"]
    team_max_hp, enemy_max_hp = env["team_max_hp"], env["enemy_max_hp"]

    # Use initial state (first intent, full HP) to determine top-k
    initial_intent = record["lds"][0] if record["lds"] else 0
    dist = _get_topk_combos(values, initial_intent, team_max_hp, enemy_max_hp, k)

    marginal = np.zeros(3)
    for combo_str, prob in dist.items():
        if prob > 0:
            marginal += prob * combo_marginal(combo_str)

    results = []
    for human_combo in record["stage_roles"]:
        results.append({
            "predicted_dist": dict(dist),
            "human_combo": human_combo,
            "model_marginal": marginal.copy(),
        })
    return results


# ── Baseline 2: Random-to-Optimal ───────────────────────────────────────────

def _optimal_combo_dist(values, intent, team_hp, enemy_hp):
    """Return a distribution that is 1.0 on the optimal combo, 0 elsewhere.
    If there are ties, split uniformly among tied combos."""
    vals = values[:, intent, team_hp, enemy_hp]
    max_val = vals.max()
    optimal_mask = vals == max_val

    dist = {}
    n_optimal = int(optimal_mask.sum())
    for combo_str in ALL_ROLE_COMBOS:
        flat_idx = ROLE_CHAR_TO_IDX[combo_str[0]] * 9 + ROLE_CHAR_TO_IDX[combo_str[1]] * 3 + ROLE_CHAR_TO_IDX[combo_str[2]]
        dist[combo_str] = 1.0 / n_optimal if optimal_mask[flat_idx] else 0.0
    return dist


def random_to_optimal_predictions(record):
    """Linearly interpolate from uniform to optimal over stages."""
    env = record["env_config"]
    values = env["values"]
    team_max_hp, enemy_max_hp = env["team_max_hp"], env["enemy_max_hp"]

    # Optimal combo at initial state
    initial_intent = record["lds"][0] if record["lds"] else 0
    optimal_dist = _optimal_combo_dist(values, initial_intent, team_max_hp, enemy_max_hp)
    uniform_dist = {c: 1.0 / 27.0 for c in ALL_ROLE_COMBOS}

    n_stages = len(record["stage_roles"])
    results = []

    for s, human_combo in enumerate(record["stage_roles"]):
        # alpha=0 at stage 0 (fully random), alpha=1 at last stage (fully optimal)
        alpha = s / max(n_stages - 1, 1)

        interp_dist = {}
        for combo_str in ALL_ROLE_COMBOS:
            interp_dist[combo_str] = (1 - alpha) * uniform_dist[combo_str] + alpha * optimal_dist[combo_str]

        marginal = np.zeros(3)
        for combo_str, prob in interp_dist.items():
            marginal += prob * combo_marginal(combo_str)

        results.append({
            "predicted_dist": interp_dist,
            "human_combo": human_combo,
            "model_marginal": marginal,
        })

    return results


# ── Generic aggregation and evaluation ───────────────────────────────────────

def run_baseline_predictions(records, baseline_fn, **kwargs):
    """Run a baseline prediction function over all records, aggregating like the main model."""
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
            preds = baseline_fn(rec, **kwargs)
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


def extract_global_metrics(correlations):
    """Extract global combo_r and marg_r from correlations dict."""
    g = correlations.get("__global__", {})
    return {
        "combo_r": g.get("combo", {}).get("r", float("nan")),
        "marg_r": g.get("marginal", {}).get("r", float("nan")),
    }


def extract_per_env_metrics(correlations):
    """Extract per-environment combo_r and marg_r."""
    per_env = {}
    for env_id, c in correlations.items():
        if env_id == "__global__":
            continue
        per_env[env_id] = {
            "combo_r": c.get("combo", {}).get("r", float("nan")),
            "marg_r": c.get("marginal", {}).get("r", float("nan")),
        }
    return per_env


# ── Top-k fitting ────────────────────────────────────────────────────────────

def fit_topk(records, k_range=range(1, 28)):
    """Find the k that maximizes marg_r for the top-k baseline."""
    best_k, best_combo_r = 1, -np.inf
    results_by_k = {}

    for k in k_range:
        preds = run_baseline_predictions(records, topk_predictions, k=k)
        corrs = compute_pearson(preds)
        metrics = extract_global_metrics(corrs)
        results_by_k[k] = metrics

        if not np.isnan(metrics["combo_r"]) and metrics["combo_r"] > best_combo_r:
            best_k = k
            best_combo_r = metrics["combo_r"]

        print(f"  k={k:2d}  combo_r={metrics['combo_r']:.4f}  marg_r={metrics['marg_r']:.4f}")

    return best_k, results_by_k


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Baseline benchmarks for role specialization model")
    parser.add_argument("--data-dir", type=str, nargs="+", default=None,
                        help="Path(s) to data directory")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    print("Loading human data ...")
    records = load_team_rounds(data_dirs=args.data_dir)
    n_envs = len(set(r["env_id"] for r in records))
    print(f"Loaded {len(records)} team-rounds across {n_envs} envs\n")

    # ── Main model (finetuned params) ──
    MAIN_TAU_PRIOR = 0.9771
    MAIN_TAU_SOFTMAX = 10.0
    MAIN_EPSILON = 0.2
    print("=" * 60)
    print(f"Main Model (τ_prior={MAIN_TAU_PRIOR}, τ_softmax={MAIN_TAU_SOFTMAX}, ε={MAIN_EPSILON})")
    print("=" * 60)
    main_results = run_all_predictions(records, tau_prior=MAIN_TAU_PRIOR, tau_softmax=MAIN_TAU_SOFTMAX, epsilon=MAIN_EPSILON)
    main_corrs = compute_pearson(main_results)
    main_metrics = extract_global_metrics(main_corrs)
    main_per_env = extract_per_env_metrics(main_corrs)
    print(f"  combo_r = {main_metrics['combo_r']:.4f}")
    print(f"  marg_r  = {main_metrics['marg_r']:.4f}")
    for env_id in sorted(main_per_env):
        m = main_per_env[env_id]
        print(f"    {env_id}: combo_r={m['combo_r']:.4f}, marg_r={m['marg_r']:.4f}")

    # Collect all baseline results: (name, global_metrics, per_env_metrics)
    baselines = []

    def run_and_print(name, baseline_fn, **kwargs):
        print(f"\n{'=' * 60}")
        print(name)
        print("=" * 60)
        preds = run_baseline_predictions(records, baseline_fn, **kwargs)
        corrs = compute_pearson(preds)
        metrics = extract_global_metrics(corrs)
        per_env = extract_per_env_metrics(corrs)
        print(f"  combo_r = {metrics['combo_r']:.4f}")
        print(f"  marg_r  = {metrics['marg_r']:.4f}")
        for env_id in sorted(per_env):
            m = per_env[env_id]
            print(f"    {env_id}: combo_r={m['combo_r']:.4f}, marg_r={m['marg_r']:.4f}")
        return metrics, per_env

    # ── 1. Random ──
    rand_metrics, rand_per_env = run_and_print("Baseline: Random (uniform 1/27)", random_predictions)
    baselines.append(("Random", rand_metrics, rand_per_env, {}))

    # ── 2. Random Walk (fit eps) ──
    print(f"\n{'=' * 60}")
    print("Baseline: Random Walk (fitting ε)")
    print("=" * 60)
    best_eps, rw_by_eps = fit_random_walk(records)
    print(f"\n  Best ε = {best_eps:.4f}")
    rw_metrics = rw_by_eps[best_eps]
    print(f"  combo_r = {rw_metrics['combo_r']:.4f}")
    print(f"  marg_r  = {rw_metrics['marg_r']:.4f}")
    rw_preds = run_baseline_predictions(records, random_walk_predictions, eps=best_eps)
    rw_corrs = compute_pearson(rw_preds)
    rw_per_env = extract_per_env_metrics(rw_corrs)
    for env_id in sorted(rw_per_env):
        m = rw_per_env[env_id]
        print(f"    {env_id}: combo_r={m['combo_r']:.4f}, marg_r={m['marg_r']:.4f}")
    baselines.append((f"Random Walk (ε={best_eps:.2f})", rw_metrics, rw_per_env, {"best_eps": best_eps}))

    # ── 3. Optimal ──
    opt_metrics, opt_per_env = run_and_print("Baseline: Optimal", optimal_predictions)
    baselines.append(("Optimal", opt_metrics, opt_per_env, {}))

    # ── 4. Random Among Top-k (fit k) ──
    print(f"\n{'=' * 60}")
    print("Baseline: Random Among Top-k (fitting k)")
    print("=" * 60)
    best_k, topk_by_k = fit_topk(records)
    print(f"\n  Best k = {best_k}")
    topk_metrics = topk_by_k[best_k]
    print(f"  combo_r = {topk_metrics['combo_r']:.4f}")
    print(f"  marg_r  = {topk_metrics['marg_r']:.4f}")
    topk_preds = run_baseline_predictions(records, topk_predictions, k=best_k)
    topk_corrs = compute_pearson(topk_preds)
    topk_per_env = extract_per_env_metrics(topk_corrs)
    for env_id in sorted(topk_per_env):
        m = topk_per_env[env_id]
        print(f"    {env_id}: combo_r={m['combo_r']:.4f}, marg_r={m['marg_r']:.4f}")
    baselines.append((f"Random Among Top-{best_k}", topk_metrics, topk_per_env, {"best_k": best_k}))

    # ── 5. Random-to-Optimal ──
    r2o_metrics, r2o_per_env = run_and_print("Baseline: Random-to-Optimal", random_to_optimal_predictions)
    baselines.append(("Random-to-Optimal", r2o_metrics, r2o_per_env, {}))

    # ── 6. Copy Others ──
    copy_metrics, copy_per_env = run_and_print("Baseline: Copy Others", copy_others_predictions)
    baselines.append(("Copy Others", copy_metrics, copy_per_env, {}))

    # ── 7. Contradict Others ──
    contra_metrics, contra_per_env = run_and_print("Baseline: Contradict Others", contradict_others_predictions)
    baselines.append(("Contradict Others", contra_metrics, contra_per_env, {}))

    # ── Summary table ──
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<30} {'combo_r':>10} {'marg_r':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    print(f"  {'Bayesian (finetuned)':<30} {main_metrics['combo_r']:>10.4f} {main_metrics['marg_r']:>10.4f}")
    for name, metrics, _, _ in baselines:
        cr = metrics['combo_r'] if not np.isnan(metrics['combo_r']) else float('nan')
        mr = metrics['marg_r'] if not np.isnan(metrics['marg_r']) else float('nan')
        print(f"  {name:<30} {cr:>10.4f} {mr:>10.4f}")

    # Per-env marg_r summary
    all_envs = sorted(set(list(main_per_env) + [e for _, _, pe, _ in baselines for e in pe]))
    header_names = ["Bayesian"] + [n[:8] for n, _, _, _ in baselines]
    print(f"\n  Per-environment marg_r:")
    print(f"  {'Environment':<25}", end="")
    for h in header_names:
        print(f" {h:>10}", end="")
    print()
    print(f"  {'-'*25}", end="")
    for _ in header_names:
        print(f" {'-'*10}", end="")
    print()
    for env_id in all_envs:
        print(f"  {env_id:<25}", end="")
        print(f" {main_per_env.get(env_id, {}).get('marg_r', float('nan')):>10.4f}", end="")
        for _, _, pe, _ in baselines:
            print(f" {pe.get(env_id, {}).get('marg_r', float('nan')):>10.4f}", end="")
        print()
    print(f"  {'GLOBAL':<25}", end="")
    print(f" {main_metrics['marg_r']:>10.4f}", end="")
    for _, metrics, _, _ in baselines:
        print(f" {metrics['marg_r']:>10.4f}", end="")
    print()

    # Save
    output = {
        "main_model": {
            "params": {"tau_prior": MAIN_TAU_PRIOR, "tau_softmax": MAIN_TAU_SOFTMAX, "epsilon": MAIN_EPSILON},
            "global": main_metrics,
            "per_env": main_per_env,
        },
    }
    for name, metrics, per_env, extra in baselines:
        key = name.lower().replace(" ", "_").replace("-", "_")
        output[key] = {
            "global": metrics,
            "per_env": {eid: dict(m) for eid, m in per_env.items()},
            **extra,
        }
    if topk_by_k:
        output["random_among_top_" + str(best_k)]["k_sweep"] = {str(k): dict(v) for k, v in topk_by_k.items()}

    out_path = Path(args.output) if args.output else SCRIPT_DIR / "baseline_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
