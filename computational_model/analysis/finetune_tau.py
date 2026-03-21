"""
Fine-tune model parameters (τ_prior, τ_softmax, and ε) by sweeping over a grid
and selecting the combination that best fits human data.

τ_prior  — temperature for the utility-based prior
τ_softmax — temperature for the per-agent softmax role selection

Evaluates each combination using:
  - Global Pearson r (combo-level and marginal)
  - Mean log-likelihood of human choices

Supports utility-based or uniform prior via --prior flag.
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from online_model_sim import (
    DEFAULT_DATA_DIRS,
    EPSILON,
    TAU_PRIOR,
    TAU_SOFTMAX,
    compute_pearson,
    load_team_rounds,
    plot_comparison,
    run_all_predictions,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def evaluate(records, tau_prior, tau_softmax, prior_type="utility", epsilon=EPSILON):
    """Run model at given (τ_prior, τ_softmax, ε) and return fit metrics."""
    results = run_all_predictions(records, tau_prior=tau_prior, tau_softmax=tau_softmax,
                                  prior_type=prior_type, epsilon=epsilon)
    correlations = compute_pearson(results)

    g = correlations.get("__global__", {})
    combo_r = g.get("combo", {}).get("r", float("nan"))
    marg_r = g.get("marginal", {}).get("r", float("nan"))

    return {
        "tau_prior": tau_prior,
        "tau_softmax": tau_softmax,
        "epsilon": epsilon,
        "combo_r": combo_r,
        "marg_r": marg_r,
        "correlations": correlations,
    }


def _evaluate_combo(args):
    """Module-level worker for parallel grid search (must be picklable)."""
    records, tp, ts, eps, prior_type = args
    return evaluate(records, tp, ts, prior_type=prior_type, epsilon=eps)


def grid_search(records, tau_prior_values, tau_softmax_values, epsilon_values, prior_type="utility"):
    """Evaluate each (τ_prior, τ_softmax, ε) combination and return results."""
    combos = list(product(tau_prior_values, tau_softmax_values, epsilon_values))
    total = len(combos)
    args_list = [(records, tp, ts, eps, prior_type) for tp, ts, eps in combos]

    results = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(_evaluate_combo, args): args for args in args_list}
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            print(f"  [{i}/{total}] τ_p={res['tau_prior']:.4f} τ_s={res['tau_softmax']:.4f} ε={res['epsilon']:.4f} ... "
                  f"combo_r={res['combo_r']:.4f}  marg_r={res['marg_r']:.4f}", flush=True)
            results.append(res)
    return results


def optimize_scipy(records, tau_prior_bounds, tau_softmax_bounds, eps_bounds, prior_type="utility"):
    """Use scipy optimization to find optimal (τ_prior, τ_softmax, ε) maximizing combo_r."""
    def objective(params):
        tp, ts, eps = params
        res = evaluate(records, tp, ts, prior_type=prior_type, epsilon=eps)
        return -res["combo_r"]

    x0 = [
        (tau_prior_bounds[0] + tau_prior_bounds[1]) / 2,
        (tau_softmax_bounds[0] + tau_softmax_bounds[1]) / 2,
        (eps_bounds[0] + eps_bounds[1]) / 2,
    ]
    bounds = [tau_prior_bounds, tau_softmax_bounds, eps_bounds]

    print(f"\nScipy optimization (maximizing combo_r) ...")
    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 50, "ftol": 1e-6})
    tp, ts, eps = result.x
    print(f"  Optimal τ_prior={tp:.4f} τ_softmax={ts:.4f} ε={eps:.6f}  (objective={-result.fun:.4f})")
    return tp, ts, eps


def pick_best(results):
    return max(results, key=lambda r: r["combo_r"] if not np.isnan(r["combo_r"]) else -np.inf)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune τ_prior, τ_softmax, and ε parameters")
    parser.add_argument("--prior", choices=["utility", "uniform"], default="utility",
                        help="Prior type: 'utility' (stat-based) or 'uniform' (flat 1/27)")

    parser.add_argument("--tau-prior-min", type=float, default=0.1, help="Min τ_prior (default: 0.1)")
    parser.add_argument("--tau-prior-max", type=float, default=5.0, help="Max τ_prior (default: 5.0)")
    parser.add_argument("--tau-prior-steps", type=int, default=20, help="Number of τ_prior grid points (default: 20)")

    parser.add_argument("--tau-softmax-min", type=float, default=0.1, help="Min τ_softmax (default: 0.1)")
    parser.add_argument("--tau-softmax-max", type=float, default=5.0, help="Max τ_softmax (default: 5.0)")
    parser.add_argument("--tau-softmax-steps", type=int, default=20, help="Number of τ_softmax grid points (default: 20)")

    parser.add_argument("--eps-min", type=float, default=0.001, help="Min ε (default: 0.001)")
    parser.add_argument("--eps-max", type=float, default=0.2, help="Max ε (default: 0.2)")
    parser.add_argument("--eps-steps", type=int, default=10, help="Number of ε grid points (default: 10)")

    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--data-dir", type=str, nargs="+", default=None,
                        help="Path(s) to data directory (can specify multiple)")
    parser.add_argument("--plot-dir", type=str, default=None,
                        help="Data dir to use for plotting (default: last --data-dir)")
    args = parser.parse_args()

    print("Loading human data ...")
    data_dirs = args.data_dir if args.data_dir else [str(d) for d in DEFAULT_DATA_DIRS]
    records = load_team_rounds(data_dirs=data_dirs)
    n_envs = len(set(r["env_id"] for r in records))
    print(f"Loaded {len(records)} team-rounds across {n_envs} envs\n")

    tau_prior_values = np.linspace(args.tau_prior_min, args.tau_prior_max, args.tau_prior_steps)
    tau_softmax_values = np.linspace(args.tau_softmax_min, args.tau_softmax_max, args.tau_softmax_steps)
    eps_values = np.geomspace(args.eps_min, args.eps_max, args.eps_steps)

    total = args.tau_prior_steps * args.tau_softmax_steps * args.eps_steps
    print(f"Coarse grid: {args.tau_prior_steps} τ_prior x {args.tau_softmax_steps} τ_softmax x {args.eps_steps} ε = {total} points")
    print(f"  τ_prior in [{args.tau_prior_min}, {args.tau_prior_max}]")
    print(f"  τ_softmax in [{args.tau_softmax_min}, {args.tau_softmax_max}]")
    print(f"  ε in [{args.eps_min}, {args.eps_max}]")
    print(f"  prior = {args.prior}\n")

    # Step 1: Coarse grid search
    all_results = grid_search(records, tau_prior_values, tau_softmax_values, eps_values, prior_type=args.prior)
    best_result = pick_best(all_results)

    # Step 2: Refine around best with finer grid
    tp_step = (args.tau_prior_max - args.tau_prior_min) / args.tau_prior_steps
    ts_step = (args.tau_softmax_max - args.tau_softmax_min) / args.tau_softmax_steps
    eps_ratio = (args.eps_max / args.eps_min) ** (1.0 / args.eps_steps)

    fine_tau_prior = np.linspace(
        max(args.tau_prior_min, best_result["tau_prior"] - tp_step),
        min(args.tau_prior_max, best_result["tau_prior"] + tp_step),
        10,
    )
    fine_tau_softmax = np.linspace(
        max(args.tau_softmax_min, best_result["tau_softmax"] - ts_step),
        min(args.tau_softmax_max, best_result["tau_softmax"] + ts_step),
        10,
    )
    fine_eps = np.geomspace(
        max(args.eps_min, best_result["epsilon"] / eps_ratio),
        min(args.eps_max, best_result["epsilon"] * eps_ratio),
        10,
    )
    print(f"\nRefining around τ_prior={best_result['tau_prior']:.4f} "
          f"τ_softmax={best_result['tau_softmax']:.4f} ε={best_result['epsilon']:.4f} ...")
    refined = grid_search(records, fine_tau_prior, fine_tau_softmax, fine_eps, prior_type=args.prior)
    all_results.extend(refined)
    best_result = pick_best(all_results)

    # Step 3: Scipy polishing
    tp_lo = max(args.tau_prior_min, best_result["tau_prior"] - tp_step / 2)
    tp_hi = min(args.tau_prior_max, best_result["tau_prior"] + tp_step / 2)
    ts_lo = max(args.tau_softmax_min, best_result["tau_softmax"] - ts_step / 2)
    ts_hi = min(args.tau_softmax_max, best_result["tau_softmax"] + ts_step / 2)
    eps_lo = max(1e-6, best_result["epsilon"] / 2)
    eps_hi = min(1.0, best_result["epsilon"] * 2)
    opt_tp, opt_ts, opt_eps = optimize_scipy(
        records, (tp_lo, tp_hi), (ts_lo, ts_hi), (eps_lo, eps_hi),
        prior_type=args.prior,
    )
    opt_result = evaluate(records, opt_tp, opt_ts, prior_type=args.prior, epsilon=opt_eps)
    all_results.append(opt_result)
    best_result = pick_best(all_results)

    # Baseline comparison (τ_prior=1.0, τ_softmax=1.0, ε=1e-10)
    baseline = evaluate(records, 1.0, 1.0, prior_type=args.prior)

    # Summary
    print(f"\n{'='*60}")
    print(f"Prior: {args.prior}")
    print(f"BEST τ_prior = {best_result['tau_prior']:.4f}, τ_softmax = {best_result['tau_softmax']:.4f}, ε = {best_result['epsilon']:.6f}")
    print(f"  combo_r  = {best_result['combo_r']:.4f}")
    print(f"  marg_r   = {best_result['marg_r']:.4f}")
    print(f"{'='*60}")
    print(f"\nComparison with defaults (τ_prior=1.0, τ_softmax=1.0, ε=1e-10):")
    print(f"  {'metric':<10} {'default':>10} {'best':>10} {'delta':>10}")
    print(f"  {'combo_r':<10} {baseline['combo_r']:>10.4f} {best_result['combo_r']:>10.4f} {best_result['combo_r'] - baseline['combo_r']:>+10.4f}")
    print(f"  {'marg_r':<10} {baseline['marg_r']:>10.4f} {best_result['marg_r']:>10.4f} {best_result['marg_r'] - baseline['marg_r']:>+10.4f}")

    # Per-environment best correlations
    best_corrs = best_result["correlations"]
    print(f"\nPer-environment correlations at best "
          f"(τ_prior={best_result['tau_prior']:.4f}, τ_softmax={best_result['tau_softmax']:.4f}, ε={best_result['epsilon']:.6f}):")
    for env_id in sorted(k for k in best_corrs if k != "__global__"):
        c = best_corrs[env_id]
        combo = c.get("combo", {})
        marg = c.get("marginal", {})
        parts = []
        if combo:
            parts.append(f"combo_r={combo['r']:.4f}")
        if marg:
            parts.append(f"marg_r={marg['r']:.4f}")
        print(f"  {env_id}: {', '.join(parts)}")
    g = best_corrs.get("__global__", {})
    if g:
        print(f"  GLOBAL: combo_r={g.get('combo', {}).get('r', float('nan')):.4f}, "
              f"marg_r={g.get('marginal', {}).get('r', float('nan')):.4f}")

    # Plot with best params
    plot_dir = args.plot_dir
    if plot_dir is None and args.data_dir:
        plot_dir = args.data_dir[-1]
    plot_records = load_team_rounds(data_dir=plot_dir) if plot_dir else records
    print(f"\nGenerating plots with best τ_prior={best_result['tau_prior']:.4f}, "
          f"τ_softmax={best_result['tau_softmax']:.4f}, ε={best_result['epsilon']:.6f} ...")
    if plot_dir:
        print(f"  (plotting data from: {plot_dir})")
    best_predictions = run_all_predictions(
        plot_records,
        tau_prior=best_result["tau_prior"],
        tau_softmax=best_result["tau_softmax"],
        prior_type=args.prior,
        epsilon=best_result["epsilon"],
    )
    plot_corrs = compute_pearson(best_predictions)
    plot_comparison(best_predictions, plot_corrs,
                    tau_prior=best_result["tau_prior"],
                    tau_softmax=best_result["tau_softmax"],
                    epsilon=best_result["epsilon"])

    # Save results
    output = {
        "prior_type": args.prior,
        "best_tau_prior": best_result["tau_prior"],
        "best_tau_softmax": best_result["tau_softmax"],
        "best_epsilon": best_result["epsilon"],
        "best_combo_r": best_result["combo_r"],
        "best_marg_r": best_result["marg_r"],
        "sweep": [
            {
                "tau_prior": r["tau_prior"],
                "tau_softmax": r["tau_softmax"],
                "epsilon": r["epsilon"],
                "combo_r": r["combo_r"],
                "marg_r": r["marg_r"],
            }
            for r in sorted(all_results, key=lambda r: (r["tau_prior"], r["tau_softmax"], r["epsilon"]))
        ],
        "best_correlations": best_result["correlations"],
    }

    out_path = Path(args.output) if args.output else SCRIPT_DIR / "finetune_tau_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
