"""Fig. 4 + SI team cases (R3) re-rendered with the new fits.

Renderer: the paper's, 2026-10-15-fig4-legend-options/final6pt.py, with its inputs swapped:
  model strip  Bayesian-Walk-BR at the updaters' fit (response-fitting/fit-results/updaters)
               on the mixed belief (1 − β)·P_N + β·P_0 at the inference fit's τ, ε, β
  belief rows  unmixed P_N at the same τ, ε (what inference reports are scored against)
Writes figures/<name>.{png,pdf} and chosen_probs.md (probability on each chosen role).

Run from analysis/:
    uv run python experiments/2026-11-02_ftm-prior-fitting/results/stage-by-stage/stage_by_stage.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
FIT_DIR = OUT_DIR.parents[1]
sys.path.insert(0, str(FIT_DIR.parent / "2026-10-15-fig4-legend-options"))

import final6pt as f6  # noqa: E402

tm, lo = f6.tm, f6.lo


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


inference = _load("inference_pipeline_1102", FIT_DIR / "inference-fitting" / "pipeline.py")
PARAMS = json.load(open(FIT_DIR / "inference-fitting" / "params.json"))
WALK_BR = json.load(open(FIT_DIR / "response-fitting" / "fit-results" / "updaters" /
                         "bayes-models.json"))["Bayesian-Walk-BR"]["params"]
S1 = {"tau_prior": PARAMS["tau_prior"], "epsilon": PARAMS["epsilon"],
      "memory_strategy": f"drift_prior_{PARAMS['beta']}"}

_orig_params, _orig_s1 = tm.load_best_model_params, tm.load_stage1_canonical
tm.load_best_model_params = lambda: (WALK_BR, S1, _orig_params()[2])
tm.load_stage1_canonical = lambda: (S1, _orig_s1()[1])
tm.human_posteriors = lambda rec, s1, strat: inference.compute_posteriors(
    rec, s1["tau_prior"], s1["epsilon"], PARAMS["beta"])


def main():
    FIG_DIR.mkdir(exist_ok=True)
    f6.HERE = FIG_DIR
    lines = ["# Probability on each chosen role (new fits)", "",
             f"Walk-BR τ_v = {WALK_BR['tau_softmax']:.2f}, ε_s = {1 - WALK_BR['epsilon_switch']:.2f}; "
             f"beliefs τ = {PARAMS['tau_prior']:.2f}, ε = {PARAMS['epsilon']:.3f}, β = {PARAMS['beta']:.3f}.", ""]
    for suffix, rnd, name in f6.CASES:
        rec, post, preds = lo.prepare_case(suffix, rnd)
        f6.render6(name, rec, post, preds, f6.S09 if name == "R3_team_case" else f6.S10)
        p = np.array([[pr["pred"][i]["FTM".index(pr["combo"][i]) if isinstance(pr["combo"][i], str)
                                     else int(pr["combo"][i])] for i in range(3)] for pr in preds])
        lines += [f"## {name} (…{suffix}, round {rnd})", "",
                  "| Stage | P1 | P2 | P3 | mean |", "|--:|--:|--:|--:|--:|"]
        lines += [f"| {s + 1} | " + " | ".join(f"{x:.2f}" for x in row) + f" | {row.mean():.2f} |"
                  for s, row in enumerate(p)]
        lines += [f"| all | | | | {p.mean():.2f} |", ""]
    (OUT_DIR / "chosen_probs.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
