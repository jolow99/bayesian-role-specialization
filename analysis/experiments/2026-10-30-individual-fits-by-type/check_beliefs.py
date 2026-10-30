"""Check: the mixed beliefs individual_fits.py gives the choice models (05-25
precompute_trajectories at the classifier's τ_prior, ε, β) equal the classifier's
own beliefs, (1 − β)·P_N + β·P_0 from 2026-10-27-inference-soft-classifier/pipeline.py,
at every stage of every clean human team-round.

Run from analysis/:
    uv run python experiments/2026-10-30-individual-fits-by-type/check_beliefs.py
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
EXP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP / "2026-10-30-individual-fits-by-type"))
import fit_helpers as fh
sys.path.insert(0, str(EXP / "2026-06-05-paper-figures-v2"))
from common import load_clean_human_teams, prepare_team
from shared.inference import utility_based_prior
spec = importlib.util.spec_from_file_location("clf_pipeline", EXP / "2026-10-27-inference-soft-classifier/pipeline.py")
cp = importlib.util.module_from_spec(spec); sys.modules["clf_pipeline"] = cp; spec.loader.exec_module(cp)

clf = json.load(open(EXP / "2026-10-27-inference-soft-classifier/best_params.json"))
b = clf["beta"]
records = fh.pf.load_records_with_pids(verbose=False)
trajs = fh.precompute_trajectories(records, clf["tau_prior"], clf["epsilon"],
                                   memory_strategy=fh.strategy_from_params("full", None, b))
teams = {(k[1], int(k[2])): v for k, v in load_clean_human_teams(verbose=False).items()}
strat = cp.MemoryStrategy("x", "drift_prior", b)
worst, n_stage, n_rec = 0.0, 0, 0
for rec, traj in zip(records, trajs):
    data = prepare_team(teams[(rec["game_id"], int(rec["round_number"]))])
    assert np.array_equal(np.asarray(data["player_stats"], float), np.asarray(rec["env_config"]["player_stats"], float))
    P = cp.compute_posteriors(data, clf["tau_prior"], clf["epsilon"], strat)
    P0 = utility_based_prior(data["player_stats"], tau=clf["tau_prior"])
    assert len(P) >= len(traj), (len(P), len(traj))
    for s, st in enumerate(traj):
        B = P0 if s == 0 else (1 - b) * P[s] + b * P0
        worst = max(worst, float(np.abs(st["prior"] - B).max())); n_stage += 1
    n_rec += 1
assert worst < 1e-12, worst
print(f"{n_rec} team-rounds, {n_stage} stages, max |05-25 belief - classifier mixed belief| = {worst:.2e}")
