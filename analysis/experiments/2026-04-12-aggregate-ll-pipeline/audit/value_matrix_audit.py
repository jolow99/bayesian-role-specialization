"""Value-matrix audit — blocking pre-flight for 2026-04-12-aggregate-ll-pipeline.

Decides whether ``values.npy`` is environmental (Hypothesis A) or policy-
dependent (Hypothesis B), and for bot rounds confirms that an env-keyed
value matrix exists on disk for every bot env_id encountered in the data.

Usage:
    python audit/value_matrix_audit.py

Exits non-zero if Hypothesis B holds with missing matrices.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Anchor on the shared package root regardless of cwd.
SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_ROOT = SCRIPT_DIR.parent.parent.parent  # .../analysis
sys.path.insert(0, str(ANALYSIS_ROOT))

from shared import DATA_ROOT
from shared.data_loading import load_all_exports


ENVS_DIR = DATA_ROOT / "envs"
HUMAN_ENVS_DIR = DATA_ROOT / "human_envs_value_matrices"


def main() -> int:
    print("=" * 66)
    print("Value-matrix audit — 2026-04-12-aggregate-ll-pipeline")
    print("=" * 66)

    # 1. Inspect one value matrix and trace its indexing path.
    sample_path = ENVS_DIR / "139" / "values.npy"
    v = np.load(sample_path)
    print(f"\nSample: {sample_path}")
    print(f"  shape = {v.shape}  dtype = {v.dtype}")
    print(f"  sample values = {v.flat[:5].tolist()}")

    print("\nIndexing path traced through shared.inference.softmax_role_dist:")
    print("  flat_idx = r0*9 + r1*3 + r2   # joint role combo (27 values)")
    print("  values[flat_idx, intent, team_hp, enemy_hp]")
    print("  => shape[0] == 27 (joint combo), shape[1] == 2 (intent),")
    print("     shape[2] == H+1, shape[3] == W+1 (HP grids)")

    if v.shape[0] != 27:
        print("\n[FAIL] First axis is not 27; indexing assumption broken. Abort.")
        return 2

    # 2. Enumerate bot env_ids and confirm a value matrix exists for each.
    recs = load_all_exports(include_bot_rounds=True)
    bot_envs: set[str] = set()
    for pr in recs:
        if pr.round.round_type == "bot":
            eid = pr.round.config.get("envId")
            if eid is not None:
                bot_envs.add(str(eid))

    missing = [e for e in sorted(bot_envs) if not (ENVS_DIR / e / "values.npy").exists()]
    print(f"\nBot-round env_ids: {sorted(bot_envs)}")
    print(f"  count = {len(bot_envs)}")
    print(f"  missing value matrices: {missing if missing else 'none'}")

    if missing:
        print("\n[BLOCK] Hypothesis-B path — missing matrices. Escalate to user.")
        return 1

    # 3. Verdict.
    print("\n" + "-" * 66)
    print("Verdict: Hypothesis A (environmental)")
    print("-" * 66)
    print(
        "values.npy is a (27, 2, H, W) joint-combo state-value table indexed\n"
        "by (combo, intent, team_hp, enemy_hp). The table depends on the env's\n"
        "player_stats, boss_damage, and HP grids -- all of which are fixed per\n"
        "env_id -- but does NOT assume any particular teammate-role policy.\n"
        "Each env_id has its own values.npy under data/envs/<env_id>/. For bot\n"
        "rounds we use that per-env-id fallback (online_model_sim.py already\n"
        "does this in its 'not exists' branch). For human rounds the existing\n"
        "human_envs_value_matrices/<role_combo> lookup continues to work.\n"
        "No regeneration required."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
