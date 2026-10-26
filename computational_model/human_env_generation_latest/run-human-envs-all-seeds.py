"""
Run new-human-envs-1.py and new-human-envs-2.py for seeds 1..50,
then aggregate: for each role combo, collect env IDs that qualify
across ALL seeds.
"""

import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SEEDS = range(1, 101)
RESULTS_DIR = SCRIPT_DIR / "output" / "seed_results_for_env"
FINAL_OUTPUT = SCRIPT_DIR / "output" / "okay_envs_by_role_all_seeds.txt"
FINAL_LDS_OUTPUT = SCRIPT_DIR / "output" / "human_lds_seqs_all_seeds.txt"


def run_seed(seed):
    turn3_file = RESULTS_DIR / f"turn3_seed{seed}.json"
    okay_file = RESULTS_DIR / f"okay_seed{seed}.json"
    lds_file = RESULTS_DIR / f"lds_seed{seed}.json"

    r1 = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "new-human-envs-1.py"), str(seed), str(turn3_file), str(lds_file)],
        capture_output=True, text=True,
    )
    if r1.returncode != 0:
        print(f"  [seed {seed}] Script 1 FAILED:\n{r1.stderr[-500:]}")
        return seed, None, None

    r2 = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "new-human-envs-2.py"), str(turn3_file), str(okay_file)],
        capture_output=True, text=True,
    )
    if r2.returncode != 0:
        print(f"  [seed {seed}] Script 2 FAILED:\n{r2.stderr[-500:]}")
        return seed, None, None

    with open(okay_file) as f:
        okay = json.load(f)
    with open(lds_file) as f:
        lds = json.load(f)
    return seed, okay, lds


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    all_results = {}
    all_lds = {}  # seed -> {env_id -> {env_id, enemy_prob, lds}}
    total = len(SEEDS)
    started = time.perf_counter()

    for i, seed in enumerate(SEEDS, 1):
        print(f"[{i}/{total}] Running seed {seed}...")
        seed_val, result, lds = run_seed(seed)
        if result is not None:
            all_results[seed_val] = result
            all_lds[seed_val] = lds

        elapsed = time.perf_counter() - started
        rate = i / elapsed if elapsed > 0 else 0
        eta = (total - i) / rate if rate > 0 else 0
        print(f"  Done. ({elapsed:.1f}s elapsed, ~{eta:.0f}s remaining)")

    if not all_results:
        print("No successful seeds.")
        return

    # Aggregate: for each role, union of qualifying envs across seeds
    all_roles = sorted(next(iter(all_results.values())).keys())
    envs_by_role_per_seed = defaultdict(list)

    for seed, result in all_results.items():
        for role, envs in result.items():
            envs_by_role_per_seed[role].append(set(envs))

    aggregated = {}
    for role in all_roles:
        sets = envs_by_role_per_seed[role]
        if sets:
            combined = sets[0].union(*sets[1:])
            aggregated[role] = sorted(combined, key=lambda x: int(x) if x.isdigit() else x)
        else:
            aggregated[role] = []

    with open(FINAL_OUTPUT, "w") as f:
        json.dump(aggregated, f, indent=2, sort_keys=True)

    # Collect LDS for qualifying envs: env_id -> {seed -> lds_binary, enemy_prob}
    qualifying_env_ids = set()
    for envs in aggregated.values():
        qualifying_env_ids.update(envs)

    lds_output = {}
    for env_id in sorted(qualifying_env_ids, key=lambda x: int(x) if x.isdigit() else x):
        env_entry = {}
        for seed, lds_dict in all_lds.items():
            if env_id in lds_dict:
                info = lds_dict[env_id]
                env_entry["enemy_prob"] = info["enemy_prob"]
                env_entry.setdefault("seeds", {})[str(seed)] = info["lds"]
        if env_entry:
            lds_output[env_id] = env_entry

    with open(FINAL_LDS_OUTPUT, "w") as f:
        json.dump(lds_output, f, indent=2, sort_keys=True)

    non_empty = {r: e for r, e in aggregated.items() if e}
    total_envs = sum(len(e) for e in aggregated.values())
    print(f"\nAggregated across {len(all_results)} seeds:")
    print(f"  Roles with qualifying envs: {len(non_empty)}/27")
    print(f"  Total env-role pairs: {total_envs}")
    print(f"  Qualifying envs with LDS: {len(lds_output)}")
    print(f"  Saved to {FINAL_OUTPUT}")
    print(f"  LDS saved to {FINAL_LDS_OUTPUT}")


if __name__ == "__main__":
    main()
