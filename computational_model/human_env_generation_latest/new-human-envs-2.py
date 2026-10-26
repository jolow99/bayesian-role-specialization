import json
import os
import ast
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_TURN3_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else SCRIPT_DIR / "output" / "turn3_states_by_env_role.txt"
OUTPUT_BY_ROLE = Path(sys.argv[2]) if len(sys.argv) > 2 else SCRIPT_DIR / "output" / "okay_envs_by_role.txt"
ENVS_DIR = SCRIPT_DIR.parent / "envs"

# A canonical role class must be uniquely optimal in >50% of turn-3 states.
# Symmetric ties from identical player stat rows count as the same class.
TIE_TOL = 1e-9

MAX_WORKERS = max(1, (os.cpu_count() or 1) - 1)
PROGRESS_EVERY = 100


ROLE_CHARS = ["F", "T", "M"]
CHAR_TO_INT = {"F": 0, "T": 1, "M": 2}
ALL_ROLE_COMBOS = [
    a + b + c
    for a in ROLE_CHARS
    for b in ROLE_CHARS
    for c in ROLE_CHARS
]


def role_combo_to_index(role_combo):
    r0 = CHAR_TO_INT[role_combo[0]]
    r1 = CHAR_TO_INT[role_combo[1]]
    r2 = CHAR_TO_INT[role_combo[2]]
    return r0 * 9 + r1 * 3 + r2


def canonical_role_combo(role_combo, player_stats):
    canonical = list(role_combo)
    stat_groups = defaultdict(list)

    for i, row in enumerate(np.asarray(player_stats).tolist()):
        stat_groups[tuple(float(x) for x in row)].append(i)

    for indices in stat_groups.values():
        if len(indices) <= 1:
            continue
        sorted_roles = sorted((canonical[i] for i in indices), key=lambda r: CHAR_TO_INT[r])
        for i, role in zip(indices, sorted_roles):
            canonical[i] = role

    return "".join(canonical)


def build_role_class_lookup(player_stats):
    return [
        (role_combo_to_index(role_combo), canonical_role_combo(role_combo, player_stats))
        for role_combo in ALL_ROLE_COMBOS
    ]


def canonical_role_combos(player_stats):
    return sorted(
        {canonical_role_combo(role_combo, player_stats) for role_combo in ALL_ROLE_COMBOS},
        key=role_combo_to_index,
    )


def load_turn3_dict(path):
    with open(path, "r") as f:
        return json.load(f)


def group_states_by_env(turn3_dict):
    """
    Converts flat dict key 'envid_ROLE' -> state into:
    env_to_states[envid][ROLE] = state
    """
    env_to_states = defaultdict(dict)
    for key, state in turn3_dict.items():
        try:
            env_id, role_combo = key.rsplit("_", 1)
        except ValueError:
            continue
        if len(role_combo) != 3:
            continue
        env_to_states[env_id][role_combo] = state
    return env_to_states


def load_values_for_env(env_id):
    values_path = ENVS_DIR / env_id / "values.npy"
    if not values_path.exists():
        return None
    return np.load(values_path)


def load_player_stats_for_env(env_id):
    config_path = ENVS_DIR / env_id / "config.py"
    if not config_path.exists():
        return None

    text = config_path.read_text()
    match = re.search(r"PLAYER_STATS\s*=\s*jnp\.array\((\[.*?\])\)", text, re.S)
    if match is None:
        return None

    return np.array(ast.literal_eval(match.group(1)), dtype=float)


def unique_best_role_class(values, intent, team_hp, enemy_hp, role_class_lookup, tie_tol=TIE_TOL):
    """
    Returns the unique best canonical role class for a state, or None if
    different role classes tie. Ties inside a class are symmetry ties.
    """
    role_values = values[:, intent, team_hp, enemy_hp]
    class_values = {}

    for role_idx, role_class in role_class_lookup:
        value = float(role_values[role_idx])
        if role_class not in class_values or value > class_values[role_class]:
            class_values[role_class] = value

    max_val = max(class_values.values())
    best_classes = [
        role_class
        for role_class, value in class_values.items()
        if np.isclose(value, max_val, atol=tie_tol, rtol=0.0)
    ]

    if len(best_classes) != 1:
        return None
    return best_classes[0]


def evaluate_env(env_id, states_for_env):
    scenario_count = len(states_for_env)
    if scenario_count <= 0:
        return env_id, []

    # Skip env if any of the 27 role combos has team dead after 2 turns
    for state in states_for_env.values():
        if int(round(state["team_hp"])) <= 0:
            return env_id, []

    values = load_values_for_env(env_id)
    if values is None:
        return env_id, []
    player_stats = load_player_stats_for_env(env_id)
    if player_stats is None:
        return env_id, []

    required_count = (scenario_count // 2)
    candidate_roles = canonical_role_combos(player_stats)
    role_class_lookup = build_role_class_lookup(player_stats)
    counts = {role: 0 for role in candidate_roles}

    for scenario_r in ALL_ROLE_COMBOS:
        state = states_for_env.get(scenario_r)
        if state is None:
            continue

        intent = int(state["intent"])
        team_hp = int(round(state["team_hp"]))
        enemy_hp = int(round(state["enemy_hp"]))

        team_hp = max(0, min(team_hp, values.shape[2] - 1))
        enemy_hp = max(0, min(enemy_hp, values.shape[3] - 1))
        intent = max(0, min(intent, values.shape[1] - 1))

        best_class = unique_best_role_class(values, intent, team_hp, enemy_hp, role_class_lookup)
        if best_class is not None:
            counts[best_class] += 1

    qualifying_roles = [
        role
        for role, count in counts.items()
        if count > required_count
    ]

    return env_id, qualifying_roles


def main():
    if not INPUT_TURN3_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_TURN3_FILE}")

    turn3_dict = load_turn3_dict(INPUT_TURN3_FILE)
    env_to_states = group_states_by_env(turn3_dict)

    # canonical role class -> env IDs where it is uniquely optimal in >=14 of 27 turn-3 states
    okay_envs_by_role = {r: [] for r in ALL_ROLE_COMBOS}

    env_ids = sorted(env_to_states.keys(), key=lambda x: int(x) if x.isdigit() else x)

    total_envs = len(env_ids)
    print(f"Evaluating {total_envs} environments (workers={MAX_WORKERS})...")
    started_at = time.perf_counter()
    completed = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(evaluate_env, env_id, env_to_states[env_id]): env_id
            for env_id in env_ids
        }

        for future in as_completed(futures):
            env_id, qualifying_roles = future.result()
            for role in qualifying_roles:
                okay_envs_by_role[role].append(env_id)

            completed += 1
            if completed % PROGRESS_EVERY == 0 or completed == total_envs:
                elapsed = time.perf_counter() - started_at
                rate = completed / elapsed if elapsed > 0 else 0.0
                pct = 100.0 * completed / total_envs if total_envs else 100.0
                print(f"Progress: {completed}/{total_envs} ({pct:.1f}%) | elapsed={elapsed:.1f}s | rate={rate:.2f} env/s")

    # Save as JSON text files for easy reuse
    with open(OUTPUT_BY_ROLE, "w") as f:
        f.write(json.dumps(okay_envs_by_role, indent=2, sort_keys=True))

    print(f"Saved role->env list to {OUTPUT_BY_ROLE}")

    non_empty_roles = {r: envs for r, envs in okay_envs_by_role.items() if envs}
    print(f"Roles with at least one qualifying env: {len(non_empty_roles)}/27")
    if non_empty_roles:
        print("Role combos:")
        print(", ".join(sorted(non_empty_roles.keys())))


if __name__ == "__main__":
    main()
