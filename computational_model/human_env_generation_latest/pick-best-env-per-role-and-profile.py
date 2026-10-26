"""
For each (canonical_role_class, stat_profile) bucket, pick the (env, seed)
where that class is uniquely optimal in the most turn-3 states out of 27.

Difference from pick-best-env-per-role.py: candidates are bucketed by
stat profile, so the same role combo can produce multiple picks — one
per profile. This fills more cells of the 5-profile x N-combo table.

Requirements per (env, seed):
  - Target role wins the full game with that LDS
  - All 27 turn-3 states have team_hp > 0
  - Target canonical role class is uniquely optimal in as many of the 27 states as possible

Outputs picked_envs_by_role_and_profile.txt and
'updated human envs by profile/PROFILE/COMBO/'.
"""

import ast
import json
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import qmc

SCRIPT_DIR = Path(__file__).resolve().parent
ENVS_DIR = SCRIPT_DIR.parent / "envs"
SEED_DIR = SCRIPT_DIR / "output" / "seed_results_for_env"
QUALIFYING_FILE = SCRIPT_DIR / "output" / "okay_envs_by_role_all_seeds.txt"
OUT_DIR = SCRIPT_DIR / "output" / "updated human envs by profile"
SEEDS = range(1, 101)
NUM_TURNS = 2

F, T, M = 0, 1, 2
ATTACK, DEFEND, HEAL = 0, 1, 2
CHAR_TO_INT = {"F": 0, "T": 1, "M": 2}
ROLE_CHARS = {0: "F", 1: "T", 2: "M"}

ALL_ROLE_COMBOS_STR = [
    a + b + c for a in "FTM" for b in "FTM" for c in "FTM"
]

TIE_TOL = 1e-9


def role_combo_index(s):
    return CHAR_TO_INT[s[0]] * 9 + CHAR_TO_INT[s[1]] * 3 + CHAR_TO_INT[s[2]]


def str_to_tuple(s):
    return tuple(CHAR_TO_INT[c] for c in s)


def load_config(env_id):
    p = ENVS_DIR / env_id / "config.py"
    text = p.read_text()
    stats_match = re.search(r"PLAYER_STATS\s*=\s*jnp\.array\((\[.*?\])\)", text, re.S)
    if stats_match is None:
        raise ValueError(f"Could not parse PLAYER_STATS from {p}")

    def parse_literal(name):
        match = re.search(rf"^{name}\s*=\s*([^\n]+)", text, re.M)
        if match is None:
            raise ValueError(f"Could not parse {name} from {p}")
        return ast.literal_eval(match.group(1).strip())

    horizon_match = re.search(r"EPSILON,\s*HORIZON\s*=\s*[^,]+,\s*(\d+)", text)
    return {
        "player_stats": np.array(ast.literal_eval(stats_match.group(1)), dtype=float),
        "boss_damage": float(parse_literal("BOSS_DAMAGE")),
        "team_max_hp": int(parse_literal("TEAM_MAX_HP")),
        "enemy_max_hp": int(parse_literal("ENEMY_MAX_HP")),
        "enemy_prob": float(parse_literal("ENEMY_ATTACK_PROB")),
        "horizon": int(horizon_match.group(1)) if horizon_match else 10,
    }


def generate_lds(horizon, prob, seed):
    if horizon <= 0:
        return np.array([], dtype=int)
    if horizon == 1:
        return np.array([1], dtype=int)
    engine = qmc.Sobol(d=1, scramble=True, seed=seed)
    samples = engine.random(horizon - 1).flatten()
    threshold = np.clip((prob * horizon - 1) / (horizon - 1), 0.0, 1.0)
    tail = (samples < threshold).astype(int)
    return np.concatenate(([1], tail))


def get_action(role, intent, team_hp, team_max_hp):
    if role == F:
        return ATTACK
    if role == T:
        return DEFEND if intent == 1 else ATTACK
    return HEAL if team_hp < team_max_hp else ATTACK


def simulate_full_game(cfg, lds, role_combo):
    """Returns True if enemy dies and team is still alive."""
    ps = cfg["player_stats"]
    team_hp = float(cfg["team_max_hp"])
    enemy_hp = float(cfg["enemy_max_hp"])
    team_max = cfg["team_max_hp"]

    for step in range(len(lds)):
        intent = int(lds[step])
        actions = [get_action(role_combo[i], intent, team_hp, team_max) for i in range(3)]
        total_dmg = sum(ps[i, ATTACK] for i in range(3) if actions[i] == ATTACK)
        total_heal = sum(ps[i, HEAL] for i in range(3) if actions[i] == HEAL)
        defenders = [ps[i, DEFEND] for i in range(3) if actions[i] == DEFEND]
        max_def = max(defenders) if defenders else 0.0

        enemy_hp = max(0.0, enemy_hp - total_dmg)
        dmg_taken = max(0.0, cfg["boss_damage"] - max_def) if intent == 1 else 0.0
        team_hp = max(0.0, min(float(team_max), team_hp - dmg_taken + total_heal))

        if team_hp <= 0:
            return False
        if enemy_hp <= 0:
            return True
    return False


def stat_profile_str(player_stats):
    """Canonical profile string like '411_222_222'."""
    return "_".join("".join(str(int(x)) for x in row) for row in player_stats)


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
        (role_combo_index(role_combo), canonical_role_combo(role_combo, player_stats))
        for role_combo in ALL_ROLE_COMBOS_STR
    ]


def unique_best_role_class(values, intent, team_hp, enemy_hp, role_class_lookup):
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
        if np.isclose(value, max_val, atol=TIE_TOL, rtol=0.0)
    ]

    if len(best_classes) != 1:
        return None
    return best_classes[0]


def main():
    with open(QUALIFYING_FILE) as f:
        qualifying = json.load(f)
    non_empty = {r: set(envs) for r, envs in qualifying.items() if envs}

    # Uniqueness: prefer envs that only qualify for one role
    env_to_roles = defaultdict(set)
    for role, envs in non_empty.items():
        for e in envs:
            env_to_roles[e].add(role)

    print(f"Qualifying roles: {sorted(non_empty.keys())}")
    print(f"Seeds: {SEEDS.start}-{SEEDS.stop - 1}")
    print()

    # For each (canonical role class, stat_profile) bucket, track best (count, env, seed)
    best = {}  # (role, profile) -> (count, env_id, seed, is_unique_env, n_states)
    # Cache env configs to avoid re-parsing
    cfg_cache = {}
    started = time.perf_counter()

    for seed in SEEDS:
        turn3_path = SEED_DIR / f"turn3_seed{seed}.json"
        lds_path = SEED_DIR / f"lds_seed{seed}.json"
        if not turn3_path.exists() or not lds_path.exists():
            continue

        with open(turn3_path) as f:
            turn3 = json.load(f)
        with open(lds_path) as f:
            lds_data = json.load(f)

        # Group turn3 states by env
        env_states = defaultdict(dict)
        for key, state in turn3.items():
            env_id, role_s = key.rsplit("_", 1)
            env_states[env_id][role_s] = state

        for role_s, candidate_envs in non_empty.items():
            for env_id in candidate_envs:
                states = env_states.get(env_id, {})
                if not states:
                    continue

                # All 27 must have team alive
                all_alive = all(int(round(s["team_hp"])) > 0 for s in states.values())
                if not all_alive:
                    continue

                if env_id not in cfg_cache:
                    cfg_cache[env_id] = load_config(env_id)
                cfg = cfg_cache[env_id]
                target_role = canonical_role_combo(role_s, cfg["player_stats"])
                target_tuple = str_to_tuple(target_role)
                role_class_lookup = build_role_class_lookup(cfg["player_stats"])

                # Load values
                values_path = ENVS_DIR / env_id / "values.npy"
                if not values_path.exists():
                    continue
                values = np.load(values_path)

                # Count states where the target canonical role class is uniquely optimal
                count = 0
                for state in states.values():
                    intent = int(state["intent"])
                    thp = max(0, min(int(round(state["team_hp"])), values.shape[2] - 1))
                    ehp = max(0, min(int(round(state["enemy_hp"])), values.shape[3] - 1))
                    intent = max(0, min(intent, values.shape[1] - 1))
                    if unique_best_role_class(values, intent, thp, ehp, role_class_lookup) == target_role:
                        count += 1

                # Check if target role wins full game with this LDS
                if count > 0 and env_id in lds_data:
                    lds_str = lds_data[env_id]["lds"]
                    lds = np.array([int(c) for c in lds_str])
                    if not simulate_full_game(cfg, lds, target_tuple):
                        continue

                    profile = stat_profile_str(cfg["player_stats"])
                    bucket_key = (target_role, profile)

                    prev = best.get(bucket_key)
                    is_unique_env = len(env_to_roles[env_id]) == 1
                    prev_unique = prev[3] if prev else False

                    n_states = len(states)
                    # Prefer unique envs, then highest count
                    if (prev is None
                            or (is_unique_env and not prev_unique)
                            or (is_unique_env == prev_unique and count > prev[0])):
                        best[bucket_key] = (count, env_id, seed, is_unique_env, n_states)

        elapsed = time.perf_counter() - started
        if seed % 10 == 0:
            print(f"  Seed {seed} done ({elapsed:.1f}s)")

    # Collect results: nested dict {profile: {role: info}}
    results = defaultdict(dict)
    for (role_s, profile), (count, env_id, seed, _, n_states) in sorted(best.items()):
        lds_path = SEED_DIR / f"lds_seed{seed}.json"
        with open(lds_path) as f:
            lds_data = json.load(f)
        lds_str = lds_data[env_id]["lds"]
        cfg = cfg_cache.get(env_id) or load_config(env_id)

        results[profile][role_s] = {
            "env_id": env_id,
            "seed": seed,
            "lds": lds_str,
            "uniquely_optimal_count": count,
            "total_states": n_states,
        }
        print(
            f"[{profile}] {role_s} -> env {env_id:>5}  "
            f"{count}/{n_states} canonically optimal  seed={seed}  "
            f"lds={lds_str}  "
            f"team={cfg['team_max_hp']}  enemy={cfg['enemy_max_hp']}  "
            f"boss={cfg['boss_damage']}  prob={cfg['enemy_prob']}"
        )

    # Summary: count picks per profile
    print("\nPicks per stat profile:")
    for profile in sorted(results.keys()):
        roles = sorted(results[profile].keys())
        print(f"  {profile}: {len(roles)} roles -> {roles}")

    # Save results
    out_path = SCRIPT_DIR / "output" / "picked_envs_by_role_and_profile.txt"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"\nSaved to {out_path}")

    # Copy to output dir: OUT_DIR/PROFILE/COMBO/
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir()
    for profile, roles in sorted(results.items()):
        profile_dir = OUT_DIR / profile
        profile_dir.mkdir()
        for role_s, info in sorted(roles.items()):
            role_dir = profile_dir / role_s
            role_dir.mkdir()
            shutil.copy2(ENVS_DIR / info["env_id"] / "config.py", role_dir / "config.py")
            (role_dir / "lds.txt").write_text(info["lds"] + "\n")
    print(f"Copied configs + LDS to '{OUT_DIR}/'")


if __name__ == "__main__":
    main()
