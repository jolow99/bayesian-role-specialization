"""
Directed env search: given a target role combo and a fixed stat profile,
cycle through env parameter combinations, solve the MDP for each, and
return any env where the target canonical role class is uniquely optimal
in > 50% of the 27 turn-3 states (>= 14/27).

Usage:
    python find-env-for-combo.py --target FTM --profile 222_222_222
    python find-env-for-combo.py --target FTM --profile 222_222_222 --limit 3
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import itertools
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import qmc

SCRIPT_DIR = Path(__file__).resolve().parent
MDP_DIR = SCRIPT_DIR.parent / "MDP"
OUTPUT_DIR = SCRIPT_DIR / "directed_search_output"

NUM_TURNS_TO_SIMULATE = 2
HORIZON = 10
TIE_TOL = 1e-9
SEEDS = range(1, 51)

# Defaults — edit these to change what the script searches for
TARGET = "FTM"
PROFILE = "222_222_222"
LIMIT = 1
THRESHOLD = 14
WORKERS = 3

F, T, M = 0, 1, 2
ATTACK, DEFEND, HEAL = 0, 1, 2
CHAR_TO_INT = {"F": 0, "T": 1, "M": 2}
ROLE_CHARS = ["F", "T", "M"]
ALL_ROLE_COMBOS_STR = [
    a + b + c
    for a in ROLE_CHARS
    for b in ROLE_CHARS
    for c in ROLE_CHARS
]

# Grid matches MDP/search.py ranges; tune as needed
GRID = {
    "TEAM_MAX_HP": [5, 10, 15, 20],
    "ENEMY_MAX_HP": list(range(10, 31, 2)),
    "BOSS_DAMAGE": list(range(1, 11)),
    "ENEMY_ATTACK_PROB": [0.2, 0.25, 0.33, 0.4, 0.5, 0.6, 0.66, 0.75, 0.8, 1.0],
}


def parse_profile(s):
    return [[int(c) for c in part] for part in s.split("_")]


def role_combo_to_index(role_str):
    return CHAR_TO_INT[role_str[0]] * 9 + CHAR_TO_INT[role_str[1]] * 3 + CHAR_TO_INT[role_str[2]]


def role_combo_to_tuple(role_str):
    return tuple(CHAR_TO_INT[c] for c in role_str)


def canonical_role_combo(role_combo, player_stats):
    canonical = list(role_combo)
    stat_groups = {}

    for i, row in enumerate(np.asarray(player_stats).tolist()):
        stat_groups.setdefault(tuple(float(x) for x in row), []).append(i)

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
        for role_combo in ALL_ROLE_COMBOS_STR
    ]


def generate_lds(horizon, prob, seed):
    if horizon <= 1:
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


def simulate_turn3(player_stats, boss_dmg, team_max, enemy_max, lds, role_combo):
    team_hp = float(team_max)
    enemy_hp = float(enemy_max)
    for step in range(NUM_TURNS_TO_SIMULATE):
        intent = int(lds[step])
        actions = [get_action(role_combo[i], intent, team_hp, team_max) for i in range(3)]
        total_dmg = sum(player_stats[i][ATTACK] for i in range(3) if actions[i] == ATTACK)
        total_heal = sum(player_stats[i][HEAL] for i in range(3) if actions[i] == HEAL)
        defenders = [player_stats[i][DEFEND] for i in range(3) if actions[i] == DEFEND]
        max_def = max(defenders) if defenders else 0.0
        enemy_hp = max(0.0, enemy_hp - total_dmg)
        dmg = max(0.0, boss_dmg - max_def) if intent == 1 else 0.0
        team_hp = max(0.0, min(float(team_max), team_hp - dmg + total_heal))
    return {
        "intent": int(lds[NUM_TURNS_TO_SIMULATE]),
        "team_hp": team_hp,
        "enemy_hp": enemy_hp,
    }


def unique_best_role_class(values, intent, team_hp, enemy_hp, role_class_lookup):
    vals = values[:, intent, team_hp, enemy_hp]
    class_values = {}

    for role_idx, role_class in role_class_lookup:
        value = float(vals[role_idx])
        if role_class not in class_values or value > class_values[role_class]:
            class_values[role_class] = value

    mx = max(class_values.values())
    best_classes = [
        role_class
        for role_class, value in class_values.items()
        if np.isclose(value, mx, atol=TIE_TOL, rtol=0.0)
    ]

    if len(best_classes) != 1:
        return None
    return best_classes[0]


def simulate_full_game(params, stats, lds, role_tuple):
    team_hp = float(params["TEAM_MAX_HP"])
    enemy_hp = float(params["ENEMY_MAX_HP"])
    team_max = params["TEAM_MAX_HP"]
    boss_dmg = params["BOSS_DAMAGE"]
    for step in range(len(lds)):
        intent = int(lds[step])
        actions = [get_action(role_tuple[i], intent, team_hp, team_max) for i in range(3)]
        total_dmg = sum(stats[i][ATTACK] for i in range(3) if actions[i] == ATTACK)
        total_heal = sum(stats[i][HEAL] for i in range(3) if actions[i] == HEAL)
        defenders = [stats[i][DEFEND] for i in range(3) if actions[i] == DEFEND]
        max_def = max(defenders) if defenders else 0.0
        enemy_hp = max(0.0, enemy_hp - total_dmg)
        dmg = max(0.0, boss_dmg - max_def) if intent == 1 else 0.0
        team_hp = max(0.0, min(float(team_max), team_hp - dmg + total_heal))
        if team_hp <= 0:
            return False
        if enemy_hp <= 0:
            return True
    return False


def write_config(params, stats, filepath):
    content = f'''import jax
import jax.numpy as jnp
from itertools import product

ATTACKER, DEFENDER, HEALER = 0, 1, 2
ATTACK, DEFEND, HEAL = 0, 1, 2
ENEMY_NO_ATTACK_INTENT, ENEMY_ATTACK_INTENT = 0, 1

ROLES = [ATTACKER, DEFENDER, HEALER]
ACTIONS = [ATTACK, DEFEND, HEAL]
ROLE_COMBOS = jnp.array(list(product(ROLES, repeat=3)))
ROLE_INDICES = jnp.array(list(range(27)))

TEAM_MAX_HP = {params['TEAM_MAX_HP']}
ENEMY_MAX_HP = {params['ENEMY_MAX_HP']}

PLAYER_STATS = jnp.array({stats})
ATTACK_STATS = PLAYER_STATS[:, ATTACK]
DEFEND_STATS = PLAYER_STATS[:, DEFEND]
HEAL_STATS = PLAYER_STATS[:, HEAL]

BOSS_DAMAGE = {params['BOSS_DAMAGE']}
ENEMY_ATTACK_PROB = {params['ENEMY_ATTACK_PROB']}

EPSILON, HORIZON = 1e-10, 10
H, W = TEAM_MAX_HP + 1, ENEMY_MAX_HP + 1
S = jnp.arange(2 * H * W)
ACTION_PROFILES = jnp.array(list(product(ACTIONS, repeat=3)))
A = jnp.arange(len(ACTION_PROFILES))

@jax.jit
def get_intent_and_hps_from_state(s):
    i = s // (H * W)
    rem = s % (H * W)
    return i, rem // W, rem % W
'''
    filepath.write_text(content)


def solve_mdp(params, stats, temp_dir, timeout=180):
    temp_dir.mkdir(parents=True, exist_ok=True)
    write_config(params, stats, temp_dir / "config.py")
    for fn in ["V.py", "pi.py", "R.py", "T.py"]:
        shutil.copy(MDP_DIR / fn, temp_dir / fn)
    try:
        result = subprocess.run(
            ["python", "V.py"],
            capture_output=True,
            cwd=temp_dir,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    values_file = temp_dir / "values.npy"
    if not values_file.exists():
        return None
    return np.load(values_file)


def evaluate_env(values, params, stats, target_role, seeds):
    """For each seed, check if all 27 turn-3 states are alive and count
    how many have the target canonical role class uniquely optimal.
    Returns the best (count, seed, lds_str).
    Also ensures target wins the full game under the chosen LDS."""
    ALL_COMBOS = list(itertools.product([F, T, M], repeat=3))
    target_class = canonical_role_combo(target_role, stats)
    target_tuple = role_combo_to_tuple(target_class)
    role_class_lookup = build_role_class_lookup(stats)

    best = (-1, None, None)

    for seed in seeds:
        lds = generate_lds(HORIZON, params["ENEMY_ATTACK_PROB"], seed)

        states = []
        alive = True
        for combo in ALL_COMBOS:
            s = simulate_turn3(
                stats, params["BOSS_DAMAGE"], params["TEAM_MAX_HP"],
                params["ENEMY_MAX_HP"], lds, combo,
            )
            if s["team_hp"] <= 0:
                alive = False
                break
            states.append(s)
        if not alive:
            continue

        count = 0
        for s in states:
            intent = max(0, min(int(s["intent"]), values.shape[1] - 1))
            thp = max(0, min(int(round(s["team_hp"])), values.shape[2] - 1))
            ehp = max(0, min(int(round(s["enemy_hp"])), values.shape[3] - 1))
            if unique_best_role_class(values, intent, thp, ehp, role_class_lookup) == target_class:
                count += 1

        if count < 14:
            continue

        if not simulate_full_game(params, stats, lds, target_tuple):
            continue

        if count > best[0]:
            best = (count, seed, "".join(str(int(x)) for x in lds))

    return best


def process_grid_point(args_tuple):
    """Worker: solve MDP + evaluate one param combo. Returns a dict or None."""
    i, total, team, enemy, boss, prob, stats, target, threshold = args_tuple
    params = {
        "TEAM_MAX_HP": team,
        "ENEMY_MAX_HP": enemy,
        "BOSS_DAMAGE": boss,
        "ENEMY_ATTACK_PROB": prob,
    }
    temp = SCRIPT_DIR / f"_temp_search_{os.getpid()}_{i}"
    try:
        values = solve_mdp(params, stats, temp)
        if values is None:
            return None
        count, seed, lds = evaluate_env(values, params, stats, target, SEEDS)
        if count < threshold:
            return {"i": i, "params": params, "count": count, "matched": False}

        out = OUTPUT_DIR / (
            f"{target}_{'_'.join(''.join(str(x) for x in row) for row in stats)}_"
            f"T{team}_E{enemy}_B{boss}_P{int(prob*100)}"
        )
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(temp / "config.py", out / "config.py")
        shutil.copy(temp / "values.npy", out / "values.npy")
        (out / "lds.txt").write_text(lds + "\n")
        (out / "info.txt").write_text(
            f"target={target}\ncount={count}/27\nseed={seed}\n"
            f"TEAM_MAX_HP={team}\nENEMY_MAX_HP={enemy}\n"
            f"BOSS_DAMAGE={boss}\nENEMY_ATTACK_PROB={prob}\n"
        )
        return {
            "i": i, "params": params, "count": count, "seed": seed,
            "matched": True, "out": out.name,
        }
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def main():
    if len(TARGET) != 3 or any(c not in "FTM" for c in TARGET):
        sys.exit(f"Invalid TARGET '{TARGET}'; must be 3 chars from FTM")

    stats = parse_profile(PROFILE)
    OUTPUT_DIR.mkdir(exist_ok=True)

    grid_combos = list(itertools.product(
        GRID["TEAM_MAX_HP"], GRID["ENEMY_MAX_HP"],
        GRID["BOSS_DAMAGE"], GRID["ENEMY_ATTACK_PROB"],
    ))
    total = len(grid_combos)
    print(f"Target: {TARGET}  profile: {PROFILE}")
    print(f"Grid size: {total} param combos  workers: {WORKERS}")
    print(f"Output dir: {OUTPUT_DIR}")
    print()

    tasks = [
        (i, total, team, enemy, boss, prob, stats, TARGET, THRESHOLD)
        for i, (team, enemy, boss, prob) in enumerate(grid_combos, 1)
    ]

    found = []
    started = time.perf_counter()
    completed = 0

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(process_grid_point, t): t for t in tasks}
        try:
            for fut in as_completed(futures):
                completed += 1
                res = fut.result()
                if res is None:
                    pass
                elif res["matched"]:
                    found.append((res["count"], res["out"], res["params"], res["seed"]))
                    elapsed = time.perf_counter() - started
                    p = res["params"]
                    print(
                        f"[{completed}/{total}] ({elapsed:.0f}s) FOUND "
                        f"T={p['TEAM_MAX_HP']} E={p['ENEMY_MAX_HP']} "
                        f"B={p['BOSS_DAMAGE']} P={p['ENEMY_ATTACK_PROB']} "
                        f"count={res['count']}/27 seed={res['seed']}"
                    )
                    if len(found) >= LIMIT:
                        for f in futures:
                            f.cancel()
                        break

                if completed % 20 == 0:
                    elapsed = time.perf_counter() - started
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else 0
                    print(f"  [{completed}/{total}] elapsed={elapsed:.0f}s eta={eta:.0f}s  found={len(found)}")
        except KeyboardInterrupt:
            for f in futures:
                f.cancel()
            raise

    print(f"\nDone. Found {len(found)} matching env(s) under {OUTPUT_DIR}/")
    print(f"Target: {TARGET}  profile: {PROFILE}  stats: {stats}")
    print()
    for count, name, p, seed in found:
        print(f"  {name}  ({count}/27)")
        print(f"    TEAM_MAX_HP      = {p['TEAM_MAX_HP']}")
        print(f"    ENEMY_MAX_HP     = {p['ENEMY_MAX_HP']}")
        print(f"    BOSS_DAMAGE      = {p['BOSS_DAMAGE']}")
        print(f"    ENEMY_ATTACK_PROB= {p['ENEMY_ATTACK_PROB']}")
        print(f"    seed             = {seed}")


if __name__ == "__main__":
    main()
