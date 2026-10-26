import os
os.environ["JAX_PLATFORMS"] = "cpu"

import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np


# --- Constants ---
ATTACK, DEFEND, HEAL = 0, 1, 2
F, T, M = 0, 1, 2
ROLE_TO_CHAR = {F: "F", T: "T", M: "M"}

SCRIPT_DIR = Path(__file__).resolve().parent
SIM_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 14
OUTPUT_FILE = Path(sys.argv[2]) if len(sys.argv) > 2 else SCRIPT_DIR / "output" / "turn3_states_by_env_role.txt"
LDS_OUTPUT_FILE = Path(sys.argv[3]) if len(sys.argv) > 3 else SCRIPT_DIR / "output" / "human_lds_seqs.txt"
NUM_TURNS_TO_SIMULATE = 2
ENVS_DIR = SCRIPT_DIR.parent / "envs"
MAX_WORKERS = min(8, (os.cpu_count() or 1))
PROGRESS_EVERY = 100

ONE_UNIQUE_FIRST_PLAYER_PROFILE_TO_FIXED_ROLE = {
    (1.0, 1.0, 4.0): M,  # 114_222_222 -> P1 fixed as M
    (1.0, 4.0, 1.0): T,  # 141_222_222 -> P1 fixed as T
    (4.0, 1.0, 1.0): F,  # 411_222_222 -> P1 fixed as F
}


def load_config_module(config_path):
    try:
        module_name = f"config_{abs(hash(str(config_path)))}"
        spec = importlib.util.spec_from_file_location(module_name, config_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


def generate_binary_qmc(n, prob, seed=None):
    if n <= 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([1], dtype=int)

    try:
        from scipy.stats import qmc
        engine = qmc.Sobol(d=1, scramble=True, seed=seed)
        samples = engine.random(n - 1).flatten()
    except Exception:
        np.random.seed(seed)
        samples = np.random.rand(n - 1)

    threshold = (prob * n - 1) / (n - 1)
    threshold = np.clip(threshold, 0.0, 1.0)
    tail = (samples < threshold).astype(int)
    return np.concatenate(([1], tail))


def get_player_action(role, intent, team_hp, team_max_hp):
    if role == F:
        return ATTACK
    if role == T:
        return DEFEND if intent == 1 else ATTACK
    return HEAL if team_hp < team_max_hp else ATTACK


def role_combo_to_string(role_combo):
    return "".join(ROLE_TO_CHAR[r] for r in role_combo)


def get_role_combos_to_simulate(player_stats):
    # Always simulate all 27 combos so survival checks cover every possibility
    return list(itertools.product([F, T, M], repeat=3))


def simulate_two_turns_and_get_turn3_state(player_stats, boss_damage, team_max_hp, enemy_max_hp, lds, role_combo):
    team_hp = float(team_max_hp)
    enemy_hp = float(enemy_max_hp)

    for step in range(NUM_TURNS_TO_SIMULATE):
        intent = int(lds[step])

        actions = [
            get_player_action(role_combo[i], intent, team_hp, team_max_hp)
            for i in range(3)
        ]

        total_dmg = np.sum([player_stats[i, ATTACK] for i in range(3) if actions[i] == ATTACK])
        total_heal = np.sum([player_stats[i, HEAL] for i in range(3) if actions[i] == HEAL])
        defenders = [player_stats[i, DEFEND] for i in range(3) if actions[i] == DEFEND]
        max_defense = np.max(defenders) if defenders else 0.0

        enemy_hp = max(0.0, enemy_hp - float(total_dmg))

        if intent == 1:
            dmg_taken = max(0.0, float(boss_damage) - float(max_defense))
        else:
            dmg_taken = 0.0

        team_hp = team_hp - dmg_taken + float(total_heal)
        team_hp = max(0.0, min(float(team_max_hp), team_hp))

    return {
        "intent": int(lds[NUM_TURNS_TO_SIMULATE]),
        "team_hp": float(team_hp),
        "enemy_hp": float(enemy_hp),
    }


def process_environment(env_dir):
    env_id = env_dir.name
    config_path = env_dir / "config.py"
    if not config_path.exists():
        return {}, None

    module = load_config_module(config_path)
    if module is None:
        return {}, None

    player_stats = getattr(module, "PLAYER_STATS", None)
    boss_damage = getattr(module, "BOSS_DAMAGE", None)
    team_max_hp = getattr(module, "TEAM_MAX_HP", None)
    enemy_max_hp = getattr(module, "ENEMY_MAX_HP", None)
    enemy_prob = getattr(module, "ENEMY_ATTACK_PROB", None)

    if any(v is None for v in [player_stats, boss_damage, team_max_hp, enemy_max_hp, enemy_prob]):
        return {}, None

    player_stats = np.array(player_stats, dtype=float)
    env_results = {}

    horizon = getattr(module, "HORIZON", 10)
    lds = generate_binary_qmc(horizon, enemy_prob, seed=SIM_SEED)

    role_combos = get_role_combos_to_simulate(player_stats)

    for role_combo in role_combos:
        role_str = role_combo_to_string(role_combo)
        key = f"{env_id}_{role_str}"

        state = simulate_two_turns_and_get_turn3_state(
            player_stats=player_stats,
            boss_damage=boss_damage,
            team_max_hp=team_max_hp,
            enemy_max_hp=enemy_max_hp,
            lds=lds,
            role_combo=role_combo,
        )
        env_results[key] = state

    lds_str = "".join(str(int(x)) for x in lds)
    lds_info = {"env_id": env_id, "enemy_prob": float(enemy_prob), "lds": lds_str}

    return env_results, lds_info


def format_eta(seconds):
    if seconds < 0 or not np.isfinite(seconds):
        return "unknown"
    seconds = int(seconds)
    mins, sec = divmod(seconds, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs}h {mins}m {sec}s"
    if mins > 0:
        return f"{mins}m {sec}s"
    return f"{sec}s"


def main():
    results = {}
    lds_records = {}

    env_dirs = [p for p in sorted(ENVS_DIR.iterdir(), key=lambda x: int(x.name)) if p.is_dir() and p.name.isdigit()]

    total_envs = len(env_dirs)
    if total_envs == 0:
        with open(OUTPUT_FILE, "w") as f:
            f.write(json.dumps(results, indent=2, sort_keys=True))
        print(f"Saved {len(results)} entries to {OUTPUT_FILE} (workers={MAX_WORKERS})")
        return

    print(
        f"Starting simulation for {total_envs} environments "
        f"(workers={MAX_WORKERS}, status every {PROGRESS_EVERY} envs)..."
    )

    started_at = time.perf_counter()
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_environment, env_dir) for env_dir in env_dirs]
        for future in as_completed(futures):
            env_results, lds_info = future.result()
            if env_results:
                results.update(env_results)
                lds_records[lds_info["env_id"]] = lds_info
            completed += 1

            if completed % PROGRESS_EVERY == 0 or completed == total_envs:
                elapsed = time.perf_counter() - started_at
                rate = completed / elapsed if elapsed > 0 else 0.0
                remaining = total_envs - completed
                eta_seconds = remaining / rate if rate > 0 else float("inf")
                progress_pct = (completed / total_envs) * 100
                print(
                    f"Progress: {completed}/{total_envs} ({progress_pct:.1f}%) | "
                    f"elapsed={elapsed:.1f}s | rate={rate:.2f} env/s | "
                    f"eta={format_eta(eta_seconds)}"
                )

    with open(OUTPUT_FILE, "w") as f:
        f.write(json.dumps(results, indent=2, sort_keys=True))

    with open(LDS_OUTPUT_FILE, "w") as f:
        json.dump(lds_records, f, indent=2, sort_keys=True)

    total_elapsed = time.perf_counter() - started_at
    print(
        f"Saved {len(results)} entries to {OUTPUT_FILE} "
        f"(workers={MAX_WORKERS}, elapsed={total_elapsed:.1f}s)"
    )
    print(f"Saved {len(lds_records)} LDS sequences to {LDS_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
