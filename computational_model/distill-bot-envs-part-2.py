import numpy as np
import shutil
import importlib.util
from pathlib import Path
from collections import defaultdict
from scipy.stats import qmc

# --- CONFIGURATION ---

F, T, M = 0, 1, 2
ROLE_CHAR_MAP = {'F': F, 'T': T, 'M': M}
ROLE_INT_MAP = {F: 'F', T: 'T', M: 'M'}
ACTION_MAP = {0: 'ATTACK', 1: 'DEFEND', 2: 'HEAL'}

INPUT_DIR = Path("optimal_envs_for_bots")
OUTPUT_DIR = Path("bot_case_envs")
ENV_DATA_DIR = Path("envs")

SIM_HORIZON = 10
SIM_SEED = 42

# SWITCHING LOGIC CONSTANTS
TURNS_BEFORE_SWITCH = 4

# --- HELPERS ---

def parse_roles_from_string(role_str):
    return tuple(ROLE_CHAR_MAP[c] for c in role_str)

def parse_env_mapping():
    env_info = {}
    try:
        with open("env_mapping.txt") as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2: continue
                env_id = parts[0]
                config = parts[1]
                try:
                    p_idx = config.rindex('_P') + 2
                    prob_attack = int(config[p_idx:]) / 100.0
                    env_info[env_id] = prob_attack
                except (ValueError, IndexError): continue
    except FileNotFoundError: return {}
    return env_info

def load_config_attr(env_id, attr_name):
    config_path = ENV_DATA_DIR / env_id / "config.py"
    if not config_path.exists(): return None
    try:
        spec = importlib.util.spec_from_file_location("config", config_path)
        if spec is None or spec.loader is None: return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, attr_name, None)
    except Exception:
        return None

def generate_binary_qmc(n, prob, seed=None):
    if n <= 0: return np.array([], dtype=int)
    if n == 1: return np.array([1], dtype=int)
    try:
        engine = qmc.Sobol(d=1, scramble=True, seed=seed)
        samples = engine.random(n - 1).flatten()
    except (AttributeError, ValueError):
        np.random.seed(seed)
        samples = np.random.rand(n - 1)
        
    threshold = (prob * n - 1) / (n - 1)
    threshold = np.clip(threshold, 0.0, 1.0)
    binary_tail = (samples < threshold).astype(int)
    return np.concatenate(([1], binary_tail))

def get_dominant_role(stats):
    return int(np.argmax(stats))

def get_player_action(role, intent, team_hp, team_max_hp):
    if role == F: return 0 # ATTACK
    elif role == T: return 1 if intent == 1 else 0 # DEFEND if intent else ATTACK
    else: return 2 if team_hp < team_max_hp else 0 # HEAL if injured else ATTACK

def simulate_outcome(env_id, p1_role, p2_role, p3_role):
    # Load Config
    player_stats = load_config_attr(env_id, "PLAYER_STATS")
    boss_damage = load_config_attr(env_id, "BOSS_DAMAGE")
    team_max_hp = load_config_attr(env_id, "TEAM_MAX_HP")
    enemy_max_hp = load_config_attr(env_id, "ENEMY_MAX_HP")
    enemy_prob   = load_config_attr(env_id, "ENEMY_ATTACK_PROB")
    
    if any(x is None for x in [player_stats, boss_damage, team_max_hp, enemy_max_hp, enemy_prob]):
        return False

    lds_array = generate_binary_qmc(SIM_HORIZON, enemy_prob, seed=SIM_SEED)

    team_hp = float(team_max_hp)
    enemy_hp = float(enemy_max_hp)
    p1_natural = get_dominant_role(player_stats[0])
    
    for step in range(len(lds_array)):
        intent = lds_array[step]
        
        current_p1 = p1_natural if step < TURNS_BEFORE_SWITCH else p1_role
        current_roles = [current_p1, p2_role, p3_role]
        
        actions = [get_player_action(current_roles[i], intent, team_hp, team_max_hp) for i in range(3)]
        
        total_dmg = np.sum([player_stats[i, 0] for i in range(3) if actions[i] == 0]) 
        total_heal = np.sum([player_stats[i, 2] for i in range(3) if actions[i] == 2]) 
        defenders = [player_stats[i, 1] for i in range(3) if actions[i] == 1] 
        max_defense = np.max(defenders) if defenders else 0.0
        
        enemy_hp = max(0, enemy_hp - total_dmg)
        
        dmg_taken = 0.0
        if intent == 1:
            dmg_taken = max(0, boss_damage - max_defense)
            
        team_hp = max(0, min(team_max_hp, team_hp - dmg_taken + total_heal))
        
        if team_hp <= 0: return False
        if enemy_hp <= 0: return True
        
    return False

def check_strict_win_conditions(env_id, human_forced, bot1, bot2):
    # 1. Forced Role must Win
    if not simulate_outcome(env_id, human_forced, bot1, bot2): return False
    # 2. Alternatives must Lose
    for alt_role in [F, T, M]:
        if alt_role == human_forced: continue
        if simulate_outcome(env_id, alt_role, bot1, bot2): return False
    return True

# --- LOGGING FUNCTION ---

def generate_debug_log(env_id, dest_path, human_forced, bot1, bot2, prob_attack):
    player_stats = load_config_attr(env_id, "PLAYER_STATS")
    boss_damage = load_config_attr(env_id, "BOSS_DAMAGE")
    team_max_hp = load_config_attr(env_id, "TEAM_MAX_HP")
    enemy_max_hp = load_config_attr(env_id, "ENEMY_MAX_HP")
    
    if any(x is None for x in [player_stats, boss_damage, team_max_hp, enemy_max_hp]):
        return

    lds_array = generate_binary_qmc(SIM_HORIZON, prob_attack, seed=SIM_SEED)
    p1_natural = get_dominant_role(player_stats[0]) 

    log_lines = []
    log_lines.append(f"=== SIMULATION LOG FOR ENV {env_id} ===")
    log_lines.append(f"Team Max HP: {team_max_hp} | Enemy Max HP: {enemy_max_hp}")
    log_lines.append(f"Boss Base Damage: {boss_damage}")
    log_lines.append(f"Attack Probability: {prob_attack}")
    log_lines.append(f"Switch Turn: {TURNS_BEFORE_SWITCH + 1}")
    log_lines.append(f"Attack Sequence (LDS): {lds_array}")
    
    log_lines.append(f"P1 Natural Role (Before Switch): {ROLE_INT_MAP[p1_natural]}")
    log_lines.append("-" * 60)

    scenarios = [("FORCED (WIN)", human_forced)]
    for r in [F, T, M]:
        if r != human_forced:
            scenarios.append(("ALTERNATIVE (LOSE)", r))

    for label, p1_target_role in scenarios:
        log_lines.append(f"\n>>> SCENARIO: {label} | Human switches to {ROLE_INT_MAP[p1_target_role]}")
        log_lines.append(f"    Bots play: P2={ROLE_INT_MAP[bot1]}, P3={ROLE_INT_MAP[bot2]}")
        
        team_hp = float(team_max_hp)
        enemy_hp = float(enemy_max_hp)
        outcome = "TIMEOUT"
        
        for step in range(len(lds_array)):
            intent = lds_array[step]
            boss_act = "ATTACK" if intent == 1 else "CHARGE"
            
            current_p1 = p1_natural if step < TURNS_BEFORE_SWITCH else p1_target_role
            current_roles = [current_p1, bot1, bot2]
            role_str = "".join([ROLE_INT_MAP[r] for r in current_roles])
            
            actions = [get_player_action(current_roles[i], intent, team_hp, team_max_hp) for i in range(3)]
            act_str = ", ".join([ACTION_MAP[a] for a in actions])
            
            total_dmg = np.sum([player_stats[i, 0] for i in range(3) if actions[i] == 0])
            total_heal = np.sum([player_stats[i, 2] for i in range(3) if actions[i] == 2])
            defenders = [player_stats[i, 1] for i in range(3) if actions[i] == 1]
            max_defense = np.max(defenders) if defenders else 0.0
            
            prev_enemy_hp = enemy_hp
            enemy_hp = max(0, enemy_hp - total_dmg)
            
            dmg_taken = 0.0
            if intent == 1:
                dmg_taken = max(0, boss_damage - max_defense)
            
            prev_team_hp = team_hp
            team_hp = max(0, min(team_max_hp, team_hp - dmg_taken + total_heal))
            
            log_lines.append(f"   [Turn {step+1}] Boss: {boss_act} | Roles: {role_str}")
            log_lines.append(f"      Actions: {act_str}")
            log_lines.append(f"      Dmg Dealt: {total_dmg:.1f} (Enemy HP: {prev_enemy_hp:.1f} -> {enemy_hp:.1f})")
            log_lines.append(f"      Defense: {max_defense:.1f} | Dmg Taken: {dmg_taken:.1f} | Heal: {total_heal:.1f}")
            log_lines.append(f"      Team HP: {prev_team_hp:.1f} -> {team_hp:.1f}")

            if team_hp <= 0:
                outcome = "LOSS (Team Died)"
                break
            if enemy_hp <= 0:
                outcome = "WIN (Enemy Died)"
                break
        
        log_lines.append(f"   >>> RESULT: {outcome}")
        log_lines.append("-" * 30)

    try:
        with open(dest_path, "w") as f:
            f.write("\n".join(log_lines))
    except Exception as e:
        print(f"Error writing log: {e}")

# --- MAIN LOGIC ---

env_probs = parse_env_mapping()

if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir()

final_assignments = []

# 1. SCAN AND COLLECT ALL CANDIDATES
if INPUT_DIR.exists():
    for stat_dir in sorted(INPUT_DIR.iterdir()):
        if not stat_dir.is_dir(): continue
        stat_key = stat_dir.name
        
        for case_dir in sorted(stat_dir.iterdir()):
            if not case_dir.is_dir(): continue
            case_name = case_dir.name 
            if "_" not in case_name: continue
            
            target_str, forced_str = case_name.split('_')
            forced_roles = parse_roles_from_string(forced_str)
            human_forced, bot1, bot2 = forced_roles
            
            env_ids_file = case_dir / "env_ids.txt"
            if not env_ids_file.exists(): continue
            
            with open(env_ids_file) as f:
                env_ids = [line.strip() for line in f if line.strip()]
            
            for env_id in env_ids:
                if env_id not in env_probs: continue
                
                # --- STRICT WIN/LOSS CHECK ---
                if not check_strict_win_conditions(env_id, human_forced, bot1, bot2):
                    continue

                # Add to final list (No EV calculation needed)
                final_assignments.append({
                    'stat_key': stat_key,
                    'case_name': case_name,
                    'env_id': env_id,
                    'prob_attack': env_probs[env_id],
                    'human_forced': human_forced,
                    'bot1': bot1,
                    'bot2': bot2
                })

# 2. SORT FOR PRINTING
final_assignments.sort(key=lambda x: (x['stat_key'], x['case_name'], x['env_id']))

# 3. PRINT TABLE & WRITE OUTPUT
print("\n" + "="*80)
print(f"{'STAT BLOCK':<15} | {'CASE':<10} | {'ENV ID':<8} | {'PROB ATTACK':<12}")
print("="*80)

for item in final_assignments:
    stat_key = item['stat_key']
    case_name = item['case_name']
    env_id = item['env_id']
    prob_attack = item['prob_attack']
    
    print(f"{stat_key:<15} | {case_name:<10} | {env_id:<8} | {prob_attack:<12.2f}")
    
    src_config = ENV_DATA_DIR / env_id / "config.py"
    dest_dir = OUTPUT_DIR / stat_key / case_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    if src_config.exists():
        shutil.copy2(src_config, dest_dir / f"{env_id}.py")
        
        lds_vector = generate_binary_qmc(SIM_HORIZON, prob_attack, seed=SIM_SEED)
        np.save(dest_dir / f"{env_id}_lds.npy", lds_vector)
        
        log_path = dest_dir / f"{env_id}_sim_log.txt"
        generate_debug_log(
            env_id, 
            log_path, 
            item['human_forced'], 
            item['bot1'], 
            item['bot2'], 
            prob_attack
        )

print("=" * 80)
print(f"Total Environments Saved: {len(final_assignments)}")