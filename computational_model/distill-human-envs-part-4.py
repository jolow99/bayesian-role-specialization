import os
import numpy as np
import importlib.util
from collections import defaultdict

HUMAN_CASE_ENVS_DIR = "human_case_envs"
LDS_FILE = "lds_arrays_human.txt"
OUTPUT_FILE = "human_results.txt"

ATTACK, DEFEND, HEAL = 0, 1, 2
FIGHTER, DEFENDER, HEALER = 0, 1, 2
ROLE_MAP = {'F': FIGHTER, 'T': DEFENDER, 'M': HEALER}

def load_attr(file_path, attr):
    try:
        spec = importlib.util.spec_from_file_location("module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, attr, None)
    except:
        return None

def load_lds_dict():
    lds_dict = {}
    with open(LDS_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            key, prob, sequence = parts[0], float(parts[1]), parts[2]
            if key not in lds_dict:
                lds_dict[key] = []
            lds_dict[key].append(np.array([int(c) for c in sequence]))
    return lds_dict

def get_player_action(role, intent, team_hp, team_max_hp):
    if role == FIGHTER:
        return ATTACK
    elif role == DEFENDER:
        return DEFEND if intent == 1 else ATTACK
    else:  # HEALER
        return HEAL if team_hp < team_max_hp else ATTACK

def simulate_battle(team_hp, enemy_hp, player_stats, boss_damage, role_combo, lds_array, team_max_hp):
    team_hp = float(team_hp)
    enemy_hp = float(enemy_hp)
    roles = [ROLE_MAP[c] for c in role_combo]
    
    for step in range(len(lds_array)):
        intent = lds_array[step] # 1 = Attack, 0 = Charge/Wait
        
        actions = [get_player_action(roles[i], intent, team_hp, team_max_hp) for i in range(3)]
        
        # 1. Calculate Potentials
        total_dmg = np.sum([player_stats[i, ATTACK] for i in range(3) if actions[i] == ATTACK])
        total_heal = np.sum([player_stats[i, HEAL] for i in range(3) if actions[i] == HEAL])
        
        # FIX 1: Max Defense, not Sum Defense
        defenders = [player_stats[i, DEFEND] for i in range(3) if actions[i] == DEFEND]
        max_defense = np.max(defenders) if defenders else 0.0
        
        # 2. Update Enemy HP
        enemy_hp = max(0, enemy_hp - total_dmg)
        
        # 3. Calculate Incoming Damage
        # FIX 2: Clamp damage to 0 so it doesn't heal
        if intent == 1:
            dmg_taken = max(0, boss_damage - max_defense)
        else:
            dmg_taken = 0.0
            
        # 4. Update Team HP
        # FIX 3: Apply Heal and Damage simultaneously, then clamp
        team_hp = team_hp - dmg_taken + total_heal
        team_hp = max(0, min(team_hp, team_max_hp))
        
        # FIX 4: Check death only after full round resolution (No mutual kills allowed)
        if team_hp <= 0:
            return False # Loss
        if enemy_hp <= 0:
            return True  # Win (Team survived)
            
    return False # Timeout

def main():
    lds_dict = load_lds_dict()
    results = defaultdict(lambda: {"wins": 0, "losses": 0})
    
    for stats_combo in os.listdir(HUMAN_CASE_ENVS_DIR):
        stats_path = os.path.join(HUMAN_CASE_ENVS_DIR, stats_combo)
        if not os.path.isdir(stats_path): continue
        
        for role_combo in os.listdir(stats_path):
            role_path = os.path.join(stats_path, role_combo)
            if not os.path.isdir(role_path): continue
            
            key = f"{stats_combo}_{role_combo}"
            lds_arrays = lds_dict.get(key, [])
            
            for fname in os.listdir(role_path):
                if not fname.endswith(".py"): continue
                
                fpath = os.path.join(role_path, fname)
                player_stats = load_attr(fpath, "PLAYER_STATS")
                boss_damage = load_attr(fpath, "BOSS_DAMAGE")
                team_max_hp = load_attr(fpath, "TEAM_MAX_HP")
                enemy_max_hp = load_attr(fpath, "ENEMY_MAX_HP")
                
                if any(x is None for x in [player_stats, boss_damage, team_max_hp, enemy_max_hp]):
                    continue
                
                for lds_array in lds_arrays:
                    win = simulate_battle(team_max_hp, enemy_max_hp, player_stats, boss_damage, role_combo, lds_array, team_max_hp)
                    if win:
                        results[key]["wins"] += 1
                    else:
                        results[key]["losses"] += 1
    
    with open(OUTPUT_FILE, 'w') as f:
        for key in sorted(results.keys()):
            w = results[key]["wins"]
            l = results[key]["losses"]
            result_str = "win" if w > l else "loss"
            f.write(f"{key},{result_str}\n")
    
    print(f"Saved results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()