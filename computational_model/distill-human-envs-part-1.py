import shutil
import numpy as np
from pathlib import Path

PLAYER_STATS_SETS = [
    "222_222_222",
    "411_141_114",
    "411_222_222",
    "141_222_222",
    "114_222_222",
]

# Role Definitions
F, T, M = 0, 1, 2 # F=Attacker, T=Defender, M=Support/Healer

# Tracked Combos

TRACKED_COMBOS = {
    "222_222_222": ["FTM", "TFF", "TMM"], # Balanced
    "411_141_114": ["FTM", "FTF", "FFM"], # All-Unique
    "411_222_222": ["FTM", "FMM", "FFM"], # One-Unique: High STR
    "141_222_222": ["TFM", "TFF", "TMM"], # One-Unique: High DEF
    "114_222_222": ["MFT", "MFF", "MMF"], # One-Unique: High SUP
}

# Role Index Mapping
ROLE_TO_INDEX = {
    "FTM": (F, T, M), "TFF": (T, F, F), "TMM": (T, M, M),
    "FTF": (F, T, F), "FFM": (F, F, M),
    "FMM": (F, M, M),
    "TFM": (T, F, M),
    "MFT": (M, F, T), "MFF": (M, F, F), "MMF": (M, M, F),
} 

# --- Helper Functions ---
def role_combo_to_index(roles): return roles[0] * 9 + roles[1] * 3 + roles[2]
def index_to_role_combo(idx): return (idx // 9, (idx % 9) // 3, idx % 3)

def get_canonical_map(stat_key):

    """Groups the 27 indices into canonical sets based on player symmetry."""
    player_stats = stat_key.split('_')
    canonical_groups = {}
    
    for idx in range(27):
        roles = index_to_role_combo(idx)
        # Create configuration: [(StatP1, RoleP1), (StatP2, RoleP2), ...]
        config = []
        for i in range(3): config.append((player_stats[i], roles[i]))
        config.sort() # Handle symmetry
        
        key = tuple(config)
        if key not in canonical_groups: canonical_groups[key] = []
        canonical_groups[key].append(idx)
        
    return canonical_groups

def get_expected_value(values, prob_attack=1.0): return values[:, 1, -1, -1]

def parse_env_mapping():
    
    env_info = {}
    try:
        with open("env_mapping.txt") as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2: continue
                
                env_id = parts[0]
                config = parts[1]
                
                s_idx = config.find('_S') + 2
                b_idx = config.find('_B')
                stat_key = config[s_idx:b_idx]
                
                p_idx = config.rfind('_P') + 2
                prob_attack = int(config[p_idx:]) / 100.0
                
                env_info[env_id] = (stat_key, prob_attack)
    except FileNotFoundError:
        print("Error: 'env_mapping.txt' not found.")
        return {}
    return env_info

# --- Main Execution ---

# Setup output directory
output_dir = Path("optimal_envs_for_humans")
if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir()

env_info = parse_env_mapping()

for env_id, (stat_key, prob_attack) in env_info.items():
    if stat_key not in PLAYER_STATS_SETS:
        continue
    
    env_dir = Path("envs") / env_id
    values_path = env_dir / "values.npy"
    
    if not values_path.exists():
        continue
    
    # 1. Load and Compute EV
    values = np.load(values_path)
    ev = get_expected_value(values, prob_attack)
    
    if np.max(np.abs(ev)) < 1e-10:
        continue

    # 2. Canonical Grouping
    canonical_map = get_canonical_map(stat_key)
    
    # 3. Determine Optimal Strategies
    group_max_evs = {}
    for key, indices in canonical_map.items():
        group_max_evs[key] = np.max(ev[indices])
        
    global_max = max(group_max_evs.values())
    
    # Identify the set of all optimal keys (strategies tied for best)
    optimal_keys = set()
    for key, val in group_max_evs.items():
        if np.isclose(val, global_max, rtol=1e-5):
            optimal_keys.add(key)
    
    # --- LOGIC UPDATE ---
    # Constraint: If there are too many optimal strategies (>3), 
    # the environment is likely a "wash" where choices don't matter much.
    # We skip those to focus on environments with distinct optimal metas.
    if len(optimal_keys) > 1:
        continue

    # 4. Check Tracked Combos against the Optimal Set
    tracked_role_names = TRACKED_COMBOS[stat_key]
    
    for role_name in tracked_role_names:
        # Get the canonical key for the current tracked role
        if role_name not in ROLE_TO_INDEX:
             continue
             
        role_combo = ROLE_TO_INDEX[role_name]
        role_idx = role_combo_to_index(role_combo)
        
        current_key = None
        for key, indices in canonical_map.items():
            if role_idx in indices:
                current_key = key
                break
        
        # Check if this specific role is one of the valid optimal keys
        if current_key in optimal_keys:
            stat_dir = output_dir / stat_key
            stat_dir.mkdir(exist_ok=True)
            
            role_dir = stat_dir / role_name
            role_dir.mkdir(exist_ok=True)
            
            with open(role_dir / "env_ids.txt", "a") as f:
                f.write(f"{env_id}\n")

print("Done! Results saved in 'optimal_envs_for_humans/'.")