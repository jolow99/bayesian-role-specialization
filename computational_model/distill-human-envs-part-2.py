import numpy as np
import shutil
from pathlib import Path
from collections import defaultdict

# --- CONFIGURATION ---

# Role Definitions
F, T, M = 0, 1, 2

# The 15 specific Stat-Role combos to analyze
TRACKED_COMBOS = {
    "222_222_222": ["FTM", "TFF", "TMM"],
    "411_141_114": ["FTM", "FTF", "FFM"],
    "411_222_222": ["FTM", "FMM", "FFM"],
    "141_222_222": ["TFM", "TFF", "TMM"],
    "114_222_222": ["MFT", "MFF", "MMF"],
}

ROLE_TO_INDEX = {
    "FTM": (F, T, M), "TFF": (T, F, F), "TMM": (T, M, M),
    "FTF": (F, T, F), "FFM": (F, F, M), "FMM": (F, M, M),
    "TFM": (T, F, M), "MFT": (M, F, T), "MFF": (M, F, F), 
    "MMF": (M, M, F),
}

# --- HELPER FUNCTIONS ---

def role_combo_to_index(roles):
    """Convert (r1, r2, r3) to flat index (0-26)."""
    return roles[0] * 9 + roles[1] * 3 + roles[2]

def index_to_role_combo(idx):
    """Convert flat index back to (r1, r2, r3)."""
    return (idx // 9, (idx % 9) // 3, idx % 3)

def get_role_str(r_tuple):
    """Helper to convert role tuple back to string (e.g., (0,1,2) -> 'FTM')"""
    mapping = {0: 'F', 1: 'T', 2: 'M'}
    return "".join([mapping[x] for x in r_tuple])

def get_canonical_key_for_index(stat_key, idx):
    """
    Generates the canonical signature for a specific raw index.
    Logic: Pairs (Stat, Role) for each player, then sorts to remove permutation order.
    """
    player_stats = stat_key.split('_')
    roles = index_to_role_combo(idx)
    
    config = []
    for i in range(3):
        config.append((player_stats[i], roles[i]))
    config.sort() # Sorts to handle permutations
    
    return tuple(config)

def get_canonical_groups(stat_key):
    """
    Returns dict: { canonical_key: [list_of_raw_indices] }
    """
    groups = defaultdict(list)
    for idx in range(27):
        key = get_canonical_key_for_index(stat_key, idx)
        groups[key].append(idx)
    return groups

def get_expected_value(values, prob_attack=1.0):
    """Calculate expected value at starting state."""
    return values[:, 1, -1, -1]

def parse_env_mapping():
    """Parses the mapping file to link EnvID -> StatBlock & ProbAttack"""
    env_info = {}
    try:
        with open("env_mapping.txt") as f:
            next(f) # Skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2: continue
                
                env_id = parts[0]
                config = parts[1]
                
                try:
                    s_idx = config.index('_S') + 2
                    b_idx = config.index('_B')
                    stat_key = config[s_idx:b_idx]
                    
                    p_idx = config.rindex('_P') + 2
                    prob_attack = int(config[p_idx:]) / 100.0
                    
                    env_info[env_id] = (stat_key, prob_attack)
                except (ValueError, IndexError):
                    continue
    except FileNotFoundError:
        print("Error: 'env_mapping.txt' not found.")
        return {}
    return env_info

# --- 1. DATA LOADING ---
print("Loading data...")

optimal_envs_human = Path("optimal_envs_for_humans")
env_info = parse_env_mapping()
env_data = defaultdict(dict)

# Load only necessary Stat Blocks
for stat_key in TRACKED_COMBOS.keys():
    stat_dir = optimal_envs_human / stat_key
    if not stat_dir.is_dir(): continue
    
    # Scan subdirectories to find valid env_ids for this stat block
    found_envs = set()
    for role_dir in stat_dir.iterdir():
        if not role_dir.is_dir(): continue
        
        env_ids_file = role_dir / "env_ids.txt"
        if not env_ids_file.exists(): continue
        
        with open(env_ids_file) as f:
            ids = [line.strip() for line in f if line.strip()]
            found_envs.update(ids)
    
    # Load data for these environments
    for env_id in found_envs:
        if env_id not in env_info: continue
        
        mapped_stat, prob_attack = env_info[env_id]
        if mapped_stat != stat_key: continue
        
        env_dir = Path("envs") / env_id
        values_path = env_dir / "values.npy"
        
        if not values_path.exists(): continue
        
        values = np.load(values_path)
        ev = get_expected_value(values, prob_attack)
        env_data[stat_key][env_id] = ev

# --- 2. ANALYSIS & EXPORT ---
output_base_dir = Path("human_case_envs")

print("\n" + "="*110)
print(f"{'STAT BLOCK':<15} | {'TARGET':<6} | {'ENV ID':<8} | {'DELTA':<8} | {'TARGET EV':<10} | {'RUNNER UP (ROLE)':<16}")
print("="*110)

for stat_key in sorted(TRACKED_COMBOS.keys()):
    target_roles = TRACKED_COMBOS[stat_key]
    
    # Pre-calculate canonical groups for this stat block
    canonical_groups = get_canonical_groups(stat_key)
    all_canonical_keys = list(canonical_groups.keys())
    
    for role_name in target_roles:
        if role_name not in ROLE_TO_INDEX:
            continue
            
        # Identify the Canonical Key for our Target Role
        raw_role_tuple = ROLE_TO_INDEX[role_name]
        raw_idx = role_combo_to_index(raw_role_tuple)
        target_canonical_key = get_canonical_key_for_index(stat_key, raw_idx)
        
        best_env_id = None
        max_delta = -float('inf')
        best_details = None
        
        # Iterate all loaded environments for this stat block
        if stat_key not in env_data:
            print(f"{stat_key:<15} | {role_name:<6} | {'No Data':<8} | ...")
            continue
            
        available_envs = sorted(env_data[stat_key].keys(), key=lambda x: (len(x), x))
        
        for env_id in available_envs:
            ev_array = env_data[stat_key][env_id]
            
            # --- FILTER ---
            # Ignore envs where almost all outcomes are zero/flat (noise filter > 1e-5)
            non_zero_count = np.sum(np.abs(ev_array) > 1e-5)
            if not (non_zero_count > 5):
                continue
            
            # A. Get Target EV (Max within its canonical group)
            target_indices = canonical_groups[target_canonical_key]
            target_ev = np.max(ev_array[target_indices])
            
            # B. Get Runner-Up EV (Best of all OTHER canonical groups)
            runner_up_ev = -float('inf')
            runner_up_key = None
            
            for other_key in all_canonical_keys:
                if other_key == target_canonical_key:
                    continue
                
                other_indices = canonical_groups[other_key]
                group_max = np.max(ev_array[other_indices])
                
                if group_max > runner_up_ev:
                    runner_up_ev = group_max
                    runner_up_key = other_key
            
            # C. Check if Target is Global Optimal
            if target_ev > runner_up_ev:
                delta = target_ev - runner_up_ev
                
                if delta > max_delta:
                    max_delta = delta
                    best_env_id = env_id
                    
                    if runner_up_key is not None:
                        rep_idx = canonical_groups[runner_up_key][0]
                        rep_roles = index_to_role_combo(rep_idx)
                        ru_name = get_role_str(rep_roles)
                    else:
                        ru_name = "None"

                    best_details = (target_ev, runner_up_ev, ru_name)
        
        # Output Result & Copy Files
        if best_env_id:
            t_ev, r_ev, r_name = best_details
            print(f"{stat_key:<15} | {role_name:<6} | {best_env_id:<8} | {max_delta:<8.4f} | {t_ev:<10.4f} | {r_name} ({r_ev:.4f})")
            
            # --- COPY LOGIC START ---
            src_config = Path("envs") / best_env_id / "config.py"
            dest_dir = output_base_dir / stat_key / role_name
            
            # Create destination directory (human_case_envs/STAT/ROLE/)
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            if src_config.exists():
                # Define destination file name as the env_id with .py extension
                dest_file = dest_dir / f"{best_env_id}.py"
                shutil.copy2(src_config, dest_file)
            else:
                print(f"   [!] Warning: config.py missing in {src_config}")
            # --- COPY LOGIC END ---

        else:
            print(f"{stat_key:<15} | {role_name:<6} | {'None':<8} | {'-':<8} | {'-':<10} | Not optimal anywhere")