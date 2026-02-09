import shutil
import numpy as np
from pathlib import Path

# --- Constants & Configuration ---

PLAYER_STATS_SETS = [
    "411_222_222",
    "141_222_222",
    "114_222_222",
]

# Role Definitions
F, T, M = 0, 1, 2

TRACKED_COMBOS = {
    "411_222_222": [
        ("FFF", "MFF"),
        ("FFF", "MFM"),
        ("FFF", "MFT"),
        ("FFF", "MMM"),
        ("FFF", "MMT"),
        ("FFF", "MTT"),
        ("FFF", "TFF"),
        ("FFF", "TFM"),
        ("FFF", "TFT"),
        ("FFF", "TMM"),
        ("FFF", "TMT"),
        ("FFF", "TTT"),
        ("FFM", "MFF"),
        ("FFM", "MFM"),
        ("FFM", "MFT"),
        ("FFM", "MMM"),
        ("FFM", "MMT"),
        ("FFM", "MTT"),
        ("FFM", "TFF"),
        ("FFM", "TFM"),
        ("FFM", "TFT"),
        ("FFM", "TMM"),
        ("FFM", "TMT"),
        ("FFM", "TTT"),
        ("FFT", "MFF"),
        ("FFT", "MFM"),
        ("FFT", "MFT"),
        ("FFT", "MMM"),
        ("FFT", "MMT"),
        ("FFT", "MTT"),
        ("FFT", "TFF"),
        ("FFT", "TFM"),
        ("FFT", "TFT"),
        ("FFT", "TMM"),
        ("FFT", "TMT"),
        ("FFT", "TTT"),
        ("FMM", "MFF"),
        ("FMM", "MFM"),
        ("FMM", "MFT"),
        ("FMM", "MMM"),
        ("FMM", "MMT"),
        ("FMM", "MTT"),
        ("FMM", "TFF"),
        ("FMM", "TFM"),
        ("FMM", "TFT"),
        ("FMM", "TMM"),
        ("FMM", "TMT"),
        ("FMM", "TTT"),
        ("FMT", "MFF"),
        ("FMT", "MFM"),
        ("FMT", "MFT"),
        ("FMT", "MMM"),
        ("FMT", "MMT"),
        ("FMT", "MTT"),
        ("FMT", "TFF"),
        ("FMT", "TFM"),
        ("FMT", "TFT"),
        ("FMT", "TMM"),
        ("FMT", "TMT"),
        ("FMT", "TTT"),
        ("FTT", "MFF"),
        ("FTT", "MFM"),
        ("FTT", "MFT"),
        ("FTT", "MMM"),
        ("FTT", "MMT"),
        ("FTT", "MTT"),
        ("FTT", "TFF"),
        ("FTT", "TFM"),
        ("FTT", "TFT"),
        ("FTT", "TMM"),
        ("FTT", "TMT"),
        ("FTT", "TTT"),
        ("TFM", "TMM"),
    ],
    "141_222_222": [
        ("TFF", "FFF"),
        ("TFF", "FFM"),
        ("TFF", "FFT"),
        ("TFF", "FMM"),
        ("TFF", "FMT"),
        ("TFF", "FTT"),
        ("TFF", "MFF"),
        ("TFF", "MFM"),
        ("TFF", "MFT"),
        ("TFF", "MMM"),
        ("TFF", "MMT"),
        ("TFF", "MTT"),
        ("TFM", "FFF"),
        ("TFM", "FFM"),
        ("TFM", "FFT"),
        ("TFM", "FMM"),
        ("TFM", "FMT"),
        ("TFM", "FTT"),
        ("TFM", "MFF"),
        ("TFM", "MFM"),
        ("TFM", "MFT"),
        ("TFM", "MMM"),
        ("TFM", "MMT"),
        ("TFM", "MTT"),
        ("TFT", "FFF"),
        ("TFT", "FFM"),
        ("TFT", "FFT"),
        ("TFT", "FMM"),
        ("TFT", "FMT"),
        ("TFT", "FTT"),
        ("TFT", "MFF"),
        ("TFT", "MFM"),
        ("TFT", "MFT"),
        ("TFT", "MMM"),
        ("TFT", "MMT"),
        ("TFT", "MTT"),
        ("TMM", "FFF"),
        ("TMM", "FFM"),
        ("TMM", "FFT"),
        ("TMM", "FMM"),
        ("TMM", "FMT"),
        ("TMM", "FTT"),
        ("TMM", "MFF"),
        ("TMM", "MFM"),
        ("TMM", "MFT"),
        ("TMM", "MMM"),
        ("TMM", "MMT"),
        ("TMM", "MTT"),
        ("TMT", "FFF"),
        ("TMT", "FFM"),
        ("TMT", "FFT"),
        ("TMT", "FMM"),
        ("TMT", "FMT"),
        ("TMT", "FTT"),
        ("TMT", "MFF"),
        ("TMT", "MFM"),
        ("TMT", "MFT"),
        ("TMT", "MMM"),
        ("TMT", "MMT"),
        ("TMT", "MTT"),
        ("TTT", "FFF"),
        ("TTT", "FFM"),
        ("TTT", "FFT"),
        ("TTT", "FMM"),
        ("TTT", "FMT"),
        ("TTT", "FTT"),
        ("TTT", "MFF"),
        ("TTT", "MFM"),
        ("TTT", "MFT"),
        ("TTT", "MMM"),
        ("TTT", "MMT"),
        ("TTT", "MTT"),
    ],
    "114_222_222": [
        ("MFF", "FFF"),
        ("MFF", "FFM"),
        ("MFF", "FFT"),
        ("MFF", "FMM"),
        ("MFF", "FMT"),
        ("MFF", "FTT"),
        ("MFF", "TFF"),
        ("MFF", "TFM"),
        ("MFF", "TFT"),
        ("MFF", "TMM"),
        ("MFF", "TMT"),
        ("MFF", "TTT"),
        ("MFM", "FFF"),
        ("MFM", "FFM"),
        ("MFM", "FFT"),
        ("MFM", "FMM"),
        ("MFM", "FMT"),
        ("MFM", "FTT"),
        ("MFM", "TFF"),
        ("MFM", "TFM"),
        ("MFM", "TFT"),
        ("MFM", "TMM"),
        ("MFM", "TMT"),
        ("MFM", "TTT"),
        ("MFT", "FFF"),
        ("MFT", "FFM"),
        ("MFT", "FFT"),
        ("MFT", "FMM"),
        ("MFT", "FMT"),
        ("MFT", "FTT"),
        ("MFT", "TFF"),
        ("MFT", "TFM"),
        ("MFT", "TFT"),
        ("MFT", "TMM"),
        ("MFT", "TMT"),
        ("MFT", "TTT"),
        ("MMM", "FFF"),
        ("MMM", "FFM"),
        ("MMM", "FFT"),
        ("MMM", "FMM"),
        ("MMM", "FMT"),
        ("MMM", "FTT"),
        ("MMM", "TFF"),
        ("MMM", "TFM"),
        ("MMM", "TFT"),
        ("MMM", "TMM"),
        ("MMM", "TMT"),
        ("MMM", "TTT"),
        ("MMT", "FFF"),
        ("MMT", "FFM"),
        ("MMT", "FFT"),
        ("MMT", "FMM"),
        ("MMT", "FMT"),
        ("MMT", "FTT"),
        ("MMT", "TFF"),
        ("MMT", "TFM"),
        ("MMT", "TFT"),
        ("MMT", "TMM"),
        ("MMT", "TMT"),
        ("MMT", "TTT"),
        ("MTT", "FFF"),
        ("MTT", "FFM"),
        ("MTT", "FFT"),
        ("MTT", "FMM"),
        ("MTT", "FMT"),
        ("MTT", "FTT"),
        ("MTT", "TFF"),
        ("MTT", "TFM"),
        ("MTT", "TFT"),
        ("MTT", "TMM"),
        ("MTT", "TMT"),
        ("MTT", "TTT"),
    ],
}

# # --- Tracked Combos & Forced Counterparts ---
# TRACKED_COMBOS = {
#     "411_222_222": [
#         # (Target_Combo, Forced_Counterpart)
#         ("FTM", "MFT"), # Bots play FT -> Human forced to M (MFT) OK
#         ("FTM", "TFM"), # Bots play FM -> Human forced to T (TFM) OK
#         ("FMM", "MFM"), # Bots play FM -> Human forced to M (MFM) OK
#         ("FFT", "TFF"), # Bots play FF -> Human forced to M (MFF) OK
#         ("FTM", "TFF"), # Bots play FF -> Human forced to T (TMM) OK
#     ],
#     "141_222_222": [
#         ("TFM", "MFT"), # High DEF P1. Bots FT -> P1 forced to M (MFT) OK
#         ("TFM", "FMT"), # High DEF P1. Bots FT -> P1 forced to M (MFF) OK
#         ("TFF", "FFT"), # High DEF P1. Bots MM -> P1 forced to F (FMM) OK
#         ("TMM", "FFT"), # High DEF P1. Bots MM -> P1 forced to F (FMM) OK
#         ("TMM", "FMT"), # OK
#     ],
#     "114_222_222": [
#         ("MFT", "TFF"), # High DEF P1. Bots FT -> P1 forced to M (MFT) NOT OK
#         ("MFT", "FFF"), # High DEF P1. Bots FT -> P1 forced to M (MFF) NOT OK
#         ("MFF", "TMM"), # High DEF P1. Bots MM -> P1 forced to F (FMM) OK
#         ("MFF", "FFT"), # High DEF P1. Bots MM -> P1 forced to F (FMM) OK
#         ("MFF", "FTM"), # OK
#     ],
# }

# # Role Index Mapping 
# ROLE_TO_INDEX = {
#     # Targets
#     "FTM": (F, T, M), "FMM": (F, M, M), "FFM": (F, F, M), "FFT": (F, F, T),
#     "TFM": (T, F, M), "TFF": (T, F, F), "TMM": (T, M, M),
#     # Forced / Counterparts
#     "MFT": (M, F, T), "MFM": (M, F, M), "MFF": (M, F, F), 
#     "TFF": (T, F, F), "TMM": (T, M, M), "FMM": (F, M, M),
# }

ROLE_TO_INDEX = {
    # Combos starting with F
    "FFF": (F, F, F),
    "FFM": (F, F, M),
    "FFT": (F, F, T),
    "FMM": (F, M, M),
    "FMT": (F, M, T),
    "FTT": (F, T, T),
    # Combos starting with T
    "TFF": (T, F, F),
    "TFM": (T, F, M),
    "TFT": (T, F, T),
    "TMM": (T, M, M),
    "TMT": (T, M, T),
    "TTT": (T, T, T),
    # Combos starting with M
    "MFF": (M, F, F),
    "MFM": (M, F, M),
    "MFT": (M, F, T),
    "MMM": (M, M, M),
    "MMT": (M, M, T),
    "MTT": (M, T, T),
}

# --- Helper Functions ---

def role_combo_to_index(roles):
    """Convert (r1, r2, r3) to flat index (Base 3 Big-Endian)."""
    return roles[0] * 9 + roles[1] * 3 + roles[2]

def index_to_role_combo(idx):
    """Convert flat index back to (r1, r2, r3)."""
    return (idx // 9, (idx % 9) // 3, idx % 3)

def get_canonical_map(stat_key):
    """Groups the 27 indices into canonical sets based on player symmetry."""
    player_stats = stat_key.split('_')
    canonical_groups = {}
    
    for idx in range(27):
        roles = index_to_role_combo(idx)
        config = []
        for i in range(3):
            config.append((player_stats[i], roles[i]))
        config.sort()
        
        key = tuple(config)
        if key not in canonical_groups:
            canonical_groups[key] = []
        canonical_groups[key].append(idx)
        
    return canonical_groups

def get_expected_value(values, prob_attack=1.0):
    """Calculate expected value at starting state."""
    return values[:, 1, -1, -1]

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

def get_key_from_role_name(role_name, canonical_map):
    if role_name not in ROLE_TO_INDEX:
        return None
    role_combo = ROLE_TO_INDEX[role_name]
    role_idx = role_combo_to_index(role_combo)
    for key, indices in canonical_map.items():
        if role_idx in indices:
            return key
    return None

# --- Main Execution ---

output_dir = Path("optimal_envs_for_bots")
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
    if np.max(np.abs(ev)) < 1e-10: continue

    # 2. Canonical Grouping
    canonical_map = get_canonical_map(stat_key)
    
    # 3. Determine Optimal Strategies
    group_max_evs = {}
    for key, indices in canonical_map.items():
        group_max_evs[key] = np.max(ev[indices])
        
    global_max = max(group_max_evs.values())
    
    optimal_keys = set()
    for key, val in group_max_evs.items():
        if np.isclose(val, global_max, rtol=1e-5):
            optimal_keys.add(key)

    # LOGIC 1: Relaxed Constraint (<= 5 optimal strategies allowed)
    # if len(optimal_keys) > 3:
    #     continue

    # 4. Check Tracked Combos
    tracked_pairs = TRACKED_COMBOS.get(stat_key, [])
    
    for role_name, forced_name in tracked_pairs:
        target_key = get_key_from_role_name(role_name, canonical_map)
        forced_key = get_key_from_role_name(forced_name, canonical_map)
        
        if not target_key: continue
        
        # Check if target is optimal
        if target_key in optimal_keys:
            
            # LOGIC 2: Conflict Check
            # If the Forced Counterpart is ALSO optimal, ignore this env
            if forced_key and forced_key in optimal_keys:
                continue

            # Save result in folder format: Target_Forced
            folder_name = f"{role_name}_{forced_name}"
            
            stat_dir = output_dir / stat_key
            stat_dir.mkdir(exist_ok=True)
            
            role_dir = stat_dir / folder_name
            role_dir.mkdir(exist_ok=True)
            
            with open(role_dir / "env_ids.txt", "a") as f:
                f.write(f"{env_id}\n")

print("Done! Results saved in 'optimal_envs_for_bots/'.")