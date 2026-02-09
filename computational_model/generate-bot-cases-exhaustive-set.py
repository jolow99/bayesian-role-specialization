"""
Generate all possible TRACKED_COMBOS entries for different player stat sets.

Key rules:
1. First half (Target): First letter corresponds to the position with value 4
   - "411_222_222" -> F (Forward) at position 0 has 4
   - "141_222_222" -> T (Tank) at position 1 has 4
   - "114_222_222" -> M (Mage) at position 2 has 4

2. Second half (Forced): First letter must be DIFFERENT from first letter in Target

3. Since bot profiles are "_22_222_222", positions are symmetric (_xy == _yx)
   - This means we only need to consider unique unordered pairs for bot positions
"""

from itertools import combinations_with_replacement

# Role Definitions
ROLES = ['F', 'T', 'M']
F, T, M = 0, 1, 2

def normalize_combo(combo):
    """
    Normalize a combo by sorting the bot positions (positions 1 and 2).
    Since bots are symmetric (_xy == _yx), we normalize to canonical form.
    
    Example: "FTM" and "FMT" both normalize to "FMT" (alphabetically sorted bots)
    """
    if len(combo) != 3:
        return combo
    
    first = combo[0]
    bots = sorted([combo[1], combo[2]])
    return first + bots[0] + bots[1]


def generate_all_combos_for_role(role):
    """
    Generate all possible 3-letter combos starting with the given role.
    
    Args:
        role: The role that must be in the first position
    
    Returns:
        Set of normalized combo strings
    """
    combos = set()
    
    # Generate all possible combinations for the two bot positions
    for bot1 in ROLES:
        for bot2 in ROLES:
            combo = role + bot1 + bot2
            # Normalize to handle symmetry
            normalized = normalize_combo(combo)
            combos.add(normalized)
    
    return combos


def generate_combos_for_stat_set(stat_set):
    """
    Generate all tracked combos for a given stat set.
    This generates the EXHAUSTIVE set of all valid pairings.
    
    Args:
        stat_set: String like "411_222_222" indicating which position has 4
    
    Returns:
        List of (target_combo, forced_combo) tuples
    """
    # Determine which role has the high stat (4)
    if stat_set == "411_222_222":
        high_stat_role = 'F'
        high_stat_idx = 0
    elif stat_set == "141_222_222":
        high_stat_role = 'T'
        high_stat_idx = 1
    elif stat_set == "114_222_222":
        high_stat_role = 'M'
        high_stat_idx = 2
    else:
        raise ValueError(f"Unknown stat set: {stat_set}")
    
    # Generate all possible target combos (start with high_stat_role)
    target_combos = generate_all_combos_for_role(high_stat_role)
    
    # Generate all possible forced combos (start with any role EXCEPT high_stat_role)
    available_roles = [r for r in ROLES if r != high_stat_role]
    forced_combos = set()
    for role in available_roles:
        forced_combos.update(generate_all_combos_for_role(role))
    
    # Generate all possible pairings (Cartesian product)
    combos = []
    for target in sorted(target_combos):
        for forced in sorted(forced_combos):
            combos.append((target, forced))
    
    return combos


def print_tracked_combos():
    """Print all TRACKED_COMBOS in the format matching the original code."""
    
    stat_sets = [
        "411_222_222",
        "141_222_222", 
        "114_222_222"
    ]
    
    print("TRACKED_COMBOS = {")
    
    for stat_set in stat_sets:
        combos = generate_combos_for_stat_set(stat_set)
        print(f'    "{stat_set}": [')
        
        for target, forced in combos:
            print(f'        ("{target}", "{forced}"),')
        
        print("    ],")
    
    print("}")
    print()
    
    # Print statistics
    print("Statistics:")
    for stat_set in stat_sets:
        combos = generate_combos_for_stat_set(stat_set)
        print(f"  {stat_set}: {len(combos)} combinations")


def verify_combos():
    """Verify that generated combos follow all rules."""
    
    stat_sets = {
        "411_222_222": 'F',
        "141_222_222": 'T',
        "114_222_222": 'M'
    }
    
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    all_valid = True
    
    for stat_set, expected_first in stat_sets.items():
        print(f"\nVerifying {stat_set}:")
        combos = generate_combos_for_stat_set(stat_set)
        
        for target, forced in combos:
            # Rule 1: Target must start with the high stat role
            if target[0] != expected_first:
                print(f"  ERROR: Target {target} doesn't start with {expected_first}")
                all_valid = False
            
            # Rule 2: Forced must NOT start with the same role as target
            if forced[0] == target[0]:
                print(f"  ERROR: Forced {forced} starts with same role as target {target}")
                all_valid = False
            
            # Rule 3: Bot roles (positions 1,2) should be from valid set
            if target[1] not in ROLES or target[2] not in ROLES:
                print(f"  ERROR: Target {target} has invalid bot roles")
                all_valid = False
                
            if forced[1] not in ROLES or forced[2] not in ROLES:
                print(f"  ERROR: Forced {forced} has invalid bot roles")
                all_valid = False
        
        print(f"  ✓ All {len(combos)} combinations valid for {stat_set}")
    
    if all_valid:
        print("\n✓ All combinations pass verification!")
    else:
        print("\n✗ Some combinations failed verification")
    
    return all_valid


def generate_role_to_index_mapping():
    """
    Generate the complete ROLE_TO_INDEX mapping from all tracked combos.
    This extracts all unique combo strings that appear in the pairs
    and maps them to their index tuples.
    """
    stat_sets = [
        "411_222_222",
        "141_222_222", 
        "114_222_222"
    ]
    
    # Collect all unique combos from all stat sets
    all_combos = set()
    
    for stat_set in stat_sets:
        combos = generate_combos_for_stat_set(stat_set)
        for target, forced in combos:
            all_combos.add(target)
            all_combos.add(forced)
    
    # Convert to role-to-index mapping
    role_name_to_idx = {'F': F, 'T': T, 'M': M}
    
    role_to_index = {}
    for combo in sorted(all_combos):
        # Convert each character to its role index
        indices = tuple(role_name_to_idx[char] for char in combo)
        role_to_index[combo] = indices
    
    return role_to_index


def print_role_to_index_mapping():
    """Print the ROLE_TO_INDEX mapping in the format matching the original code."""
    
    mapping = generate_role_to_index_mapping()
    
    print("\nROLE_TO_INDEX = {")
    
    # Group by first letter for better readability
    for first_role in ['F', 'T', 'M']:
        combos_with_first = {k: v for k, v in mapping.items() if k[0] == first_role}
        
        if combos_with_first:
            print(f"    # Combos starting with {first_role}")
            for combo in sorted(combos_with_first.keys()):
                indices = combos_with_first[combo]
                # Format indices as (F, T, M) or actual values
                indices_str = ', '.join(
                    ['F' if idx == F else 'T' if idx == T else 'M' 
                     for idx in indices]
                )
                print(f'    "{combo}": ({indices_str}),')
    
    print("}")
    print(f"\nTotal unique combos: {len(mapping)}")


if __name__ == "__main__":
    print("Generating all possible TRACKED_COMBOS...\n")
    print_tracked_combos()
    verify_combos()
    print("\n" + "="*60)
    print("ROLE_TO_INDEX MAPPING")
    print("="*60)
    print_role_to_index_mapping()
