"""
Pretty print value function for intent and no-intent starting states with weighted average.
"""
import jax.numpy as jnp

# Load the values
file_str = "envs/7457/values.npy"
# file_str = "values.npy"
values = jnp.load(file_str)

# Initialize defaults
start_team_hp = 10
start_enemy_hp = 10
enemy_attack_prob = 0.5  # Default in case not found

# Read config to get max HP values and Attack Probability
with open(file_str.replace("values.npy", "config.py")) as f:
    config_text = f.read()
    for line in config_text.split('\n'):
        # Strip whitespace for safer parsing
        clean_line = line.strip()
        
        if clean_line.startswith('TEAM_MAX_HP'):
            start_team_hp = int(clean_line.split('=')[1].strip())
        elif clean_line.startswith('ENEMY_MAX_HP'):
            start_enemy_hp = int(clean_line.split('=')[1].strip())
        elif clean_line.startswith('ENEMY_ATTACK_PROB'):
            enemy_attack_prob = float(clean_line.split('=')[1].strip())

print(f"Loaded Configuration -> Team HP: {start_team_hp}, Enemy HP: {start_enemy_hp}, Attack Prob: {enemy_attack_prob}")

# Role names matching config.py constants
role_names = ["Attacker", "Defender", "Healer"]

# Collect all results first
results = []
for role_idx in range(27):
    # Extract individual roles from role_idx
    r0 = role_idx % 3
    r1 = (role_idx // 3) % 3
    r2 = (role_idx // 9) % 3
    role_combo = f"{role_names[r2]},{role_names[r1]},{role_names[r0]}"
    
    # Access values: shape is (27, 2, H, W)
    # intent=0 for no intent, intent=1 for attack intent
    # no_intent_val = values[role_idx, 0, start_team_hp, start_enemy_hp]
    intent_val = values[role_idx, 1, start_team_hp, start_enemy_hp]
    
    # Calculate Weighted Average
    # Value = (No Intent * P(No Attack)) + (Intent * P(Attack))
    weighted_avg = intent_val
    
    results.append((role_combo, weighted_avg))

# Sort by weighted average (descending)
results.sort(key=lambda x: x[1], reverse=True)

# Print table
print(f"\n{'Role Combination':<30}{'Intent':>12}")
print("-" * 72)
for role_combo, intent_val in results:
    print(f"{role_combo:<30} {intent_val:12.6f}")