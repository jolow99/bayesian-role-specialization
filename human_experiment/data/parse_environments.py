#!/usr/bin/env python3
"""
Parse environment data from optimal_case_envs and deviate_case_envs
to generate roundConfigPools.json
"""

import os
import re
import json
from pathlib import Path

# Stat profile name mapping
STAT_PROFILE_NAMES = {
    "114_222_222": "imbalanced-oneunique-sup",
    "141_222_222": "imbalanced-oneunique-def",
    "411_222_222": "imbalanced-oneunique-str",
    "222_222_222": "balanced",
    "411_141_114": "imbalanced-allunique",
}

# Role character to number mapping
# F = Fighter (0), T = Tank (1), M = Medic (2)
ROLE_MAPPING = {"F": 0, "T": 1, "M": 2}


def parse_python_file_for_env_params(py_file_path):
    """Extract TEAM_MAX_HP, ENEMY_MAX_HP, BOSS_DAMAGE from a python file."""
    with open(py_file_path, "r") as f:
        content = f.read()

    team_max_hp = None
    enemy_max_hp = None
    boss_damage = None

    # Search for the variable assignments
    team_match = re.search(r"TEAM_MAX_HP\s*=\s*(\d+)", content)
    enemy_match = re.search(r"ENEMY_MAX_HP\s*=\s*(\d+)", content)
    boss_match = re.search(r"BOSS_DAMAGE\s*=\s*(\d+)", content)

    if team_match:
        team_max_hp = int(team_match.group(1))
    if enemy_match:
        enemy_max_hp = int(enemy_match.group(1))
    if boss_match:
        boss_damage = int(boss_match.group(1))

    return {
        "maxTeamHealth": team_max_hp,
        "maxEnemyHealth": enemy_max_hp,
        "bossDamage": boss_damage,
    }


def find_python_file(folder_path):
    """Find the .py file in a folder (excluding __pycache__)."""
    for file in os.listdir(folder_path):
        if file.endswith(".py"):
            return os.path.join(folder_path, file)
    return None


def parse_human_configs(data_dir):
    """Parse optimal_case_envs for humanConfigs."""
    configs = []
    lds_file = os.path.join(data_dir, "optimal_case_envs", "lds_arrays_human.txt")

    with open(lds_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Format: statprofile_playerconfig,attackprob,intentsequence
            # e.g., 114_222_222_MFF,0.8,1011111100
            parts = line.split(",")
            if len(parts) != 3:
                continue

            full_id = parts[0]
            attack_prob = float(parts[1])
            intent_sequence = parts[2]

            # Parse the stat profile and player config
            # Format: XXX_XXX_XXX_YYY where YYY is player config
            match = re.match(r"(\d+_\d+_\d+)_(\w+)", full_id)
            if not match:
                continue

            stat_profile_code = match.group(1)
            player_config = match.group(2)

            stat_profile_name = STAT_PROFILE_NAMES.get(stat_profile_code)
            if not stat_profile_name:
                print(f"Warning: Unknown stat profile {stat_profile_code}")
                continue

            # Find the python file for this environment
            env_folder = os.path.join(
                data_dir, "optimal_case_envs", stat_profile_code, player_config
            )
            if not os.path.exists(env_folder):
                print(f"Warning: Folder not found {env_folder}")
                continue

            py_file = find_python_file(env_folder)
            if not py_file:
                print(f"Warning: No python file found in {env_folder}")
                continue

            env_params = parse_python_file_for_env_params(py_file)

            # Convert player_config to optimal roles array (e.g., "TFF" -> [1, 0, 0])
            optimal_roles = [ROLE_MAPPING[c] for c in player_config]

            config = {
                "maxEnemyHealth": env_params["maxEnemyHealth"],
                "maxTeamHealth": env_params["maxTeamHealth"],
                "bossDamage": env_params["bossDamage"],
                "statProfile": stat_profile_name,
                "optimalRoles": optimal_roles,
                "playerDeviateProbability": 0.00,
                "enemyAttackProbability": attack_prob,
                "enemyIntentSequence": intent_sequence,
            }
            configs.append(config)

    # Add IDs
    for i, config in enumerate(configs, 1):
        config["id"] = i

    # Reorder keys to match expected format
    ordered_configs = []
    for config in configs:
        ordered_config = {
            "id": config["id"],
            "maxEnemyHealth": config["maxEnemyHealth"],
            "maxTeamHealth": config["maxTeamHealth"],
            "bossDamage": config["bossDamage"],
            "statProfile": config["statProfile"],
            "optimalRoles": config["optimalRoles"],
            "playerDeviateProbability": config["playerDeviateProbability"],
            "enemyAttackProbability": config["enemyAttackProbability"],
            "enemyIntentSequence": config["enemyIntentSequence"],
        }
        ordered_configs.append(ordered_config)

    return ordered_configs


def parse_bot_configs(data_dir):
    """Parse deviate_case_envs for botConfigs."""
    configs = []
    lds_file = os.path.join(data_dir, "deviate_case_envs", "bot_case_lds_arrays.txt")

    with open(lds_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Format: statprofile_optimalconfig_botconfig,attackprob,intentsequence
            # e.g., 114_222_222_MFF_FFT,1.0,1111111111
            parts = line.split(",")
            if len(parts) != 3:
                continue

            full_id = parts[0]
            attack_prob = float(parts[1])
            intent_sequence = parts[2]

            # Parse: XXX_XXX_XXX_YYY_ZZZ
            match = re.match(r"(\d+_\d+_\d+)_(\w+)_(\w+)", full_id)
            if not match:
                continue

            stat_profile_code = match.group(1)
            optimal_config = match.group(2)
            bot_config = match.group(3)

            stat_profile_name = STAT_PROFILE_NAMES.get(stat_profile_code)
            if not stat_profile_name:
                print(f"Warning: Unknown stat profile {stat_profile_code}")
                continue

            # Find the python file for this environment
            folder_name = f"{optimal_config}_{bot_config}"
            env_folder = os.path.join(
                data_dir, "deviate_case_envs", stat_profile_code, folder_name
            )
            if not os.path.exists(env_folder):
                print(f"Warning: Folder not found {env_folder}")
                continue

            py_file = find_python_file(env_folder)
            if not py_file:
                print(f"Warning: No python file found in {env_folder}")
                continue

            env_params = parse_python_file_for_env_params(py_file)

            # Convert optimal_config to optimal roles array (e.g., "TFF" -> [1, 0, 0])
            optimal_roles = [ROLE_MAPPING[c] for c in optimal_config]

            # Human's role is the first character of bot_config (e.g., "FFT" -> F -> 0)
            human_role = ROLE_MAPPING[bot_config[0]]

            # Create botPlayers from the remaining characters of bot_config
            # (e.g., "FFT" -> bots play "FT" -> [{role:0}, {role:1}])
            bot_players = []
            for char in bot_config[1:3]:  # Characters 2 and 3 for the 2 bots
                role = ROLE_MAPPING.get(char, 0)
                bot_players.append({"strategy": {"type": "fixed", "role": role}})

            config = {
                "maxEnemyHealth": env_params["maxEnemyHealth"],
                "maxTeamHealth": env_params["maxTeamHealth"],
                "bossDamage": env_params["bossDamage"],
                "statProfile": stat_profile_name,
                "optimalRoles": optimal_roles,
                "humanRole": human_role,
                "playerDeviateProbability": 0.00,
                "enemyAttackProbability": attack_prob,
                "enemyIntentSequence": intent_sequence,
                "botPlayers": bot_players,
            }
            configs.append(config)

    # Add IDs
    for i, config in enumerate(configs, 1):
        config["id"] = i

    # Reorder keys to match expected format
    ordered_configs = []
    for config in configs:
        ordered_config = {
            "id": config["id"],
            "maxEnemyHealth": config["maxEnemyHealth"],
            "maxTeamHealth": config["maxTeamHealth"],
            "bossDamage": config["bossDamage"],
            "statProfile": config["statProfile"],
            "optimalRoles": config["optimalRoles"],
            "humanRole": config["humanRole"],
            "playerDeviateProbability": config["playerDeviateProbability"],
            "enemyAttackProbability": config["enemyAttackProbability"],
            "enemyIntentSequence": config["enemyIntentSequence"],
            "botPlayers": config["botPlayers"],
        }
        ordered_configs.append(ordered_config)

    return ordered_configs


def main():
    # Get the data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir

    print("Parsing human configs from optimal_case_envs...")
    human_configs = parse_human_configs(data_dir)
    print(f"Found {len(human_configs)} human configs")

    print("\nParsing bot configs from deviate_case_envs...")
    bot_configs = parse_bot_configs(data_dir)
    print(f"Found {len(bot_configs)} bot configs")

    # Create the output structure
    output = {"humanConfigs": human_configs, "botConfigs": bot_configs}

    # Write to roundConfigPools.json
    output_path = script_dir.parent / "server" / "src" / "roundConfigPools.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWritten to {output_path}")

    # Also print the configs for verification
    print("\n--- Human Configs ---")
    for config in human_configs:
        print(json.dumps(config))

    print("\n--- Bot Configs ---")
    for config in bot_configs:
        print(json.dumps(config))


if __name__ == "__main__":
    main()
