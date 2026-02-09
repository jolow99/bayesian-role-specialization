"""
run_sim.py: Main entry point for running the simulation using a specific environment.
"""

import sys
import os
import argparse
import jax
import jax.numpy as jnp

def setup_environment(env_id):
    """
    Adds the specific environment directory to path so 'config' can be imported.
    """
    env_path = os.path.abspath(os.path.join("envs", str(env_id)))
    if not os.path.exists(env_path):
        raise FileNotFoundError(f"Environment {env_id} not found at {env_path}")
    
    # Insert at 0 to ensure this config is loaded instead of root config
    sys.path.insert(0, env_path)
    return env_path

def main():
    parser = argparse.ArgumentParser(description="Run Markov Game Simulation")
    parser.add_argument("--env_id", type=str, required=True, help="ID of the environment in /envs/ to use")
    parser.add_argument("--steps", type=int, default=10, help="Number of simulation steps")
    parser.add_argument("--initial_intent", type=int, choices=[0, 1], default=1, 
                        help="Initial enemy intent (0: No Attack, 1: Attack)")
    args = parser.parse_args()

    # 1. Setup Environment Path
    env_path = setup_environment(args.env_id)

    # 2. Import Modules (Must happen AFTER setup_environment)
    # These imports will now pick up the config.py from envs/<id>/
    import config
    from config import (
        ENEMY_ATTACK_INTENT, ENEMY_NO_ATTACK_INTENT, 
        TEAM_MAX_HP, ENEMY_MAX_HP, H, W, 
        get_intent_and_hps_from_state
    )
    
    # IMPORT FROM R instead of config
    from R import is_terminal

    # Import local modules
    from inference import role_inference
    from mechanics import game_step, choose_action, softmax_dist_over_roles

    # 3. Load Value Function
    values_path = os.path.join(env_path, "values.npy")
    if not os.path.exists(values_path):
        raise FileNotFoundError("values.npy not found. Please run search.py or V.py first.")

    values = jnp.load(values_path)

    # 4. Setup output file (save in global directory)
    output_filename = f"{args.initial_intent}.txt"
    output_path = output_filename  # Save in current working directory
    
    # 5. Initialize Simulation
    key = jax.random.PRNGKey(42)
    
    # Use the command line argument for initial intent
    start_intent = args.initial_intent
    
    initial_state_idx = start_intent * (H * W) + TEAM_MAX_HP * W + ENEMY_MAX_HP
    intent, team_hp, enemy_hp = get_intent_and_hps_from_state(initial_state_idx)
    
    # Initialize Prior (Uniform)
    role_prior = jnp.ones((3, 3, 3))
    
    # Initial arbitrary roles for first sampling
    current_roles = [0, 0, 0] 

    # Open file for writing (only for the table and game data)
    with open(output_path, 'w') as log_file:
        log_file.write(f"{'Step':<5} | {'Intent':<10} | {'Team HP':<8} | {'Enemy HP':<8} | {'Roles':<15} | {'Actions':<15}\n")
        log_file.write("-" * 80 + '\n')

        role_names = {0: "ATK", 1: "DEF", 2: "HEL"}
        action_names = {0: "ATK", 1: "DEF", 2: "HEL"}

        for t in range(args.steps):
            current_state = intent * (H * W) + team_hp * W + enemy_hp
            
            # Check termination
            if is_terminal(current_state):
                log_file.write("-" * 80 + '\n')
                log_file.write(f"FINAL | {'-':<10} | {team_hp:<8} | {enemy_hp:<8} | {'-':<15} | {'-':<15}\n")
                break

            # A. Sample Roles (Every 2 turns or initially)
            if t % 2 == 0:
                new_roles = list(current_roles)
                for i in range(3):
                    key, subkey = jax.random.split(key)
                    probs = softmax_dist_over_roles(i, current_state, role_prior, values)
                    role = jax.random.categorical(subkey, jnp.log(probs))
                    new_roles[i] = int(role)
                current_roles = new_roles

            # B. Choose Actions
            current_actions = []
            for i in range(3):
                key, subkey = jax.random.split(key)
                action = choose_action(current_roles[i], intent, team_hp, key=subkey)
                current_actions.append(int(action))

            # C. Update Beliefs (Role Inference)
            role_prior = role_inference(
                role_prior,
                current_actions[0], current_actions[1], current_actions[2],
                current_state
            )

            # Write Step Info
            r_str = f"{role_names[current_roles[0]]},{role_names[current_roles[1]]},{role_names[current_roles[2]]}"
            a_str = f"{action_names[current_actions[0]]},{action_names[current_actions[1]]},{action_names[current_actions[2]]}"
            intent_str = "ATTACK" if intent == ENEMY_ATTACK_INTENT else "WAIT"
            
            log_file.write(f"{t:<5} | {intent_str:<10} | {team_hp:<8} | {enemy_hp:<8} | {r_str:<15} | {a_str:<15}\n")

            # D. Advance Environment
            key, subkey = jax.random.split(key)
            intent, team_hp, enemy_hp = game_step(
                intent, team_hp, enemy_hp, jnp.array(current_actions), key=subkey
            )

if __name__ == "__main__":
    main()