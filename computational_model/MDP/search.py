import subprocess
import itertools
import os
import shutil
from multiprocessing import Pool

PARAM_GRID = {
    'TEAM_MAX_HP': [5, 10, 15][::-1],
    'ENEMY_MAX_HP': [10, 15, 20, 25, 30][::-1],
    'PLAYER_STATS_SETS': [
        [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
        [[4, 1, 1], [1, 4, 1], [1, 1, 4]],
        [[4, 1, 1], [2, 2, 2], [2, 2, 2]],
        [[1, 4, 1], [2, 2, 2], [2, 2, 2]],
        [[1, 1, 4], [2, 2, 2], [2, 2, 2]],
    ],
    'BOSS_DAMAGE': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10][::-1],
    'ENEMY_ATTACK_PROB': [0.2, 0.25, 0.33, 0.4, 0.5, 0.6, 0.66, 0.75, 0.8, 1.0], # grid add .25 and .5
}

def create_config(params, filepath):
    stats = params['PLAYER_STATS_SETS']
    content = f'''import jax
import jax.numpy as jnp
from itertools import product

ATTACKER, DEFENDER, HEALER = 0, 1, 2
ATTACK, DEFEND, HEAL = 0, 1, 2
ENEMY_NO_ATTACK_INTENT, ENEMY_ATTACK_INTENT = 0, 1

ROLES = [ATTACKER, DEFENDER, HEALER]
ACTIONS = [ATTACK, DEFEND, HEAL]
ROLE_COMBOS = jnp.array(list(product(ROLES, repeat=3)))
ROLE_INDICES = jnp.array(list(range(27)))

TEAM_MAX_HP = {params['TEAM_MAX_HP']}
ENEMY_MAX_HP = {params['ENEMY_MAX_HP']}

PLAYER_STATS = jnp.array({stats})
ATTACK_STATS = PLAYER_STATS[:, ATTACK]
DEFEND_STATS = PLAYER_STATS[:, DEFEND]
HEAL_STATS = PLAYER_STATS[:, HEAL]

BOSS_DAMAGE = {params['BOSS_DAMAGE']}
ENEMY_ATTACK_PROB = {params['ENEMY_ATTACK_PROB']}

EPSILON, HORIZON = 1e-10, 10
H, W = TEAM_MAX_HP + 1, ENEMY_MAX_HP + 1
S = jnp.arange(2 * H * W)
ACTION_PROFILES = jnp.array(list(product(ACTIONS, repeat=3)))
A = jnp.arange(len(ACTION_PROFILES))

@jax.jit
def get_intent_and_hps_from_state(s):
    i = s // (H * W)
    rem = s % (H * W)
    return i, rem // W, rem % W
'''
    with open(filepath, 'w') as f:
        f.write(content)

def run_sim(args):
    idx, combo, param_names = args
    params = dict(zip(param_names, combo))
    temp_dir = f'temp_{idx}'
    final_dir = f'envs/{idx}'
    
    try:
        os.makedirs(temp_dir, exist_ok=True)
        create_config(params, f'{temp_dir}/config.py')
        
        for f in ['V.py', 'pi.py', 'R.py', 'T.py']:
            shutil.copy(f, temp_dir)
        
        result = subprocess.run(['python', 'V.py'], capture_output=True, timeout=120, cwd=temp_dir)
        
        if result.returncode == 0 and os.path.exists(f'{temp_dir}/V.npy'):
            os.makedirs(final_dir, exist_ok=True)
            shutil.copy(f'{temp_dir}/config.py', f'{final_dir}/config.py')
            shutil.copy(f'{temp_dir}/V.npy', f'{final_dir}/values.npy')
            shutil.rmtree(temp_dir)
            
            stats = params['PLAYER_STATS_SETS']
            desc = f"T{params['TEAM_MAX_HP']}_E{params['ENEMY_MAX_HP']}_S{stats[0][0]}{stats[0][1]}{stats[0][2]}_{stats[1][0]}{stats[1][1]}{stats[1][2]}_{stats[2][0]}{stats[2][1]}{stats[2][2]}_B{params['BOSS_DAMAGE']}_P{int(params['ENEMY_ATTACK_PROB']*100)}"
            return idx, desc, True
        else:
            # Print error for debugging
            if result.returncode != 0:
                print(f"\nConfig {idx} failed with return code {result.returncode}")
                print(f"STDERR: {result.stderr.decode()[:200]}")
            shutil.rmtree(temp_dir)
            return idx, None, False
    except Exception as e:
        print(f"\nConfig {idx} exception: {str(e)[:200]}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return idx, None, False

if __name__ == "__main__":
    if os.path.exists('envs'):
        shutil.rmtree('envs')
    os.makedirs('envs')
    
    combos = list(itertools.product(*PARAM_GRID.values()))
    param_names = list(PARAM_GRID.keys())
    
    print(f"Running {len(combos)} configs across 3 cores")
    
    args_list = [(i, combo, param_names) for i, combo in enumerate(combos, 1)]
    
    results = []
    batch_size = 3
    
    for batch_start in range(0, len(args_list), batch_size):
        batch = args_list[batch_start:batch_start + batch_size]
        with Pool(processes=3) as pool:
            batch_results = pool.map(run_sim, batch)
        results.extend(batch_results)
        n_success = sum(1 for _, _, s in results if s)
        print(f"Progress: {len(results)}/{len(combos)} ({n_success} successful)")
    
    results.sort()
    
    with open('env_mapping.txt', 'w') as f:
        f.write("ID\tConfiguration\n")
        for idx, desc, success in results:
            if success:
                f.write(f"{idx}\t{desc}\n")
    
    n_success = sum(1 for _, _, s in results if s)
    print(f"Done: {n_success}/{len(combos)} saved")