"""
sim_all.py: Run simulations for all environments with both initial intents.
Creates sim_results/{env_id}/{0,1}.txt for each environment.
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path
import time
from tqdm import tqdm

def get_all_env_ids():
    """Get all environment IDs from the envs directory."""
    envs_dir = Path("envs")
    if not envs_dir.exists():
        return []
    
    env_ids = []
    for item in envs_dir.iterdir():
        if item.is_dir() and item.name.isdigit():
            env_ids.append(item.name)
    
    return sorted(env_ids, key=int)

def run_simulation(env_id, initial_intent, steps):
    """
    Run a single simulation and move the output file to the appropriate location.
    Args:
        env_id: Environment ID
        initial_intent: Initial intent (0 or 1)
        steps: Number of steps to run
    Returns:
        Tuple of (success, message)
    """
    try:
        result = subprocess.run(
            ["python", "run_sim.py", 
             "--env_id", str(env_id), 
             "--initial_intent", str(initial_intent),
             "--steps", str(steps)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            return (False, f"Exit code: {result.returncode}")
        
        # The script outputs to {initial_intent}.txt
        output_file = f"{initial_intent}.txt"
        
        if not os.path.exists(output_file):
            return (False, "Output file not found")
        
        # Move to final destination
        dest_dir = Path("sim_results") / str(env_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / f"{initial_intent}.txt"
        shutil.move(output_file, dest_file)
        
        return (True, "Success")
        
    except subprocess.TimeoutExpired:
        # Clean up output file if it exists
        output_file = f"{initial_intent}.txt"
        if os.path.exists(output_file):
            os.remove(output_file)
        return (False, "Timeout (60s)")
    except Exception as e:
        # Clean up output file if it exists
        output_file = f"{initial_intent}.txt"
        if os.path.exists(output_file):
            os.remove(output_file)
        return (False, f"Error: {str(e)}")

def main():
    start_time = time.time()
    
    env_ids = get_all_env_ids()
    if not env_ids:
        print("No environments found in 'envs' directory")
        return
    
    print(f"Found {len(env_ids)} environments")
    
    # Clean and recreate sim_results directory
    sim_results_dir = Path("sim_results")
    if sim_results_dir.exists():
        shutil.rmtree(sim_results_dir)
    sim_results_dir.mkdir(exist_ok=True)
    
    # Create all tasks
    tasks = []
    for env_id in env_ids:
        tasks.append((env_id, 0, 10))
        tasks.append((env_id, 1, 10))
    
    print(f"Running {len(tasks)} simulations sequentially...")
    
    # Run simulations sequentially with progress bar
    results = []
    with tqdm(total=len(tasks), desc="Simulations", unit="sim") as pbar:
        for env_id, initial_intent, steps in tasks:
            success, message = run_simulation(env_id, initial_intent, steps)
            results.append((env_id, initial_intent, success, message))
            
            # Update progress bar
            successful = sum(1 for _, _, s, _ in results if s)
            failed = len(results) - successful
            pbar.set_postfix(success=successful, failed=failed)
            pbar.update(1)
    
    # Final statistics
    total_sims = len(results)
    successful = sum(1 for _, _, success, _ in results if success)
    failed = total_sims - successful
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Total: {total_sims} | Successful: {successful} | Failed: {failed}")
    print(f"Success rate: {successful/total_sims*100:.1f}%")
    print(f"Time elapsed: {elapsed_time:.2f}s")
    print(f"Average time per simulation: {elapsed_time/total_sims:.2f}s")
    print(f"{'='*60}")
    
    # Show failed simulations if any
    if failed > 0:
        print("\nFailed simulations:")
        for env_id, intent, success, msg in results:
            if not success:
                print(f"  - Env {env_id}, Intent {intent}: {msg}")

if __name__ == "__main__":
    main()