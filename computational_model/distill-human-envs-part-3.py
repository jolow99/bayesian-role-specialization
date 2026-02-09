import os
import numpy as np
import importlib.util
from scipy.stats import qmc


N = 10
HUMAN_CASE_ENVS_DIR = "human_case_envs"
OUTPUT_FILE = "lds_arrays_human.txt"


def generate_binary_qmc(prob, seed):
    """Generate length-N binary sequence: [1] + Sobol-guided tail."""
    engine = qmc.Sobol(d=1, scramble=True, seed=seed)
    samples = engine.random(N - 1).ravel()

    threshold = (prob * N - 1) / (N - 1)
    threshold = np.clip(threshold, 0.0, 1.0)

    tail = (samples < threshold).astype(int)
    return np.concatenate(([1], tail))


def load_attr(file_path, attr):
    """Load a single attribute from a Python file."""
    try:
        spec = importlib.util.spec_from_file_location("module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, attr, None)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def generate_lds_list():
    lds_list = []

    for stats_combo in os.listdir(HUMAN_CASE_ENVS_DIR):
        stats_path = os.path.join(HUMAN_CASE_ENVS_DIR, stats_combo)
        if not os.path.isdir(stats_path):
            continue

        for role_combo in os.listdir(stats_path):
            role_path = os.path.join(stats_path, role_combo)
            if not os.path.isdir(role_path):
                continue

            for fname in os.listdir(role_path):
                if not fname.endswith(".py"):
                    continue

                file_path = os.path.join(role_path, fname)
                prob = load_attr(file_path, "ENEMY_ATTACK_PROB")
                if prob is None:
                    continue

                seed = abs(hash(fname)) % (2**32)
                lds = generate_binary_qmc(prob, seed)

                lds_list.append({
                    "key": f"{stats_combo}_{role_combo}",
                    "prob": prob,
                    "lds_array": lds,
                    "file": file_path,
                })

    return lds_list


if __name__ == "__main__":
    lds_list = generate_lds_list()

    with open(OUTPUT_FILE, "w") as f:
        for entry in lds_list:
            lds_str = "".join(map(str, entry["lds_array"]))
            f.write(f"{entry['key']},{entry['prob']},{lds_str}\n")

    print(f"Saved {len(lds_list)} entries to {OUTPUT_FILE}")
