import re
import shutil
import numpy as np
from pathlib import Path
from itertools import groupby

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_DIR = SCRIPT_DIR / "output" / "bot_case_envs"
OUTPUT_DIR = SCRIPT_DIR / "output" / "bot_case_envs_final_1"
LDS_OUTPUT = SCRIPT_DIR / "output" / "bot_lds_seqs_1.txt"
TOP_N = 5

# --- Step 1: Find best env per combo ---

results = []

for stat_dir in sorted(INPUT_DIR.iterdir()):
    if not stat_dir.is_dir(): continue

    for combo_dir in sorted(stat_dir.iterdir()):
        if not combo_dir.is_dir() or "_" not in combo_dir.name: continue

        candidates = []
        for log_file in combo_dir.glob("*_sim_log.txt"):
            text = log_file.read_text()
            match = re.search(r'FORCED \(WIN\).*?>>> RESULT: WIN', text, re.DOTALL)
            if not match: continue
            turns = [int(t) for t in re.findall(r'\[Turn (\d+)\]', match.group())]
            if turns and turns[-1] != 5:
                candidates.append((turns[-1], log_file.stem.replace("_sim_log", "")))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            best_turns, best_env = candidates[0]
            results.append((stat_dir.name, combo_dir.name, best_env, best_turns))

# --- Step 2: Filter to top N per stat block ---

results.sort(key=lambda x: (x[0], x[3]))

selected = []
seen_envs = set()
for stat_key, group in groupby(results, key=lambda x: x[0]):
    items = list(group)
    filtered = [r for r in items if r[3] != 5]
    if not filtered:
        filtered = items
    top_n = 6 if stat_key == "411_222_222" else TOP_N
    # Deduplicate: each env_id can only be selected once globally
    count = 0
    for r in filtered:
        if count >= top_n:
            break
        if r[2] not in seen_envs:
            seen_envs.add(r[2])
            selected.append(r)
            count += 1

# --- Step 3: Copy configs & extract LDS sequences ---

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)
lds_lines = []

for stat, combo, env_id, turns in selected:
    src_dir = INPUT_DIR / stat / combo
    dst_dir = OUTPUT_DIR / stat / combo
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy config .py and extract ENEMY_ATTACK_PROB
    src_py = src_dir / f"{env_id}.py"
    attack_prob = ""
    if src_py.exists():
        shutil.copy2(src_py, dst_dir / src_py.name)
        py_text = src_py.read_text()
        prob_match = re.search(r'ENEMY_ATTACK_PROB\s*=\s*([0-9.]+)', py_text)
        if prob_match:
            attack_prob = prob_match.group(1)

    # Copy sim log
    src_log = src_dir / f"{env_id}_sim_log.txt"
    if src_log.exists():
        shutil.copy2(src_log, dst_dir / src_log.name)

    # Read LDS and format
    lds_file = src_dir / f"{env_id}_lds.npy"
    if lds_file.exists():
        lds = np.load(lds_file)
        binary_str = ''.join(str(int(x)) for x in lds)
        lds_lines.append(f"{stat}_{combo},{attack_prob},{binary_str}")

# --- Step 4: Write LDS sequences file ---

LDS_OUTPUT.write_text('\n'.join(lds_lines) + '\n')

# --- Report ---

print(f"\nCopied {len(selected)} environments to {OUTPUT_DIR}/")
print(f"LDS sequences written to {LDS_OUTPUT}\n")

print(f"{'STAT BLOCK':<15} | {'COMBO':<20} | {'ENV ID':<10} | {'TURNS':<6}")
print("=" * 65)
for stat, combo, env_id, turns in selected:
    print(f"{stat:<15} | {combo:<20} | {env_id:<10} | {turns:<6}")
print(f"\nTotal: {len(selected)}")

print(f"\n--- {LDS_OUTPUT} ---")
for line in lds_lines:
    print(line)
