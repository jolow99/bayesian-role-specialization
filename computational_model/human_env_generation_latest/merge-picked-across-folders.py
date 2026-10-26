"""
Merge picked envs across all output*_per_profile/ and output_<number>/
folders by tie-breaking
on uniquely_optimal_count. Produces a single consolidated pick per
(role, profile) bucket.

Reads: picked_envs_by_role_and_profile.txt from each source output folder
Writes:
  output_merged/picked_envs_by_role_and_profile.txt
  output_merged/updated human envs by profile/PROFILE/COMBO/{config.py, lds.txt}
"""

import json
import re
import shutil
import ast
from collections import defaultdict
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "output_merged"
VALUE_MATRICES_DIR = SCRIPT_DIR.parent / "human_envs_value_matrices"
VALUE_MATRICES_SOURCE = "human_envs_value_matrices"


def folder_created_at(folder):
    stat_result = folder.stat()
    return getattr(stat_result, "st_birthtime", stat_result.st_mtime)


def format_timestamp(timestamp):
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def find_source_folders():
    sources = []
    for folder in SCRIPT_DIR.glob("output*"):
        if not folder.is_dir() or folder.name == "output_merged":
            continue
        if folder.name == "output" or re.fullmatch(r"output_\d+", folder.name) or folder.name.endswith("_per_profile"):
            if (folder / "picked_envs_by_role_and_profile.txt").exists():
                sources.append(folder)
    return sorted(sources, key=lambda folder: (folder_created_at(folder), folder.name))


def load_picks(folder):
    """Returns {profile: {role: info}} from the folder's picked file."""
    f = folder / "picked_envs_by_role_and_profile.txt"
    if not f.exists():
        return {}
    with open(f) as h:
        return json.load(h)


def parse_profile_from_config(config_path):
    text = config_path.read_text()
    match = re.search(r"PLAYER_STATS\s*=\s*jnp\.array\((\[.*?\])\)", text, re.S)
    if match is None:
        raise ValueError(f"Could not parse PLAYER_STATS from {config_path}")
    stats = ast.literal_eval(match.group(1))
    return "_".join("".join(str(int(x)) for x in row) for row in stats)


def load_value_matrix_summary():
    summary_path = VALUE_MATRICES_DIR / "README.md"
    summary = {}
    if not summary_path.exists():
        return summary

    for line in summary_path.read_text().splitlines():
        if not line.startswith("|"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != 5 or not re.fullmatch(r"[FTM]{3}", parts[0]):
            continue

        role, env_id, seed, count_text, lds = parts
        count_match = re.search(r"(\d+)\s*/\s*(\d+)", count_text)
        if count_match:
            count = int(count_match.group(1))
            total = int(count_match.group(2))
        else:
            count = 0
            total = 27

        summary[role] = {
            "env_id": env_id,
            "seed": int(seed),
            "uniquely_optimal_count": count,
            "total_states": total,
            "lds": lds,
        }

    return summary


def load_value_matrix_picks():
    if not VALUE_MATRICES_DIR.exists():
        return []

    summary = load_value_matrix_summary()
    source_created_at = folder_created_at(VALUE_MATRICES_DIR)
    picks = []

    for role_dir in sorted(VALUE_MATRICES_DIR.iterdir()):
        if not role_dir.is_dir() or not re.fullmatch(r"[FTM]{3}", role_dir.name):
            continue

        config_path = role_dir / "config.py"
        lds_path = role_dir / "lds.txt"
        if not config_path.exists() or not lds_path.exists():
            continue

        role = role_dir.name
        profile = parse_profile_from_config(config_path)
        info = dict(summary.get(role, {}))
        info.setdefault("env_id", role)
        info.setdefault("seed", None)
        info.setdefault("uniquely_optimal_count", 0)
        info.setdefault("total_states", 27)
        info["lds"] = lds_path.read_text().strip()

        picks.append({
            "key": (profile, role),
            "priority": 0,
            "count": info.get("uniquely_optimal_count", 0),
            "source_created_at": source_created_at,
            "info": info,
            "source_name": VALUE_MATRICES_SOURCE,
            "config_path": config_path,
            "lds_path": lds_path,
        })

    return picks


def load_output_folder_picks(folder):
    source_created_at = folder_created_at(folder)
    entries = []
    picks = load_picks(folder)

    for profile, roles in picks.items():
        for role, info in roles.items():
            entries.append({
                "key": (profile, role),
                "priority": 1,
                "count": info.get("uniquely_optimal_count", 0),
                "source_created_at": source_created_at,
                "info": info,
                "source_name": folder.name,
                "config_path": (
                    folder
                    / "updated human envs by profile"
                    / profile
                    / role
                    / "config.py"
                ),
                "lds_path": (
                    folder
                    / "updated human envs by profile"
                    / profile
                    / role
                    / "lds.txt"
                ),
            })

    return entries


def should_replace_best(candidate, previous):
    if previous is None:
        return True
    if candidate["priority"] != previous["priority"]:
        return candidate["priority"] < previous["priority"]
    if candidate["count"] != previous["count"]:
        return candidate["count"] > previous["count"]
    if candidate["source_created_at"] != previous["source_created_at"]:
        return candidate["source_created_at"] < previous["source_created_at"]
    return candidate["source_name"] < previous["source_name"]


def main():
    sources = find_source_folders()
    value_matrix_entries = load_value_matrix_picks()
    if value_matrix_entries:
        print(f"Priority source: {VALUE_MATRICES_SOURCE}")
    print(f"Merging from: {[s.name for s in sources]}")
    print("Source creation order:")
    if value_matrix_entries:
        created_at = value_matrix_entries[0]["source_created_at"]
        print(f"  {VALUE_MATRICES_SOURCE}: {format_timestamp(created_at)} (priority)")
    for src in sources:
        print(f"  {src.name}: {format_timestamp(folder_created_at(src))}")

    # (profile, role) -> candidate entry. Lower priority wins first; within a
    # priority, higher count wins, then older source creation time.
    best = {}

    for candidate in value_matrix_entries:
        key = candidate["key"]
        if should_replace_best(candidate, best.get(key)):
            best[key] = candidate

    for src in sources:
        for candidate in load_output_folder_picks(src):
            key = candidate["key"]
            if should_replace_best(candidate, best.get(key)):
                best[key] = candidate

    # Organize: profile -> role -> info
    merged = defaultdict(dict)
    for (profile, role), candidate in best.items():
        info = candidate["info"]
        entry = dict(info)
        entry["source_folder"] = candidate["source_name"]
        entry["source_created_at"] = format_timestamp(candidate["source_created_at"])
        if candidate["priority"] == 0:
            entry["source_priority"] = "value_matrices"
        merged[profile][role] = entry

    # Save merged JSON
    OUT_DIR.mkdir(exist_ok=True)
    out_json = OUT_DIR / "picked_envs_by_role_and_profile.txt"
    with open(out_json, "w") as f:
        json.dump(merged, f, indent=2, sort_keys=True)
    print(f"\nSaved merged picks to {out_json}")

    # Copy configs + LDS into OUT_DIR/updated human envs by profile/PROFILE/COMBO/
    env_dir = OUT_DIR / "updated human envs by profile"
    if env_dir.exists():
        shutil.rmtree(env_dir)
    env_dir.mkdir()

    for profile, roles in sorted(merged.items()):
        profile_out = env_dir / profile
        profile_out.mkdir()
        for role, info in sorted(roles.items()):
            role_out = profile_out / role
            role_out.mkdir()
            candidate = best[(profile, role)]
            shutil.copy2(candidate["config_path"], role_out / "config.py")
            shutil.copy2(candidate["lds_path"], role_out / "lds.txt")

    print(f"Copied configs + LDS to '{env_dir}/'")

    # Print coverage table
    profile_order = [
        "222_222_222",
        "411_141_114",
        "411_222_222",
        "141_222_222",
        "114_222_222",
    ]
    names = {
        "222_222_222": "Balanced",
        "411_141_114": "All-Unique",
        "411_222_222": "High STR",
        "141_222_222": "High DEF",
        "114_222_222": "High HEAL",
    }

    print("\n=== Coverage summary ===")
    total = 0
    for profile in profile_order:
        roles = sorted(merged.get(profile, {}).keys())
        non_fff = [r for r in roles if r != "FFF"]
        # Prefer non-FFF combos; cap at 3
        if len(non_fff) >= 3:
            chosen = non_fff[:3]
        else:
            chosen = (non_fff + (["FFF"] if "FFF" in roles else []))[:3]
        total += len(chosen)
        print(
            f"  {names[profile]:<12} ({profile}): {len(chosen)}/3  "
            f"-> {chosen}  (available: {roles})"
        )
    print(f"\nTotal filled: {total}/15")


if __name__ == "__main__":
    main()
