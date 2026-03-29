"""MDP environment config loading (values.npy + config.py) without JAX dependency."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import ENVS_DIR, HUMAN_ENVS_DIR
from .constants import ROLE_COMBO_TO_ENV_NUM


def load_env_config(env_dir: str | Path) -> dict:
    """Load a single env directory (config.py + values.npy).

    Parses config.py via regex to avoid requiring JAX (the files use jnp.array).

    Returns:
        Dict with keys: values, player_stats, boss_damage, team_max_hp, enemy_max_hp.
    """
    env_dir = Path(env_dir)
    config_path = env_dir / "config.py"
    values_path = env_dir / "values.npy"

    text = config_path.read_text()

    team_max_hp = int(re.search(r"TEAM_MAX_HP\s*=\s*(\d+)", text).group(1))
    enemy_max_hp = int(re.search(r"ENEMY_MAX_HP\s*=\s*(\d+)", text).group(1))
    boss_damage = float(re.search(r"BOSS_DAMAGE\s*=\s*([\d.]+)", text).group(1))

    # Match the outer 2D array: PLAYER_STATS = jnp.array([[...], [...], [...]])
    ps_match = re.search(
        r"PLAYER_STATS\s*=\s*(?:jnp\.array|np\.array)?\(?\s*(\[\[.+?\]\])\s*\)?",
        text,
        re.DOTALL,
    )
    if ps_match:
        outer = ps_match.group(1)
        rows = re.findall(r"\[([^\[\]]+)\]", outer)
        player_stats = np.array([[float(x) for x in row.split(",")] for row in rows])
    else:
        raise ValueError(f"Could not parse PLAYER_STATS from {config_path}")

    values = np.load(values_path)

    return {
        "values": values,
        "player_stats": player_stats,
        "boss_damage": boss_damage,
        "team_max_hp": team_max_hp,
        "enemy_max_hp": enemy_max_hp,
    }


def get_env_dir(
    role_combo: str | None = None,
    env_id: int | None = None,
    human_envs_dir: str | Path | None = None,
    envs_dir: str | Path | None = None,
) -> Path:
    """Resolve the env directory, preferring human_envs_value_matrices/.

    Args:
        role_combo: Role combo string like "FTM".
        env_id: Numeric env ID (used as fallback in envs/ directory).
        human_envs_dir: Override path for human_envs_value_matrices/.
        envs_dir: Override path for envs/.

    Returns:
        Path to the env directory.
    """
    human_dir = Path(human_envs_dir) if human_envs_dir else HUMAN_ENVS_DIR
    env_dir = Path(envs_dir) if envs_dir else ENVS_DIR

    # Prefer human_envs_value_matrices if role_combo is given
    if role_combo:
        candidate = human_dir / role_combo
        if (candidate / "values.npy").exists():
            return candidate

    # Fall back to envs/{env_id}
    if env_id is not None:
        candidate = env_dir / str(env_id)
        if (candidate / "values.npy").exists():
            return candidate

    # Try to resolve env_id from role_combo
    if role_combo and env_id is None:
        env_id = ROLE_COMBO_TO_ENV_NUM.get(role_combo)
        if env_id is not None:
            candidate = env_dir / str(env_id)
            if (candidate / "values.npy").exists():
                return candidate

    raise FileNotFoundError(
        f"No env found for role_combo={role_combo}, env_id={env_id}"
    )


def make_env_loader(
    human_envs_dir: str | Path | None = None,
    envs_dir: str | Path | None = None,
) -> callable:
    """Create a cached env loader function.

    Returns a callable: loader(role_combo, env_id=None) -> dict
    Results are cached in memory.
    """
    cache: dict[tuple, dict] = {}

    def loader(role_combo: str | None = None, env_id: int | None = None) -> dict:
        key = (role_combo, env_id)
        if key not in cache:
            d = get_env_dir(role_combo, env_id, human_envs_dir, envs_dir)
            cache[key] = load_env_config(d)
        return cache[key]

    return loader
