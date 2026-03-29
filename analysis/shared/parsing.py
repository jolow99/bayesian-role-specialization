"""Parsing utilities for inference strings, config IDs, and role combos."""

from __future__ import annotations

import re

from .constants import ROLE_MAP, ROLE_CHAR_TO_IDX, SYMMETRIC_PROFILES, ALL_ROLE_COMBOS


def parse_inferred_roles(inf_str: str | None) -> dict[int, int]:
    """Parse inference string like 'P2: F, P3: T' -> {1: 0, 2: 1}.

    Returns a dict mapping 0-indexed player position to role index.
    P1 -> 0, P2 -> 1, P3 -> 2.
    """
    if not inf_str:
        return {}
    result = {}
    for match in re.finditer(r"P(\d+):\s*([FTM])", inf_str):
        p_num = int(match.group(1)) - 1  # convert 1-based to 0-based
        role_char = match.group(2)
        result[p_num] = ROLE_CHAR_TO_IDX[role_char]
    return result


def parse_deviate_roles(deviate_id: str | None) -> list[int] | None:
    """Parse optimalDeviateRolesId like 'FMT_TMM' -> [1, 2, 2] (deviate-optimal roles).

    The second group (after '_') gives the deviate-optimal role for each player.
    Index 0 = human, indices 1-2 = bots.
    """
    if not deviate_id or "_" not in deviate_id:
        return None
    _, dev_part = deviate_id.split("_", 1)
    return [ROLE_MAP[c] for c in dev_part]


def parse_stat_optimal_roles(deviate_id: str | None) -> list[int] | None:
    """Parse optimalDeviateRolesId like 'FMT_TMM' -> [0, 2, 1] (stat-optimal roles).

    The first group (before '_') gives the stat-optimal role for each player.
    Index 0 = human, indices 1-2 = bots.
    """
    if not deviate_id or "_" not in deviate_id:
        return None
    stat_part, _ = deviate_id.split("_", 1)
    return [ROLE_MAP[c] for c in stat_part]


def canonical_combo(combo: str, stat_profile: str) -> str:
    """Canonicalize a role combo string based on stat profile symmetry.

    For symmetric profiles (e.g., 222_222_222), players are interchangeable,
    so "FTM" and "TFM" are the same combo. This returns a canonical form
    by sorting the interchangeable positions.
    """
    sym = SYMMETRIC_PROFILES.get(stat_profile)
    if sym == "all":
        return "".join(sorted(combo))
    elif sym == "last_two":
        return combo[0] + "".join(sorted(combo[1:]))
    return combo


def get_canonical_combos(stat_profile: str) -> list[str]:
    """Get the list of unique canonical combos for a stat profile."""
    seen = set()
    canonical = []
    for c in ALL_ROLE_COMBOS:
        cc = canonical_combo(c, stat_profile)
        if cc not in seen:
            seen.add(cc)
            canonical.append(cc)
    return canonical
