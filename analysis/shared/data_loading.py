"""Format-aware data loading from player.csv gameSummary.

Supports v2 and v3 Empirica exports. v1 exports (no gameSummary column)
are not supported — use the original notebooks for those.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from . import EXPORTS_DIR
from .constants import GAME_ROLE_TO_IDX, DROPOUT_GAME_IDS
from .parsing import parse_inferred_roles, parse_deviate_roles


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StageRecord:
    """One stage within a round, from one player's perspective."""
    stage: int
    role: str                          # "FIGHTER" / "TANK" / "MEDIC"
    role_idx: int                      # 0, 1, 2
    inferred_roles: dict[int, int]     # {position: role_idx}, parsed
    is_bot: bool                       # auto-submitted by dropout replacement
    turns: list[dict]                  # [{turn, action, teamHealth, enemyHealth}]


@dataclass
class RoundRecord:
    """One round from one player's perspective."""
    round_number: int
    round_type: str                    # "human" | "bot"
    outcome: str                       # "WIN" | "LOSE" | "TIMEOUT"
    config: dict                       # raw config dict
    player_stats: dict                 # {"STR": int, "DEF": int, "SUP": int}
    stages: list[StageRecord]
    points_earned: int
    turns_taken: int
    # Derived convenience fields
    optimal_roles: list[int]
    deviate_roles: list[int] | None
    stat_profile_id: str
    optimal_deviate_roles_id: str | None
    enemy_intent_sequence: str


@dataclass
class PlayerRound:
    """Primary unit: one player in one round."""
    export_name: str
    game_id: str
    player_id: int                     # gamePlayerId (0-2)
    participant_id: str | None
    round: RoundRecord
    is_dropout: bool


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_format(data_dir: Path) -> Literal["v1", "v1.5", "v2", "v3"]:
    """Auto-detect export format version from player.csv header.

    - v1: No gameSummary column (Jan 25 export)
    - v1.5: Has gameSummary but summary-only (roundResults, no stages/turns).
            Jan 28 export — gameSummary exists but uses old schema.
    - v2: Has gameSummary with full stage/turn data, no dropout fields (Feb 13)
    - v3: Has gameSummary + dropout tracking fields (Mar 6, Mar 18 exports)
    """
    player_csv = Path(data_dir) / "player.csv"
    with open(player_csv) as f:
        header = set(f.readline().strip().split(","))

    if "gameSummary" not in header:
        return "v1"

    # Check if gameSummary uses old schema (roundResults vs rounds)
    df = pd.read_csv(player_csv, nrows=5, usecols=["gameSummary"])
    for val in df["gameSummary"].dropna():
        gs = json.loads(val) if isinstance(val, str) else val
        if "roundResults" in gs and "rounds" not in gs:
            return "v1.5"
        break

    if "isDropout" in header:
        return "v3"
    return "v2"


# ---------------------------------------------------------------------------
# Internal: build records from gameSummary
# ---------------------------------------------------------------------------

def _parse_game_summary(
    gs_json: str | dict,
    game_id: str,
    player_id: int,
    participant_id: str | None,
    export_name: str,
    is_dropout_player: bool,
    include_bot_rounds: bool,
) -> list[PlayerRound]:
    """Parse a single player's gameSummary into PlayerRound records."""
    gs = json.loads(gs_json) if isinstance(gs_json, str) else gs_json
    records = []

    for rnd in gs.get("rounds", []):
        round_type = rnd.get("roundType", "human")
        if not include_bot_rounds and round_type == "bot":
            continue

        config = rnd.get("config", {})

        # Build StageRecords
        stages = []
        for stg in rnd.get("stages", []):
            role_str = stg.get("role", "")
            stages.append(StageRecord(
                stage=stg.get("stage", 0),
                role=role_str,
                role_idx=GAME_ROLE_TO_IDX.get(role_str, -1),
                inferred_roles=parse_inferred_roles(stg.get("inferredRoles")),
                is_bot=stg.get("isBot", False),
                turns=stg.get("turns", []),
            ))

        # Derived fields
        optimal_roles = config.get("optimalRoles", [])
        deviate_roles_id = config.get("optimalDeviateRolesId")
        deviate_roles = parse_deviate_roles(deviate_roles_id)

        round_record = RoundRecord(
            round_number=rnd.get("roundNumber", 0),
            round_type=round_type,
            outcome=rnd.get("outcome", ""),
            config=config,
            player_stats=rnd.get("playerStats", {}),
            stages=stages,
            points_earned=rnd.get("pointsEarned", 0),
            turns_taken=rnd.get("turnsTaken", 0),
            optimal_roles=optimal_roles,
            deviate_roles=deviate_roles,
            stat_profile_id=config.get("statProfileId", ""),
            optimal_deviate_roles_id=deviate_roles_id,
            enemy_intent_sequence=config.get("enemyIntentSequence", ""),
        )

        records.append(PlayerRound(
            export_name=export_name,
            game_id=game_id,
            player_id=player_id,
            participant_id=participant_id,
            round=round_record,
            is_dropout=is_dropout_player,
        ))

    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_export(
    data_dir: str | Path,
    name: str | None = None,
    include_bot_rounds: bool = True,
    include_dropout_games: bool = True,
) -> list[PlayerRound]:
    """Load a single Empirica export directory into PlayerRound records.

    Args:
        data_dir: Path to export directory containing player.csv.
        name: Label for this export (defaults to directory name).
        include_bot_rounds: If False, filter out bot rounds.
        include_dropout_games: If False, filter out known dropout games.

    Returns:
        List of PlayerRound records (one per player per round).

    Raises:
        ValueError: If the export is v1 format (no gameSummary).
    """
    data_dir = Path(data_dir)
    export_name = name or data_dir.name
    version = detect_format(data_dir)

    if version in ("v1", "v1.5"):
        raise ValueError(
            f"Export '{export_name}' is {version} format "
            f"({'no gameSummary column' if version == 'v1' else 'summary-only gameSummary, no stage/turn data'}). "
            "These early exports are not supported. Use the original notebooks."
        )

    df = pd.read_csv(data_dir / "player.csv")
    records: list[PlayerRound] = []

    for _, row in df.iterrows():
        game_id = str(row.get("gameID", ""))

        if not include_dropout_games and game_id in DROPOUT_GAME_IDS:
            continue

        gs_raw = row.get("gameSummary")
        if pd.isna(gs_raw):
            continue

        player_id = int(row.get("gamePlayerId", -1))
        participant_id = str(row["participantID"]) if pd.notna(row.get("participantID")) else None

        # Dropout detection: v3 has explicit column, v2 defaults to False
        if version == "v3" and pd.notna(row.get("isDropout")):
            is_dropout = bool(row["isDropout"])
        else:
            is_dropout = False

        records.extend(_parse_game_summary(
            gs_json=gs_raw,
            game_id=game_id,
            player_id=player_id,
            participant_id=participant_id,
            export_name=export_name,
            is_dropout_player=is_dropout,
            include_bot_rounds=include_bot_rounds,
        ))

    return records


def load_all_exports(
    data_root: str | Path | None = None,
    data_dirs: list[str | Path] | None = None,
    **kwargs,
) -> list[PlayerRound]:
    """Load multiple exports, auto-discovering directories if data_root given.

    Args:
        data_root: Directory containing export subdirectories
                   (defaults to data/exports/).
        data_dirs: Explicit list of export directories to load.
        **kwargs: Passed to load_export().

    Returns:
        Combined list of PlayerRound records from all exports.
    """
    if data_dirs is not None:
        dirs = [Path(d) for d in data_dirs]
    else:
        root = Path(data_root) if data_root else EXPORTS_DIR
        dirs = sorted(root.iterdir())
        dirs = [d for d in dirs if d.is_dir() and (d / "player.csv").exists()]

    all_records: list[PlayerRound] = []
    for d in dirs:
        version = detect_format(d)
        if version in ("v1", "v1.5"):
            print(f"Skipping {d.name} ({version} format, not supported)")
            continue
        records = load_export(d, **kwargs)
        print(f"Loaded {d.name}: {len(records)} player-rounds")
        all_records.extend(records)

    print(f"Total: {len(all_records)} player-rounds from {len(dirs)} exports")
    return all_records


def to_dataframe(records: list[PlayerRound]) -> pd.DataFrame:
    """Flatten PlayerRound records into a pandas DataFrame.

    Each row = one player in one round. Stage/turn data is kept as lists.
    Useful for quick filtering and aggregation.
    """
    rows = []
    for pr in records:
        rnd = pr.round
        rows.append({
            "export_name": pr.export_name,
            "game_id": pr.game_id,
            "player_id": pr.player_id,
            "participant_id": pr.participant_id,
            "is_dropout": pr.is_dropout,
            "round_number": rnd.round_number,
            "round_type": rnd.round_type,
            "outcome": rnd.outcome,
            "points_earned": rnd.points_earned,
            "turns_taken": rnd.turns_taken,
            "stat_profile_id": rnd.stat_profile_id,
            "optimal_roles": rnd.optimal_roles,
            "deviate_roles": rnd.deviate_roles,
            "optimal_deviate_roles_id": rnd.optimal_deviate_roles_id,
            "enemy_intent_sequence": rnd.enemy_intent_sequence,
            "player_stats": rnd.player_stats,
            "n_stages": len(rnd.stages),
            "roles": [s.role for s in rnd.stages],
            "role_idxs": [s.role_idx for s in rnd.stages],
            "inferences": [s.inferred_roles for s in rnd.stages],
            "config": rnd.config,
        })
    return pd.DataFrame(rows)
