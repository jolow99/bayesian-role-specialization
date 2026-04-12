"""Game-level leave-one-out fold helpers for Stage 2 cross-validation."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def stratified_game_folds(records) -> list[dict]:
    """Build stratified leave-one-out folds keyed on game_id.

    Ensures every held-out fold contains at least one record of each round
    type (human, bot) where feasible. Games that are entirely one round type
    are merged into a multi-game "singletons" fold so per-fold disaggregation
    stays well-defined.

    Returns a list of {"heldout": [record, ...], "train": [record, ...]}.
    """
    by_game: dict[str, list] = defaultdict(list)
    for r in records:
        by_game[r["game_id"]].append(r)

    full_games = []  # games with both round types
    human_only = []
    bot_only = []

    for gid, recs in by_game.items():
        rts = {r["round_type"] for r in recs}
        if "human" in rts and "bot" in rts:
            full_games.append(gid)
        elif "human" in rts:
            human_only.append(gid)
        else:
            bot_only.append(gid)

    folds = []

    # Each "full" game is its own fold (natural stratification).
    for gid in full_games:
        heldout = by_game[gid]
        train = [r for r in records if r["game_id"] != gid]
        folds.append({"heldout_game_ids": [gid], "heldout": heldout, "train": train})

    # Pair human-only and bot-only games so each fold has both types.
    pairs = []
    while human_only and bot_only:
        pairs.append([human_only.pop(), bot_only.pop()])
    # Any remaining singletons get merged into the last pair (if any), or
    # become their own unstratified fold.
    tail = human_only + bot_only
    if pairs and tail:
        pairs[-1].extend(tail)
        tail = []
    if tail:
        pairs.append(tail)

    for gids in pairs:
        heldout_set = set(gids)
        heldout = [r for r in records if r["game_id"] in heldout_set]
        train = [r for r in records if r["game_id"] not in heldout_set]
        folds.append({"heldout_game_ids": list(gids), "heldout": heldout, "train": train})

    return folds
