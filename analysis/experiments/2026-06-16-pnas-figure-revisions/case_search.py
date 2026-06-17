"""Search clean human team-rounds for the R3_team_case story:

  the team starts mis-coordinated (a suboptimal joint combo), at least
  one player STAYS in their role for a stage even though best-response
  (max expected team value under the Bayesian posterior over teammates)
  says switch, and the team EVENTUALLY deviates to a better combination
  and improves / WINS. Bonus: a symmetry-breaking story — two
  identical-stat players clash then one breaks away.

Prints a ranked shortlist with one-line rationales.

Run from analysis/:
    uv run python experiments/2026-06-16-pnas-figure-revisions/case_search.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from common_human import (  # noqa: E402
    best_response, human_posteriors, identical_pairs, load_human_records,
    load_stage1_canonical, stage_value_rank,
)
from shared.constants import ROLE_SHORT  # noqa: E402


def combo_str(role_by_pos):
    return "".join(ROLE_SHORT[r] for r in role_by_pos)


def analyze(rec, s1, strat):
    """Return a dict of story-relevant facts for one team-round, or None."""
    n = rec["n_stages"]
    if n < 2:
        return None
    posteriors = human_posteriors(rec, s1, strat)
    ranks = [stage_value_rank(rec, s) for s in range(n)]

    # best-response per (stage>=1, player); "stayed against BR" events
    stay_vs_br = []   # (stage, pid, stayed_role, br_role)
    for s in range(1, n):
        for pid in range(3):
            stayed = rec["role_seq"][s][pid] == rec["role_seq"][s - 1][pid]
            if not stayed:
                continue
            br, _ev = best_response(rec, posteriors, s, pid)
            if br != rec["role_seq"][s][pid]:
                stay_vs_br.append((s, pid, rec["role_seq"][s][pid], br))

    # symmetry: identical-stat pair that clashes early then splits
    sym_break = []
    for (a, b) in identical_pairs(rec):
        traj = [(rec["role_seq"][s][a], rec["role_seq"][s][b]) for s in range(n)]
        clash0 = traj[0][0] == traj[0][1]
        first_split = next((s for s, (ra, rb) in enumerate(traj) if ra != rb),
                           None)
        if clash0 and first_split is not None:
            sym_break.append((a, b, first_split))

    # "relent": a player stayed-against-BR at stage s, then SWITCHED at s+1
    stayed_at = {(s, pid) for (s, pid, _g, _b) in stay_vs_br}
    relents = []
    for (s, pid) in stayed_at:
        if s + 1 < n and rec["role_seq"][s + 1][pid] != rec["role_seq"][s][pid]:
            relents.append((s, pid))

    final = rec["role_seq"][-1]
    heterogeneous = len(set(final)) == 3       # fully specialized end-state

    improved = ranks[-1] < ranks[0]
    return {
        "ranks": ranks,
        "combos": [combo_str(rec["role_seq"][s]) for s in range(n)],
        "stay_vs_br": stay_vs_br,
        "relents": relents,
        "heterogeneous": heterogeneous,
        "sym_break": sym_break,
        "improved": improved,
        "rank_drop": ranks[0] - ranks[-1],
        "final_rank": ranks[-1],
        "win": rec["outcome"] == "WIN",
        "n_switches": sum(1 for s in range(1, n) for pid in range(3)
                          if rec["role_seq"][s][pid] != rec["role_seq"][s - 1][pid]),
    }


def score(rec, a):
    """Heuristic desirability of a round as the case study."""
    if not a["win"]:
        return -1e9
    if not a["stay_vs_br"]:
        return -1e9
    s = 0.0
    s += 4.0 * a["rank_drop"]                 # improvement magnitude
    s += 8.0 if a["final_rank"] <= 3 else 0.0  # ends near-optimal
    s += 10.0 if a["heterogeneous"] else 0.0   # ends fully specialized
    s += 2.0 * len(a["stay_vs_br"])           # stubbornness episodes
    s += 12.0 * len(a["relents"])             # stubborn-THEN-relent (key beat)
    s += 8.0 if a["sym_break"] else 0.0       # symmetry-breaking bonus
    s -= 1.5 * max(0, a["n_switches"] - 4)    # prefer a clean, readable story
    s -= 0.5 * rec["n_stages"]                # prefer compact rounds
    return s


def main():
    s1, strat = load_stage1_canonical()
    records = load_human_records()

    scored = []
    for rec in records:
        a = analyze(rec, s1, strat)
        if a is None:
            continue
        sc = score(rec, a)
        if sc > -1e8:
            scored.append((sc, rec, a))
    scored.sort(key=lambda t: -t[0])

    print(f"\n{len(scored)} candidate rounds (WIN + >=1 stay-against-BR).\n")
    print("Shortlist (top 8):\n")
    for rank, (sc, rec, a) in enumerate(scored[:8], 1):
        traj = " -> ".join(f"{c}[{r}]" for c, r in zip(a["combos"], a["ranks"]))
        sym = ""
        if a["sym_break"]:
            ab = a["sym_break"][0]
            sym = (f"  SYMMETRY: P{ab[0]+1}/P{ab[1]+1} identical, "
                   f"clash@1 split@stage{ab[2]+1}")
        stays = ", ".join(f"P{pid+1}@s{s+1} stayed {ROLE_SHORT[got]} "
                          f"(BR {ROLE_SHORT[br]})"
                          for s, pid, got, br in a["stay_vs_br"])
        relent = ", ".join(f"P{pid+1} relents after s{s+1}"
                           for s, pid in a["relents"])
        het = "HETEROGENEOUS-final" if a["heterogeneous"] else "homogeneous-ish"
        print(f"#{rank}  score={sc:.1f}  game={rec['game_id']} "
              f"r{rec['round_number']}  profile={rec['stat_profile_id']} "
              f"({rec['symmetry']})  {rec['n_stages']} stages  {rec['outcome']}"
              f"  {het}")
        print(f"      combos[rank]: {traj}")
        print(f"      rank {a['ranks'][0]} -> {a['ranks'][-1]} "
              f"(drop {a['rank_drop']}), {a['n_switches']} switches")
        print(f"      stays-vs-BR: {stays}")
        if relent:
            print(f"      RELENT: {relent}")
        if sym:
            print(f"     {sym}")
        print(f"      export={rec['export_name']}")
        print()


if __name__ == "__main__":
    main()
