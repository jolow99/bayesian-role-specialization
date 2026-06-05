"""Task 5 — search the clean human rounds for symmetry-breaking examples.

For every clean human team-round containing a pair of players with
identical stat profiles, classify the pair's joint trajectory:

(a) symmetry-breaking SUCCESS — the pair starts clashing (same role),
    then splits into distinct roles and stays split through the end of a
    WIN round. Scored by clash length, post-split stability (no role
    changes after the split), round length, and win.

(b) symmetry-breaking FAILURE then recovery — the pair clashes and/or
    simultaneously switches roles repeatedly (mirroring) before ending
    distinct. Scored by the number of failure events (clash stages +
    simultaneous-switch boundaries), requiring distinct roles at the end.

Prints the top 3 of each with a one-line trajectory summary, and writes
example_candidates.md.
"""

from __future__ import annotations

from common import SCRIPT_DIR, load_clean_human_teams
from shared.constants import ROLE_SHORT

OUT_PATH = SCRIPT_DIR / "example_candidates.md"

# Rounds currently used by figures 3 / 4, for comparison.
CURRENT = {
    "fig3 (best_respond)": ("01KQ6YDA7NHWHSTQNZB82F8H1E", 2),
    "fig4 (flip_flop)": ("01KRBT2X2GA4BYS41KYT2R4QJ6", 5),
}


def identical_pairs(stat_profile_id):
    parts = stat_profile_id.split("_")
    return [(a, b) for a in range(3) for b in range(a + 1, 3)
            if parts[a] == parts[b]]


def pair_trajectory(team_prs, a, b):
    roles = {pr.player_id: [s.role_idx for s in pr.round.stages]
             for pr in team_prs}
    n = min(len(roles[a]), len(roles[b]))
    return [(roles[a][s], roles[b][s]) for s in range(n)]


def fmt_traj(traj, a, b):
    return (f"P{a+1},P{b+1}: "
            + " | ".join(f"{ROLE_SHORT[x]}{ROLE_SHORT[y]}" for x, y in traj))


def analyze(team_prs, a, b):
    traj = pair_trajectory(team_prs, a, b)
    if len(traj) < 3:
        return None
    clash = [x == y for x, y in traj]
    simul_switch = [
        s for s in range(1, len(traj))
        if traj[s][0] != traj[s - 1][0] and traj[s][1] != traj[s - 1][1]
    ]
    outcome = team_prs[0].round.outcome

    # success: clash from stage 1, single split point, stable & distinct
    # after, and the split tail must last >= 2 stages ("stay split").
    success_score = None
    if clash[0]:
        split = next((s for s, c in enumerate(clash) if not c), None)
        if (split is not None and not any(clash[split:])
                and len(traj) - split >= 2):
            stable_after = all(traj[s] == traj[split]
                               for s in range(split, len(traj)))
            success_score = (
                (outcome == "WIN") * 100
                + stable_after * 40           # they hold the split roles
                + (len(traj) - split) * 12    # longer stable tail = clearer
                + min(split, 3) * 8           # some initial clash = clearer
                + len(traj))

    # failure→recovery: >=1 failure event after stage 1, ends distinct
    n_fail = sum(clash[1:]) + len(simul_switch)
    failure_score = None
    if n_fail >= 1 and not clash[-1] and (sum(clash) + len(simul_switch)) >= 2:
        end_stable = len(traj) >= 2 and traj[-1] == traj[-2]
        failure_score = (
            n_fail * 20
            + (outcome == "WIN") * 15
            + end_stable * 10
            + len(traj))

    return {
        "traj": traj, "clash": clash, "simul_switch": simul_switch,
        "outcome": outcome, "success_score": success_score,
        "failure_score": failure_score,
    }


def main():
    teams = load_clean_human_teams()
    successes, failures = [], []

    for (export, gid, rn), team_prs in teams.items():
        rnd = team_prs[0].round
        for a, b in identical_pairs(rnd.stat_profile_id):
            res = analyze(team_prs, a, b)
            if res is None:
                continue
            entry = {
                "game_id": gid, "round": rn,
                "stats": rnd.stat_profile_id, "pair": (a, b),
                "outcome": res["outcome"],
                "traj_str": fmt_traj(res["traj"], a, b),
                "n_fail": sum(res["clash"][1:]) + len(res["simul_switch"]),
                **res,
            }
            if res["success_score"] is not None:
                successes.append(entry)
            if res["failure_score"] is not None:
                failures.append(entry)

    successes.sort(key=lambda e: -e["success_score"])
    failures.sort(key=lambda e: -e["failure_score"])

    lines = ["# Task 5 — symmetry-breaking example candidates", ""]

    def block(title, entries, score_key):
        out = [f"## {title}", ""]
        for i, e in enumerate(entries[:3], 1):
            cur = next((k for k, v in CURRENT.items()
                        if v == (e["game_id"], e["round"])), None)
            tag = f"  ← current {cur}" if cur else ""
            out.append(
                f"{i}. `{e['game_id']}` r{e['round']} · stats "
                f"{e['stats']} · {e['outcome']} · score "
                f"{e[score_key]}{tag}")
            out.append(f"   - {e['traj_str']}")
        out.append("")
        return out

    lines += block("(a) Symmetry-breaking SUCCESS "
                   "(clash → split → stable until win)",
                   successes, "success_score")
    lines += block("(b) Symmetry-breaking FAILURE then recovery "
                   "(repeated mirroring before converging)",
                   failures, "failure_score")

    # Where do the currently-used rounds sit?
    lines += ["## Current figure rounds", ""]
    for name, (gid, rn) in CURRENT.items():
        s_rank = next((i for i, e in enumerate(successes, 1)
                       if (e["game_id"], e["round"]) == (gid, rn)), None)
        f_rank = next((i for i, e in enumerate(failures, 1)
                       if (e["game_id"], e["round"]) == (gid, rn)), None)
        lines.append(f"- {name}: `{gid}` r{rn} — success-rank: "
                     f"{s_rank or '—'} / {len(successes)}, failure-rank: "
                     f"{f_rank or '—'} / {len(failures)}")

    text = "\n".join(lines)
    OUT_PATH.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
