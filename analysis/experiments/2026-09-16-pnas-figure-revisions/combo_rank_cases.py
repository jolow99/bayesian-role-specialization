import sys, importlib.util
from pathlib import Path
import numpy as np
EXP = Path("experiments")
sys.path.insert(0, str(EXP / "2026-06-17-pnas-figure-revisions"))
sys.path.insert(0, str(EXP / "2026-06-16-pnas-figure-revisions"))
from common_human import load_human_records, stage_start_hp, value_vector_27, _combo_index, stage_value_rank
spec = importlib.util.spec_from_file_location("tc0617", EXP / "2026-06-17-pnas-figure-revisions/team_case.py")
tc = importlib.util.module_from_spec(spec); sys.modules["tc0617"] = tc; spec.loader.exec_module(tc)
from shared.constants import ROLE_SHORT
from shared.parsing import canonical_combo

CASES = [("2F8H1E", 2), ("11B0J4", 8), ("K72DV0", 5), ("RBRN0Z", 8), ("11B0J4", 6)]
records = load_human_records()
def combo_str(idx):
    return "".join(ROLE_SHORT[r] for r in (idx // 9, (idx // 3) % 3, idx % 3))

for suffix, rnd in CASES:
    rec = [r for r in records if r["game_id"].endswith(suffix) and r["round_number"] == rnd][0]
    rec = tc.trim_to_live(rec)
    vals = rec["env_config"]["values"]
    print(f"\n=== {rec['game_id'][-6:]} r{rnd}  profile {rec['stat_profile_id']}  values shape {vals.shape} "
          f"range [{vals.min():.3f},{vals.max():.3f}]  eis={rec['eis']}")
    hps = stage_start_hp(rec)
    for s in range(rec["n_stages"]):
        thp, ehp = hps[s]
        V = value_vector_27(rec, thp, ehp)
        chosen = _combo_index(rec["role_seq"][s]); cs = combo_str(chosen)
        raw = stage_value_rank(rec, s)
        Vr = np.round(V, 6)
        dense = int((Vr > Vr[chosen]).sum()) + 1                     # ties share rank
        n_tied_top = int((Vr == Vr.max()).sum())
        n_tied_chosen = int((Vr == Vr[chosen]).sum())
        # canonical: collapse position-symmetric combos
        canon = {}
        for i in range(27):
            canon.setdefault(canonical_combo(combo_str(i), rec["stat_profile_id"]), []).append(V[i])
        cvals = {k: float(np.mean(v)) for k, v in canon.items()}
        ccs = canonical_combo(cs, rec["stat_profile_id"])
        crank = int(sum(1 for v in cvals.values() if round(v, 6) > round(cvals[ccs], 6))) + 1
        ncanon = len(cvals); n_distinct = len(set(Vr))
        top = sorted(range(27), key=lambda i: -V[i])[:4]
        print(f"  s{s+1} hp=({thp:.0f},{ehp:.0f}) chosen {cs} V={V[chosen]:.3f} | best V={V.max():.3f} "
              f"gap={V.max()-V[chosen]:.3f} | raw rank {raw}/27, dense {dense}/27 (chosen tied x{n_tied_chosen}, top tied x{n_tied_top}), "
              f"canonical {crank}/{ncanon}, distinct values {n_distinct} | top: "
              + ", ".join(f"{combo_str(i)}={V[i]:.3f}" for i in top))
