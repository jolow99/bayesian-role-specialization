import sys, importlib.util
from pathlib import Path
import numpy as np
EXP = Path("experiments")
sys.path.insert(0, str(EXP / "2026-06-17-pnas-figure-revisions"))
sys.path.insert(0, str(EXP / "2026-06-16-pnas-figure-revisions"))
from common_human import load_human_records, stage_start_hp, value_vector_27, _combo_index, stage_value_rank
spec = importlib.util.spec_from_file_location("tc0617", EXP / "2026-06-17-pnas-figure-revisions/team_case.py")
tc = importlib.util.module_from_spec(spec); sys.modules["tc0617"] = tc; spec.loader.exec_module(tc)

records = [tc.trim_to_live(r) for r in load_human_records()]
rows = []   # per stage: raw, dense, gap, n_tied_top, is_last, same_combo_as_prev, best_changed
for rec in records:
    hps = stage_start_hp(rec)
    prev_combo = None; prev_best = None
    for s in range(rec["n_stages"]):
        thp, ehp = hps[s]
        V = value_vector_27(rec, thp, ehp); Vr = np.round(V, 6)
        chosen = _combo_index(rec["role_seq"][s])
        raw = stage_value_rank(rec, s)
        dense = int((Vr > Vr[chosen]).sum()) + 1
        best_set = frozenset(np.where(Vr == Vr.max())[0])
        rows.append(dict(raw=raw, dense=dense, gap=float(V.max() - V[chosen]), vbest=float(V.max()),
                         vch=float(V[chosen]), n_top=len(best_set), last=(s == rec["n_stages"] - 1),
                         same=(chosen == prev_combo), best_changed=(prev_best is not None and best_set != prev_best),
                         s=s))
        prev_combo, prev_best = chosen, best_set
import pandas as pd
df = pd.DataFrame(rows)
print("player-stages:", len(df))
print("raw != dense (tie-break artefact):", (df.raw != df.dense).sum(), f"({(df.raw != df.dense).mean():.1%})")
print("raw rank 1:", (df.raw == 1).mean().round(3), " dense rank 1:", (df.dense == 1).mean().round(3))
print("stages with >=5 combos tied at the top:", (df.n_top >= 5).sum(), "; of which last stage:", ((df.n_top >= 5) & df.last).sum())
print("median n tied at top, last stage vs not:", df[df.last].n_top.median(), df[~df.last].n_top.median())
# rank increases between consecutive stages (within a round)
inc_raw = inc_dense = inc_dense_real = 0; gaps = []
prev = None
for r in rows:
    if r["s"] > 0 and prev is not None:
        if r["raw"] > prev["raw"]: inc_raw += 1
        if r["dense"] > prev["dense"]:
            inc_dense += 1; gaps.append(r["gap"])
    prev = r
print("consecutive-stage rank increases: raw", inc_raw, " dense", inc_dense)
print("value gap at dense-rank increases: median %.1f, IQR %.1f–%.1f, max %.1f" % (np.median(gaps), *np.percentile(gaps, [25, 75]), np.max(gaps)))
print("dense increases where the team kept the same combo (optimum moved under them):",
      sum(1 for i, r in enumerate(rows) if r["s"] > 0 and r["dense"] > rows[i-1]["dense"] and r["same"]))
print("gap distribution overall (value points, 0-100): dense rank 1 share by stage:")
print(df.groupby("s").agg(n=("dense", "size"), dense1=("dense", lambda x: (x == 1).mean()), raw1=("raw", lambda x: (x == 1).mean()),
                          med_gap=("gap", "median"), mean_gap=("gap", "mean"), med_ntop=("n_top", "median")).round(2))
