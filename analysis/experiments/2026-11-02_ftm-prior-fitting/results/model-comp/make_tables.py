"""One markdown table per (group, model family) from response-fitting/fit-results.

Run from analysis/:
    uv run python experiments/2026-11-02_ftm-prior-fitting/results/model-comp/make_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
FIT_DIR = OUT_DIR.parents[1] / "response-fitting" / "fit-results"
METRICS = [("combo_r", "combo-r*"), ("marg_r", "marg-r"), ("agg_ll", "agg-LL"), ("mean_ll", "mean-LL")]


def pstr(p):
    out = []
    if "tau_softmax" in p:
        out.append(f"τ_v = {p['tau_softmax']:.2f}")
    if "epsilon_switch" in p:
        out.append(f"ε_s = {1 - p['epsilon_switch']:.2f}")  # paper's ε_s = 1 − ε_switch
    if "delta" in p:
        out.append(f"δ = {p['delta']:.3f}")
    if "eps" in p:
        out.append(f"ε = {p['eps']:.3f}")
    if "k" in p:
        out.append(f"k = {p['k']} (fixed)")
    return ", ".join(out) or "—"


def main():
    for group in ("updaters", "non-updaters", "population"):
        for fam in ("bayes-models", "non-bayes-models"):
            res = json.load(open(FIT_DIR / group / f"{fam}.json"))
            n = len(next(iter(res.values()))["participant_ll"])
            best = {k: max(r[k] for r in res.values()) for k, _ in METRICS}
            lines = [f"# {group} — {fam}", "",
                     f"Fit on {n} {group} by per-player agg-LL (λ = 0.01 lapse); higher is better. "
                     "Bold = best in column. *combo-r is team-level: all 204 team-rounds, scored "
                     f"with the {group} fit, each player on their own type's belief.", "",
                     "| Model | Fitted params | " + " | ".join(h for _, h in METRICS) + " |",
                     "|---|---|" + "--:|" * len(METRICS)]
            for m, r in res.items():
                cells = [f"**{r[k]:.3f}**" if r[k] == best[k] else f"{r[k]:.3f}" for k, _ in METRICS]
                lines.append(f"| {m} | {pstr(r['params'])} | " + " | ".join(cells) + " |")
            (OUT_DIR / f"{group}_{fam}.md").write_text("\n".join(lines) + "\n")
        write_tex(group)


def write_tex(group):
    """Table 2 rows (LaTeX) for one group's fit, best per column in bold."""
    res = {}
    for fam in ("bayes-models", "non-bayes-models"):
        res.update(json.load(open(FIT_DIR / group / f"{fam}.json")))
    cols = ["combo_r", "marg_r", "agg_ll", "mean_ll"]
    best = {c: max(round(r[c], 3) for r in res.values()) for c in cols}

    def cell(c, x):
        x = round(x, 3) + 0.0
        if x == best[c]:
            return f"\\textbf{{{x:.3f}}}" if x >= 0 else f"$\\mathbf{{{x:.3f}}}$"
        return f"{x:.3f}".replace("-", "$-$")
    rows = [f"{'Top-$k$ ($k=5$)' if m == 'Top-5' else m} & " + " & ".join(cell(c, r[c]) for c in cols)
            + " \\\\" for m, r in res.items()]
    rows.insert(6, "\\midrule")
    (OUT_DIR / f"{group}_table2.tex").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
