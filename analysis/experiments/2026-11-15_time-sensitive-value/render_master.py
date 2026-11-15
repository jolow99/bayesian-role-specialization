"""Render the master table (all_fits.json) as two shareable PNGs via headless Chrome.

  master_summary.png     Bayesian-Walk-BR (winner), every variant, all metrics + fitted params
  master_all_models.png  all six Bayesian models, every variant: agg-LL and combo-r

Run (plain python, no deps):  python3 render_master.py [numpy|memo]
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = sys.argv[1] if len(sys.argv) > 1 else "numpy"
D = HERE / f"master_{SRC}"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
FITS = json.load(open(D / "all_fits.json"))
MODELS = ["Bayesian-BR", "Bayesian-Walk-BR", "Bayesian-Threshold-BR",
          "Bayesian-PS", "Bayesian-Walk-PS", "Bayesian-Threshold-PS"]
SHORT = {"Bayesian-BR": "BR", "Bayesian-Walk-BR": "Walk-BR", "Bayesian-Threshold-BR": "Thr-BR",
         "Bayesian-PS": "PS", "Bayesian-Walk-PS": "Walk-PS", "Bayesian-Threshold-PS": "Thr-PS"}
GROUPS = [
    ("U", "Updaters (n = 50) — own beliefs, refit"),
    ("NU-FTM", "Non-updaters (n = 52) — paper's fixed FTM prior, refit"),
    ("NU-as-U", "Non-updaters (n = 52) — on updater beliefs, refit"),
    ("NU-as-U-nr", "Non-updaters (n = 52) — on updater beliefs AND updater params (no refit)"),
]
VALUES = [("paper", "paper V(10)"), ("time-aware", "time-aware V(10−2·stage), refit"),
          ("time-aware (frozen)", "time-aware V, params frozen at paper fit")]
NAMES = {"tau_softmax": "τ<sub>v</sub>", "epsilon_switch": "ε<sub>s</sub>", "delta": "δ"}


def params(p):
    return ", ".join(f"{NAMES[k]} = {v:.3g}" for k, v in p.items()) or "–"


def variants(g):
    out = []
    for v, vlabel in VALUES:
        for b in (("n/a",) if g == "NU-FTM" else ("mixed", "unmixed")):
            if f"{g}|{v}|{b}" in FITS:
                if g == "NU-as-U-nr":
                    vlabel = {"paper": "paper V(10), updaters' paper params",
                              "time-aware": "time-aware V, updaters' time-aware params",
                              "time-aware (frozen)": "time-aware V, updaters' paper params"}[v]
                out.append((v, vlabel, b, FITS[f"{g}|{v}|{b}"]))
    return out


CSS = """
body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:28px 34px;color:#1c1c1e;background:#fff;width:1500px}
h1{font-size:25px;margin:0 0 4px} .sub{color:#555;font-size:14px;margin-bottom:18px;line-height:1.45}
h2{font-size:16px;margin:22px 0 6px;color:#1a3c6e}
table{border-collapse:collapse;font-size:13.5px;width:100%;margin-bottom:4px}
th{background:#eef2f8;text-align:right;padding:5px 9px;border-bottom:2px solid #b9c6dc;white-space:nowrap}
th:first-child,th:nth-child(2),th.l{text-align:left}
td{padding:4px 9px;text-align:right;border-bottom:1px solid #e6e6e6;font-variant-numeric:tabular-nums;white-space:nowrap}
td:first-child,td:nth-child(2),td.l{text-align:left}
tr.paper td{background:#fafafa} tr.frozen td{color:#555;font-style:italic}
b.best{background:#fff1b8;padding:0 3px;border-radius:3px}
.note{font-size:12.5px;color:#555;margin-top:14px;line-height:1.5}
"""


def page(title, sub, body, note):
    return (f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body><h1>{title}</h1>"
            f"<div class='sub'>{sub}</div>{body}<div class='note'>{note}</div></body></html>")


def rowclass(v):
    return "frozen" if "frozen" in v else ("paper" if v == "paper" else "")


def summary():
    body = ""
    for g, gname in GROUPS:
        vs = variants(g)
        best = max(r["Bayesian-Walk-BR"]["agg_ll"] for *_, r in vs)
        body += (f"<h2>{gname}</h2><table><tr><th>Value function</th><th>Belief</th><th class='l'>Fitted params</th>"
                 "<th>agg-LL</th><th>mean-LL</th><th>marg-r</th><th>combo-r</th><th>Σ participant LL</th></tr>")
        for v, vlabel, b, res in vs:
            r = res["Bayesian-Walk-BR"]
            agg = f"{r['agg_ll']:.4f}"
            agg = f"<b class='best'>{agg}</b>" if abs(r["agg_ll"] - best) < 1e-12 else agg
            body += (f"<tr class='{rowclass(v)}'><td>{vlabel}</td><td>{b}</td><td class='l'>{params(r['params'])}</td>"
                     f"<td>{agg}</td><td>{r['mean_ll']:.4f}</td><td>{r['marg_r']:.3f}</td><td>{r['combo_r']:.3f}</td>"
                     f"<td>{sum(r['participant_ll'].values()):.1f}</td></tr>")
        body += "</table>"
    return page("Bayesian-Walk-BR across every variant",
                "Winning model (best agg-LL) in every variant. Value function × belief × group. Fit objective: agg-LL; "
                "inference params fixed (Nov 2 fit). Yellow = best agg-LL within each group.",
                body,
                "mixed = (1−β)·P<sub>N</sub> + β·P<sub>0</sub> (paper); unmixed = P<sub>N</sub>. Frozen rows reuse the paper-V "
                "fitted params with no refit (they show the effect of swapping V alone; τ<sub>v</sub> was tuned to the paper's V "
                "so this slightly disadvantages time-aware). Non-updaters on the FTM prior have no mixed/unmixed distinction. "
                "PS-type models never use V.")


def all_models():
    body = ""
    for g, gname in GROUPS:
        vs = variants(g)
        body += (f"<h2>{gname}</h2><table><tr><th>Value function</th><th>Belief</th>"
                 + "".join(f"<th>{SHORT[m]}<br><span style='font-weight:400'>agg-LL · combo-r</span></th>" for m in MODELS)
                 + "</tr>")
        for v, vlabel, b, res in vs:
            best = max(MODELS, key=lambda m: res[m]["agg_ll"])
            cells = ""
            for m in MODELS:
                t = f"{res[m]['agg_ll']:.4f} · {res[m]['combo_r']:.3f}"
                cells += f"<td>{'<b class=best>' + t + '</b>' if m == best else t}</td>"
            body += f"<tr class='{rowclass(v)}'><td>{vlabel}</td><td>{b}</td>{cells}</tr>"
        body += "</table>"
    return page("All six Bayesian models across every variant",
                "Each cell: agg-LL · combo-r. Yellow = best agg-LL in the row. Same variants and conventions as the summary figure.",
                body,
                "Full parameters, mean-LL, marg-r and Σ participant LL for every cell are in master.md / master.csv.")


def shot(name, html, height):
    (D / f"{name}.html").write_text(html)
    # taller-than-needed window, then trim by rendering at a measured height
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                    f"--window-size=1560,{height}", f"--screenshot={D / (name + '.png')}", f"file://{D / (name + '.html')}"],
                   check=True, capture_output=True)
    print("wrote", D / f"{name}.png")


if __name__ == "__main__":
    shot("master_summary", summary(), 1030)
    shot("master_all_models", all_models(), 1010)
