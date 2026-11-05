"""Build the Confluence page body (storage format) of the Verification section from the newest verification_flags_*.csv.
Publish with ../publish_confluence_page.py <existing_page_url> <title> <body.xhtml> [--draft]."""

from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).parent
csv = max(HERE.glob("verification_flags_*.csv"))
df = pd.read_csv(csv, keep_default_na=False, na_values=[""])
TOL = yaml.safe_load((HERE / "verification_tolerances.yaml").read_text())
BASES = {
    "Varda 1.0": "Varda-single 1.0 (+1 to +120 h)",
    "ICON-CH1": "ICON-CH1 (+1 to +33 h)",
    "ICON-CH2": "ICON-CH2 (+1 to +120 h)",
}
COLOUR = {"OK": "Green", "marginal": "Yellow", "deficient": "Red", "NA": "Grey"}


def status(flag):
    return (
        f'<ac:structured-macro ac:name="status"><ac:parameter ac:name="colour">{COLOUR[flag]}</ac:parameter>'
        f'<ac:parameter ac:name="title">{escape(flag)}</ac:parameter></ac:structured-macro>'
    )


def value(r, sc):
    if r[f"{sc}_flag"] == "NA":
        return status("NA")
    if sc == "RMSE":
        txt = f"{r.RMSE_ratio:.3f} [{r.RMSE_lo:.3f}, {r.RMSE_hi:.3f}]"
    else:
        txt = f"{r.BIAS_diff:+.3g} [{r.BIAS_lo:+.3g}, {r.BIAS_hi:+.3g}] {r.unit}"
    return f"{status(r[f'{sc}_flag'])} {escape(txt)}"


cell = lambda x: f"<td><p>{x}</p></td>"
head = "".join(
    f"<th><p>{escape(h)}</p></th>"
    for h in ["Variable", "Score", *BASES.values(), "Comment"]
)
rows = ""
for v in df.variable.unique():
    d = df[df.variable == v].set_index("baseline")
    unit = next((u for u in d.unit if isinstance(u, str)), "")
    for sc, label in (("RMSE", "RMSE ratio"), ("BIAS", "|BIAS| difference")):
        rows += (
            "<tr>"
            + cell(f"{escape(v)}" + (f" [{escape(unit)}]" if unit else ""))
            + cell(label)
            + "".join(cell(value(d.loc[b], sc)) for b in BASES)
            + cell("")
            + "</tr>"
        )
tols = ", ".join(f"{v} {t['tol']} {t['unit']}" for v, t in TOL["BIAS"].items())
body = f"""<h2>Verification</h2>
<p>Here, the flags indicate whether the release candidate (RC) brings an added value over a given baseline, one flag per baseline,
restricted to RMSE and BIAS.</p>
<p>Generated automatically from the per-forecast verification files of experiment rc-cloud-skt-eval-2025-04-2026-03
(209 forecasts from 2025-04 to 2026-03, inits every 42 h, whole ICON domain, truth KENDA-CH1, all seasons and init hours).</p>
<ul>
<li><strong>Lead times:</strong> all lead times available for the baseline (see column header).</li>
<li><strong>RMSE ratio:</strong> RMSE of RC / RMSE of baseline, with the RMSE first computed per lead time over all forecasts
and then averaged over lead times. Below 1: RC better.</li>
<li><strong>|BIAS| difference:</strong> |BIAS| of RC minus |BIAS| of baseline, in physical units, with the BIAS first computed per lead time
over all forecasts (systematic error at that lead time), then made absolute and averaged over lead times. Below 0: RC better.</li>
<li><strong>[range]:</strong> 95 % range from a paired block bootstrap (10,000 rounds, circular overlapping blocks of 3 consecutive
forecasts, about 5 days, because forecasts overlap in time). Blocks of 6 gave practically the same ranges.</li>
<li><strong>Flag</strong>, based on the lower bound of the range:
<ul><li>OK: RC not significantly worse (lower bound at or below no change).</li>
<li>marginal: RC significantly worse, but possibly by less than the tolerance.</li>
<li>deficient: RC significantly worse by more than the tolerance (lower bound above the tolerance).</li></ul>
Tolerances: RMSE ratio {TOL["RMSE_ratio"]}; |BIAS| difference {escape(tols)}.</li>
<li><strong>NA:</strong> variable not available for the baseline (Varda-single 1.0 has no clouds or SKT;
ICON's T_G is ground temperature, not skin temperature).</li>
</ul>
<table><tbody><tr>{head}</tr>{rows}</tbody></table>
<p><em>Source: {escape(csv.name)}</em></p>
"""
out = HERE / f"verification_page_{datetime.now():%Y-%m-%d_%H-%M-%S}.xhtml"
out.write_text(body)
print(out)
