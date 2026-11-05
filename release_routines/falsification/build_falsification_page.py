"""Build the Confluence page body (storage format) of the Falsification section from the newest falsification_table_*.csv."""

from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
table_csv = max(HERE.glob("falsification_table_*.csv"))
df = pd.read_csv(table_csv, keep_default_na=False)
MODELS = ["RC", "Varda 1.0", "ICON-CH1", "ICON-CH2", "KENDA"]
COLOUR = {"OK": "Green", "marginal": "Yellow", "deficient": "Red"}


def status(flag):
    return (
        f'<ac:structured-macro ac:name="status"><ac:parameter ac:name="colour">{COLOUR[flag]}</ac:parameter>'
        f'<ac:parameter ac:name="title">{escape(flag)}</ac:parameter></ac:structured-macro>'
    )


cell = lambda x: f"<td><p>{x}</p></td>"
head = "".join(
    f"<th><p>{escape(h)}</p></th>"
    for h in ["Category", "Variable", "Flag", *MODELS, "Comment"]
)
rows = "".join(
    "<tr>"
    + cell(escape(r.Category))
    + cell(f"{escape(r.Variable)} [{escape(r.Unit)}]")
    + cell(status(r.Flag))
    + "".join(cell(escape(str(r[m]))) for m in MODELS)
    + cell(escape(r.Comment))
    + "</tr>"
    for _, r in df.iterrows()
)
body = f"""<h2>Falsification</h2>
<p>The idea here is to check whether the release candidate produces any completely unrealistic values.</p>
<p>Generated automatically from the per-forecast verification files of experiment rc-cloud-skt-eval-2025-04-2026-03
(209 forecasts from 2025-04 to 2026-03, whole ICON domain, lead times +1 h to +120 h; ICON-CH1 only up to +33 h).</p>
<ul>
<li><strong>Maxima / Minima:</strong> highest / lowest value at any grid point over all forecasts and lead times.</li>
<li><strong>Mean:</strong> average of the hourly domain means. The flag checks every single hourly domain mean against the limits.</li>
<li><strong>Flag:</strong> whether the release candidate (RC) stays within physical limits (see Comment); the other columns are for reference.
Two flags were set by hand after inspecting the cases (see Comment).</li>
<li><strong>NA:</strong> variable not available (Varda 1.0 has no clouds or SKT; ICON's T_G is ground temperature, not skin temperature).</li>
</ul>
<table><tbody><tr>{head}</tr>{rows}</tbody></table>
<p><em>Source: {escape(table_csv.name)}</em></p>
"""
out = HERE / f"falsification_page_{datetime.now():%Y-%m-%d_%H-%M-%S}.xhtml"
out.write_text(body)
print(out)
