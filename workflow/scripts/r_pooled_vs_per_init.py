"""Pooled band timing correlation against the mean of per-init correlations.

joint_variance_skill.py forms r from init-averaged covariance and variances
(pooled), which the amplitude/phase split of the band MSE requires. This checks
that the pooled choice does not drive the result: per (season, param), median
over stations of pooled r, of the plain mean of per-init r, and of the
Fisher-z mean, for both models. Reads output/joint_variance_skill/table.csv only.

    uv run workflow/scripts/r_pooled_vs_per_init.py
"""
import numpy as np, pandas as pd

t = pd.read_csv("output/joint_variance_skill/table.csv")
rows = []
for (season, param), g in t.groupby(["season", "param"], sort=False):
    row = dict(season=season, param=param)
    for s in ("varda", "multi"):
        ri = g[f"cov_{s}"] / np.sqrt(g[f"bvar_{s}"] * g["bvar_truth"])
        m = g.assign(ri=ri, zi=np.arctanh(ri.clip(-0.999, 0.999))
                     ).groupby("station").mean(numeric_only=True)
        row[f"pooled_{s}"] = (m[f"cov_{s}"]
                              / np.sqrt(m[f"bvar_{s}"] * m["bvar_truth"])).median()
        row[f"mean_r_{s}"] = m["ri"].median()
        row[f"fisher_{s}"] = np.tanh(m["zi"]).median()
    rows.append(row)
print(pd.DataFrame(rows).round(3).to_string(index=False))
