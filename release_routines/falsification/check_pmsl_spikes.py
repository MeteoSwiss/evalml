"""For every RC forecast whose PMSL exceeds LIMIT (lead time > 0), take the lead time with the highest PMSL and describe the
cells above LIMIT: count, position, and how far they stand above the median of their surrounding cells (ICON neighbours
up to 2 steps away, i.e. ~12 cells within ~2 km). A large excess over the surroundings means a single-cell spike."""

from pathlib import Path

import eccodes
import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).parent
RUN = (
    HERE.parents[1]
    / "output/data/runs/temporal_downscaler-d642-b38b-on-forecaster-07f6-6e42/14ea"
)
ICON_GRID = Path("/store_new/mch/msopr/ml/grids/icon_grid_0001_R19B08_mch.nc")
LIMIT = 1041.0  # hPa, above anything KENDA has in the whole period

with xr.open_dataset(ICON_GRID) as g:
    nb = g["neighbor_cell_index"].values.T - 1  # (cell, 3), 0-based, <0 = none
    clon, clat = np.degrees(g["clon"].values), np.degrees(g["clat"].values)


def surroundings(c):
    n1 = nb[c][nb[c] >= 0]
    n2 = nb[n1].ravel()
    return np.setdiff1d(np.union1d(n1, n2[n2 >= 0]), [c])


def read_pmsl(f):
    with open(f, "rb") as fh:
        while (h := eccodes.codes_grib_new_from_file(fh)) is not None:
            try:
                if eccodes.codes_get(h, "shortName") == "PMSL":
                    return eccodes.codes_get_values(h) / 100
            finally:
                eccodes.codes_release(h)


rows = []
for f in sorted(RUN.glob("20*/verif_f791.nc")):
    with xr.open_dataset(f) as ds:
        x = (
            ds["PMSL.max"]
            .sel(region="icon")
            .isel(source=0, forecast_reference_time=0)
            .isel(step=slice(1, None))
            .load()
            / 100
        )
    if float(x.max()) <= LIMIT:
        continue
    init = f.parent.name
    step = int(x.step.values[int(x.argmax())] / np.timedelta64(1, "h"))
    v = read_pmsl(f.parent / "grib" / f"{init}_{step}.grib")
    assert v.size == clon.size
    cells = np.flatnonzero(v > LIMIT)
    excess = np.array([v[c] - np.median(v[surroundings(c)]) for c in cells])
    k = cells[np.argmax(v[cells])]
    rows.append(
        {
            "init": init,
            "step": step,
            "n_hours_above": int((x > LIMIT).sum()),
            "n_cells_above": cells.size,
            "max_hPa": round(float(v[k]), 1),
            "lon": round(float(clon[k]), 2),
            "lat": round(float(clat[k]), 2),
            "surroundings_median_hPa": round(float(np.median(v[surroundings(k)])), 1),
            "min_excess_of_cells_above_hPa": round(float(excess.min()), 1),
        }
    )
    print(rows[-1], flush=True)

df = pd.DataFrame(rows)
df.to_csv(HERE / "pmsl_spike_check.csv", index=False)
print(df.to_string(index=False))
