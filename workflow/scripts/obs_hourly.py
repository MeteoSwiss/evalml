"""Build hourly SwissMetNet series from the 10-minute DWH data.

The DWH returned 10-minute data despite the hourly increment request. We take
the value stamped HH:00 as the instantaneous value at the full hour, on the
assumption that a 10-minute record is stamped at the END of its interval, so
15:00 covers 14:50-15:00. That assumption is recorded, not verified: the CLI
exposes no parameter metadata and no local catalogue documents it. It only
affects event timing, not spectral power, since a 5-minute error is a phase
shift.

Gap filling, in order, per station and hour:

  1. the value stamped HH:00
  2. HH:10, else HH-1:50          (one 10-minute step away)
  3. HH:20, else HH-1:40          (two steps away)
  4. linear interpolation between the nearest available 10-minute values

Each filled value is tagged with the stage used, so the analysis can exclude or
report them. This matters: interpolation is a low-pass filter, so filled values
carry less high-frequency variance than real ones and bias the observed spectrum
downwards, which flatters the models.

Wind is kept as measured speed (fkl010z0). Direction is deliberately unused, so
no circular interpolation problem arises; the model U and V are converted to
speed instead.

    uv run workflow/scripts/obs_hourly.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

LOG = logging.getLogger("obs_hourly")

OBS = Path("output/obs/obs_all.csv")
STATIONS = Path("resources/smn_grid_points.csv")
OUT = Path("output/points_obs/obs_hourly.nc")
# DWH short name -> our name, and the conversion to model units
VARS = {"tre200s0": ("T_2M", 273.15), "tde200s0": ("TD_2M", 273.15),
        "fkl010z0": ("SP_10M", 0.0)}
# offsets in 10-minute steps, in the order they are tried
LADDER = [0, 1, -1, 2, -2]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    st = pd.read_csv(STATIONS)
    id2abbr = dict(zip(st["station_id"], st["nat_abbr"]))

    LOG.info("reading %s", OBS)
    d = pd.read_csv(OBS)
    d = d[d["station"].isin(id2abbr)].copy()
    d["time"] = pd.to_datetime(d["termin"], format="%Y%m%d%H%M%S")
    # chunk boundaries overlap by one timestamp
    d = d.drop_duplicates(subset=["station", "time"]).sort_values(["station", "time"])
    LOG.info("%d rows, %d of %d sample stations", len(d), d["station"].nunique(), len(st))

    abbrs = st["nat_abbr"].tolist()
    full10 = pd.date_range(d["time"].min(), d["time"].max(), freq="10min")
    # Only the hours the forecasts actually cover. The fetch blocks are
    # separated by months with no data, and counting those as "missing" would
    # make the fill statistics meaningless.
    need = xr.open_mfdataset(sorted(Path("output/points_truth").glob("*.nc")),
                             combine="by_coords")["time"].values
    hours = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(need)) & set(full10)))
    LOG.info("%d ten-minute slots -> %d forecast hours (of %d wanted)",
             len(full10), len(hours), len(need))

    data, prov = {}, {}
    for short, (name, offset) in VARS.items():
        vals = np.full((len(hours), len(abbrs)), np.nan)
        stage = np.zeros((len(hours), len(abbrs)), dtype="int8")  # 0=direct..4, 5=interp, -1=none
        for j, ab in enumerate(abbrs):
            sid = st.loc[st["nat_abbr"] == ab, "station_id"].iloc[0]
            s = d.loc[d["station"] == sid, ["time", short]].set_index("time")[short]
            s = s.reindex(full10)  # regular 10-minute grid, gaps become NaN
            if offset:
                s = s + offset
            hs = s.reindex(hours)
            take = hs.to_numpy(float).copy()
            st_arr = np.where(np.isnan(take), -1, 0).astype("int8")
            for k, off in enumerate(LADDER[1:], start=1):
                need = np.isnan(take)
                if not need.any():
                    break
                cand = s.shift(-off).reindex(hours).to_numpy(float)
                use = need & ~np.isnan(cand)
                take[use] = cand[use]
                st_arr[use] = k
            need = np.isnan(take)
            if need.any():  # linear interpolation over the 10-minute series
                filled = s.interpolate(method="time", limit_direction="both")
                cand = filled.reindex(hours).to_numpy(float)
                use = need & ~np.isnan(cand)
                take[use] = cand[use]
                st_arr[use] = 5
            vals[:, j] = take
            stage[:, j] = st_arr
        data[name] = (("time", "station"), vals.astype("float32"))
        prov[name + "_fill"] = (("time", "station"), stage)
        tot = stage.size
        LOG.info("%-7s direct %5.2f%%  +-10min %5.2f%%  +-20min %5.2f%%  interp %5.2f%%  none %5.2f%%",
                 name, 100 * (stage == 0).mean(),
                 100 * np.isin(stage, [1, 2]).mean(), 100 * np.isin(stage, [3, 4]).mean(),
                 100 * (stage == 5).mean(), 100 * (stage == -1).mean())

    ds = xr.Dataset({**data, **prov},
                    coords={"time": hours, "station": abbrs},
                    attrs={"source": "SwissMetNet 10-minute DWH data",
                           "convention": "value stamped HH:00 assumed to cover HH-10min..HH:00",
                           "fill_stages": "0=direct 1=+10min 2=-10min 3=+20min 4=-20min 5=interp -1=missing"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(OUT)
    print(f"\nwrote {OUT}  ({len(hours)} hours x {len(abbrs)} stations)")


if __name__ == "__main__":
    main()
