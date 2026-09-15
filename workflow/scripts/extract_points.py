"""Extract forecast time series at the SwissMetNet grid points, one init at a time.

Stage 1 of the two-stage temporal-variability analysis in exp_plan_01.md. Reading
the GRIB is the expensive part (~1.7 s per step, because each 164 MB file holds
the full state and we decode it to take a few fields at 145 points), so it is
done once and cached. Stage 2 (spectra) then works on these small files and can
be rerun freely when the window length or the diagnostic changes.

The **full 0-120 h series** is written, undivided. Slicing into 48 h lead-time
windows happens later on the cheap cached data, so that choice stays revisable
without touching GRIB again.

The GRIB output and the REA-L zarr share bit-identical point ordering (verified),
so the grid_index column of the station mapping addresses both. Truth and
forecasts are therefore read at exactly the same points by construction.

Surface fields only: evalml's GRIB loader has no concept of pressure levels.
PS is included as a control, being a field with little genuine sub-6 h
variability. PMSL is deliberately absent: the ICON-template output names it
`prmsl`, which the loader does not map, and it was dropped from this project
earlier for reasons we could not reconstruct.

Run (normally via workflow/scripts/extract_points.sbatch):
    uv run python workflow/scripts/extract_points.py \
        --grib-dir output/winter2024/data/runs/<run>/<hash>/<init>/grib \
        --out output/points/winter2024/<label>/<init>.nc
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import data_input
from data_input import load_forecast_data

LOG = logging.getLogger("extract_points")

# load_forecast_data() tries to attach an elevation coordinate. When the run's
# step-0 GRIB carries no FIS (the temporal downscaler's output does not) it falls
# back to an ICON topography lookup that raises KeyError('HSURF') rather than
# giving up, so the whole load fails. We never use that coordinate: station and
# model elevations come from resources/smn_grid_points.csv. Disable it here
# instead of changing shared code.
data_input._try_assign_elevation = lambda ds: ds

PARAMS = ("T_2M", "TD_2M", "U_10M", "V_10M", "TOT_PREC1", "PS")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grib-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stations", type=Path, default=Path("resources/smn_grid_points.csv"))
    ap.add_argument("--params", nargs="+", default=list(PARAMS))
    ap.add_argument("--max-step", type=int, default=120)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.out.exists():
        LOG.info("%s exists, nothing to do", args.out)
        return

    st = pd.read_csv(args.stations)
    pt = st["grid_index"].to_numpy()

    # Init time from the directory name (.../<YYYYmmddHHMM>/grib).
    init = datetime.strptime(args.grib_dir.parent.name, "%Y%m%d%H%M")

    # Steps actually present: hourly runs write 0..120, the 6-hourly forecaster
    # writes 0,6,...,120. Derive from the files rather than assuming. The
    # 'ifs-' files hold the global points and are not wanted here.
    steps = sorted({int(f.stem.rsplit("_", 1)[1]) for f in args.grib_dir.glob("*.grib")
                    if not f.name.startswith("ifs-")})
    steps = [s for s in steps if s <= args.max_step]
    if not steps:
        raise SystemExit(f"no usable GRIB steps in {args.grib_dir}")
    spacing = steps[1] - steps[0] if len(steps) > 1 else 1
    LOG.info("%s: %d steps (%d..%d, every %d h)",
             args.grib_dir, len(steps), steps[0], steps[-1], spacing)

    # A 1 h accumulation cannot be formed from 6-hourly output. This is not a
    # technicality to work around: the 6-hourly forecaster simply has no hourly
    # precipitation, so it gets no interpolated-precipitation reference line
    # either. Linearly interpolating a 6 h accumulation to hourly would invent a
    # quantity the model never produced.
    params = list(args.params)
    if spacing > 1 and "TOT_PREC1" in params:
        params.remove("TOT_PREC1")
        LOG.info("dropping TOT_PREC1: output is %d-hourly, no 1 h accumulation", spacing)

    ds = load_forecast_data(args.grib_dir, init, steps, params)

    data = {}
    for p in params:
        a = np.asarray(ds[p].values).reshape(len(steps), -1)
        data[p] = (("step", "station"), a[:, pt].astype("float32"))

    out = xr.Dataset(
        data,
        coords={
            "step": np.asarray(steps, dtype="int16"),
            "station": st["nat_abbr"].to_numpy(),
            "valid_time": ("step", np.array(
                [np.datetime64(init, "ns") + np.timedelta64(s, "h") for s in steps])),
            "grid_index": ("station", pt),
            "lat": ("station", st["lat"].to_numpy()),
            "lon": ("station", st["lon"].to_numpy()),
        },
        attrs={"init": init.isoformat(), "grib_dir": str(args.grib_dir)},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(args.out)
    LOG.info("wrote %s (%d steps x %d stations x %d params)",
             args.out, len(steps), len(pt), len(params))


if __name__ == "__main__":
    main()
