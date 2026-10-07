"""Extract REA-L-CH1 at the 145 station grid points, for the forecast valid times.

Truth counterpart of extract_points.py. The forecast spectra must be compared
against truth on the *same* valid times and the *same* points, otherwise the two
samples see different weather and different terrain.

Writes one file per time chunk so that partial progress survives if the run is
interrupted (written for the 2026-09-15 CAPSTOR unmount, where the filesystem
could disappear mid-run). Chunks already present are skipped, so rerunning
resumes.

Run one chunk:
    uv run python workflow/scripts/extract_truth_points.py --chunk 3 --n-chunks 8
All chunks in parallel on the login node:
    for i in $(seq 0 7); do uv run python workflow/scripts/extract_truth_points.py \
        --chunk $i --n-chunks 8 & done; wait
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

LOG = logging.getLogger("extract_truth_points")

REAL_ZARR = "/store_new/mch/msopr/ml/datasets/mch-realch1-fdb-1km-2005-2025-1h-pl13-v1.0.zarr"
# REA-L names; TOT_PREC_1H is the hourly accumulation (TOT_PREC1 on the forecast side).
PARAMS = ("T_2M", "TD_2M", "U_10M", "V_10M", "TOT_PREC_1H", "PS")


def wanted_times(points_root: Path) -> np.ndarray:
    """Union of valid times over every extracted forecast file."""
    times = set()
    for f in sorted(points_root.rglob("*.nc")):
        with xr.open_dataset(f) as d:
            times.update(np.asarray(d["valid_time"].values).tolist())
    return np.array(sorted(times), dtype="datetime64[ns]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--points-root", type=Path, default=Path("output/points"))
    ap.add_argument("--stations", type=Path, default=Path("resources/smn_grid_points.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("output/points_truth"))
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--n-chunks", type=int, default=1)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format=f"%(asctime)s [chunk {args.chunk}] %(message)s")

    out = args.out_dir / f"truth_{args.chunk:03d}_of_{args.n_chunks:03d}.nc"
    if out.exists():
        LOG.info("%s exists, nothing to do", out)
        return

    times = wanted_times(args.points_root)
    mine = np.array_split(times, args.n_chunks)[args.chunk]
    LOG.info("%d of %d valid times", len(mine), len(times))

    st = pd.read_csv(args.stations)
    pt = st["grid_index"].to_numpy()

    z = zarr.open(REAL_ZARR, mode="r")
    dates = z["dates"][:]
    vidx = [list(z.attrs["variables"]).index(p) for p in PARAMS]

    idx = np.searchsorted(dates, mine.astype("datetime64[s]"))
    if not np.array_equal(dates[idx], mine.astype(dates.dtype)):
        raise SystemExit("some requested valid times are not in REA-L")

    data = np.empty((len(mine), len(PARAMS), len(pt)), dtype="float32")
    for k, i in enumerate(idx):
        data[k] = z["data"].oindex[int(i), vidx, 0, :][:, pt]
        if k % 200 == 0:
            LOG.info("  %d/%d", k, len(mine))

    ds = xr.Dataset(
        {p: (("time", "station"), data[:, j]) for j, p in enumerate(PARAMS)},
        coords={"time": mine, "station": st["nat_abbr"].to_numpy(),
                "grid_index": ("station", pt)},
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out)
    LOG.info("wrote %s (%d times x %d stations)", out, len(mine), len(pt))


if __name__ == "__main__":
    main()
