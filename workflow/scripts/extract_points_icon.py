"""Extract ICON-CH1/CH2 control-run time series at the SwissMetNet stations.

The ICON counterpart of extract_points.py, writing files in the same format to
output/points/<season>/<label>/<init>.nc so that load_season() reads them like
any other source.

Why not load_forecast_data(): for ICON's unstructured grid, earthkit asks
eckit::geo for the cell coordinates, which expands '~' via a passwd lookup that
fails inside the husk sandbox, and would then download the grid from
sites.ecmwf.int, which is not reachable. The cell-centre coordinates are in the
ICON horizontal-constants GRIB (tlat/tlon) that load_forecast_data() already uses
for orography, so the fields are read here with eccodes directly and the
stations mapped by nearest neighbour on those coordinates, the same way
station_grid_map.py maps them onto REA-L.

TOT_PREC1 is the hourly difference of the accumulated 'tp', NaN at step 0, as
in the ML point files.

    uv run workflow/scripts/extract_points_icon.py --model ICON-CH2-EPS \
        --init 202401010000 --out output/points/winter2024/ICON-CH2-CTRL/202401010000.nc
"""
import argparse, logging
from datetime import datetime
from pathlib import Path

import eccodes
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

LOG = logging.getLogger("extract_points_icon")
ARCHIVE = Path("/store_new/mch/msopr/osm")
CONST = {"ICON-CH1-EPS": ("i1eff", "horizontal_constants_icon-ch1-eps.grib2", 33),
         "ICON-CH2-EPS": ("i2eff", "horizontal_constants_icon-ch2-eps.grib2", 120)}
# eccodes shortName -> point-file variable
FIELDS = {"2t": "T_2M", "2d": "TD_2M", "10u": "U_10M", "10v": "V_10M", "sp": "PS",
          "tp": "TOT_PREC"}


def to_xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def const_fields(path: Path, names) -> tuple[dict, str]:
    out, uuid = {}, None
    with open(path, "rb") as fh:
        while (g := eccodes.codes_grib_new_from_file(fh)) is not None:
            sn = eccodes.codes_get(g, "shortName")
            if sn in names:
                out[sn] = eccodes.codes_get_values(g)
                uuid = eccodes.codes_get(g, "uuidOfHGrid")
            eccodes.codes_release(g)
    return out, uuid


def station_map(model: str, st: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Nearest ICON cell per station, cached as resources/icon_station_map_<model>.csv."""
    cache = Path(f"resources/icon_station_map_{model}.csv")
    c, uuid = const_fields(Path("resources/icon-constants") / CONST[model][1], {"tlat", "tlon"})
    if cache.exists():
        m = pd.read_csv(cache)
        if (m.nat_abbr.to_numpy() == st.nat_abbr.to_numpy()).all():
            return m, uuid
    tree = cKDTree(to_xyz(c["tlat"], c["tlon"]))
    d, idx = tree.query(to_xyz(st["lat"].to_numpy(), st["lon"].to_numpy()))
    m = pd.DataFrame(dict(nat_abbr=st["nat_abbr"], cell=idx, cell_lat=c["tlat"][idx],
                          cell_lon=c["tlon"][idx], distance_m=np.round(d * 6371e3, 1)))
    m.to_csv(cache, index=False)
    LOG.info("wrote %s (max distance %.0f m)", cache, m.distance_m.max())
    return m, uuid


def read_step(path: Path, idx: np.ndarray, uuid: str) -> dict:
    out = {}
    with open(path, "rb") as fh:
        while (g := eccodes.codes_grib_new_from_file(fh)) is not None:
            sn = eccodes.codes_get(g, "shortName")
            if sn in FIELDS and FIELDS[sn] not in out:
                if eccodes.codes_get(g, "uuidOfHGrid") != uuid:
                    raise SystemExit(f"{path}: grid uuid differs from the constants file")
                out[FIELDS[sn]] = eccodes.codes_get_values(g)[idx]
            eccodes.codes_release(g)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=CONST, required=True)
    ap.add_argument("--init", required=True, help="YYYYmmddHHMM")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--member", default="000")
    ap.add_argument("--stations", type=Path, default=Path("resources/smn_grid_points.csv"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.out.exists():
        LOG.info("%s exists, nothing to do", args.out)
        return

    prefix, _, max_step = CONST[args.model]
    init = datetime.strptime(args.init, "%Y%m%d%H%M")
    dirs = sorted((ARCHIVE / args.model / f"FCST{init:%y}").glob(f"{init:%y%m%d%H}_*"))
    if not dirs:
        raise SystemExit(f"no archive directory for {init:%y%m%d%H}")
    gdir = dirs[-1] / "grib"   # same choice as data_input._collect_icon_archive_files
    st = pd.read_csv(args.stations)
    m, uuid = station_map(args.model, st)
    idx = m["cell"].to_numpy()

    steps = list(range(max_step + 1))
    rows = [read_step(gdir / f"{prefix}{s // 24:02}{s % 24:02}0000_{args.member}", idx, uuid)
            for s in steps]
    data = {v: np.stack([r.get(v, np.full(len(idx), np.nan)) for r in rows])
            for v in FIELDS.values()}
    tp = data.pop("TOT_PREC")
    data["TOT_PREC1"] = np.vstack([np.full((1, len(idx)), np.nan), np.diff(tp, axis=0)])

    out = xr.Dataset(
        {v: (("step", "station"), a.astype("float32")) for v, a in data.items()},
        coords={
            "step": np.asarray(steps, dtype="int16"),
            "station": st["nat_abbr"].to_numpy(),
            "valid_time": ("step", np.array([np.datetime64(init, "ns") + np.timedelta64(s, "h")
                                             for s in steps])),
            "grid_index": ("station", idx),
            "lat": ("station", st["lat"].to_numpy()),
            "lon": ("station", st["lon"].to_numpy()),
        },
        attrs={"init": init.isoformat(), "grib_dir": str(gdir), "member": args.member},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(args.out)
    LOG.info("wrote %s (%d steps x %d stations)", args.out, len(steps), len(idx))


if __name__ == "__main__":
    main()
