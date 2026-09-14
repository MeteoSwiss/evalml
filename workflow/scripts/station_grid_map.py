"""Map the SwissMetNet station sample onto REA-L-CH1 grid points (nearest neighbour).

Produces the fixed point set used by every temporal-variability diagnostic in
exp_plan_01.md, so that the truth spectra, the forecast spectra and (later) the
station observations are all evaluated at the same locations.

Station coordinates come from output/station_selection/station_sample.csv, which
already carries latitude, longitude and elevation. No DWH query is needed.

Also reports two things that matter for interpretation:

- Horizontal distance station to grid point. On a 1 km grid this should be a few
  hundred metres. Anything large means the station sits outside the domain.
- Elevation difference station minus model. This is the representativeness gap.
  A station several hundred metres below its grid cell (common in Alpine
  valleys) samples a different boundary layer than the model column does, which
  matters for exactly the near-surface, terrain-forced variability this
  experiment is about.

Run:
    uv run python workflow/scripts/station_grid_map.py
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial import cKDTree

LOG = logging.getLogger("station_grid_map")

REAL_ZARR = "/store_new/mch/msopr/ml/datasets/mch-realch1-fdb-1km-2005-2025-1h-pl13-v1.0.zarr"
G = 9.80665


def to_xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path,
                    default=Path("output/station_selection/station_sample.csv"))
    ap.add_argument("--out", type=Path, default=Path("resources/smn_grid_points.csv"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    import pandas as pd  # some station names contain commas, so parse properly

    # station_sample.csv is the single source: it carries nat_abbr as well as the
    # coordinates, so station_sample_abbr.txt is not needed here.
    rows = pd.read_csv(args.csv)
    abbr = rows["nat_abbr"].to_numpy()
    slat, slon = rows["latitude"].to_numpy(float), rows["longitude"].to_numpy(float)
    selev = rows["elevation"].to_numpy(float)
    LOG.info("%d stations from %s", len(abbr), args.csv)

    z = zarr.open(REAL_ZARR, mode="r")
    glat, glon = z["latitudes"][:], z["longitudes"][:]
    dist_chord, idx = cKDTree(to_xyz(glat, glon)).query(to_xyz(slat, slon))
    # chord length on the unit sphere -> metres
    dist_m = 2 * 6371000.0 * np.arcsin(np.clip(dist_chord / 2, 0, 1))

    # model elevation from surface geopotential at one timestep (a constant field)
    fis_i = list(z.attrs["variables"]).index("FIS")
    gelev = z["data"].oindex[0, [fis_i], 0, :][0][idx] / G

    dz = selev - gelev
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "nat_abbr": abbr, "station_id": rows["station_id"].to_numpy(),
        "lat": slat, "lon": slon, "elevation": selev,
        "grid_index": idx, "grid_lat": glat[idx], "grid_lon": glon[idx],
        "grid_elevation": gelev.round(1),
        "distance_m": dist_m.round(1), "elevation_diff": dz.round(1),
    }).to_csv(args.out, index=False)

    uniq = len(np.unique(idx))
    print(f"\n=== {len(abbr)} stations -> {uniq} distinct grid points "
          f"({len(abbr) - uniq} collisions) ===")
    print(f"distance station->grid point: median {np.median(dist_m):.0f} m, "
          f"max {dist_m.max():.0f} m ({abbr[np.argmax(dist_m)]})")
    print(f"elevation diff (station - model): median {np.median(dz):+.0f} m, "
          f"IQR {np.percentile(dz, 25):+.0f} to {np.percentile(dz, 75):+.0f} m")
    for lbl, sel in (("station >300 m BELOW model", dz < -300),
                     ("station >300 m ABOVE model", dz > 300)):
        if sel.any():
            print(f"  {lbl} ({sel.sum()}): "
                  + ", ".join(f"{a}({d:+.0f})" for a, d in zip(abbr[sel], dz[sel])))
    if dist_m.max() > 2000:
        far = dist_m > 2000
        print(f"  WARNING possibly outside domain: {', '.join(abbr[far])}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
