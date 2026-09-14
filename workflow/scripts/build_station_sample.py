"""Build the SMN station sample used for the temporal spectra of exp_plan_01.

The sample is the set of SwissMetNet stations (DWH station groups 1 and 2) that
report all four variables the spectral analysis needs: T_2M, TD_2M and the two
10m wind components. Membership is determined from *data* queries, because the
DWH ignores the station-group filter on ``--meta-info`` queries (verified: a
nonexistent group id returns the full catalog from the metadata endpoint and
zero rows from the data endpoint).

Inputs are the CSVs already retrieved into ``output/station_selection/``:
``snap_<termin>_<param>.csv`` for membership, ``meta_core_params.csv`` for
coordinates. Coordinates are resolved with ``StationCatalog.from_meta`` so the
sample matches how the rest of evalml locates stations.

NOTE: membership is a two-timestamp snapshot, not a completeness check over the
2024 forecast windows. Screening for gaps is a separate, larger retrieval.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.data_input.jretrieve import StationCatalog  # noqa: E402

OUT = Path("output/station_selection")
PARAMS = ("tre200s0", "tde200s0", "fkl010z0", "dkl010z0")
TERMINS = ("202402150600", "202406150600")


def reporting(param: str) -> set[int]:
    """Stations that delivered a value for ``param`` at either sample time."""
    s: set[int] = set()
    for t in TERMINS:
        d = pd.read_csv(OUT / f"snap_{t}_{param}.csv", sep=";")
        s |= set(d.loc[d[param].notna(), "station"])
    return s


def main() -> None:
    sample = set.intersection(*(reporting(p) for p in PARAMS))

    meta = pd.read_csv(OUT / "meta_core_params.csv", sep=";")
    cat = StationCatalog.from_meta(meta[meta["station"].isin(sample)])

    df = pd.DataFrame(
        {
            "nat_abbr": cat.nat_abbr,
            "station_id": cat.station_id,
            "name": cat.name,
            "latitude": cat.latitude,
            "longitude": cat.longitude,
            "elevation": cat.elevation,
        }
    )
    missing = sample - set(cat.station_id)
    if missing:
        print(f"WARNING: {len(missing)} stations absent from metadata: {sorted(missing)}")

    df.to_csv(OUT / "station_sample.csv", index=False)
    (OUT / "station_sample_abbr.txt").write_text(",".join(df["nat_abbr"]) + "\n")

    e = df["elevation"]
    print(f"{len(df)} stations written to {OUT/'station_sample.csv'}")
    print(
        f"elevation: min {e.min():.0f}  p25 {e.quantile(.25):.0f}  "
        f"median {e.median():.0f}  p75 {e.quantile(.75):.0f}  max {e.max():.0f}"
    )
    for lo, hi in ((0, 600), (600, 1000), (1000, 1500), (1500, 2000), (2000, 5000)):
        print(f"  {lo:>4}-{hi:<4} m: {((e >= lo) & (e < hi)).sum():>3}")
    print(
        f"lat {df.latitude.min():.2f}-{df.latitude.max():.2f}  "
        f"lon {df.longitude.min():.2f}-{df.longitude.max():.2f}"
    )


if __name__ == "__main__":
    main()
