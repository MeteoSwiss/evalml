"""Fetch SwissMetNet hourly observations for the spectral analysis.

Run by Louis, not by Claude: it touches the DWH credentials.

Fetches only the periods the 2024 forecast sample actually covers, in three
contiguous blocks, rather than the whole year:

    winter-a  2024-01-01 .. 2024-03-05   (Jan/Feb inits + 120 h)
    winter-b  2024-12-01 .. 2025-01-05   (Dec inits + 120 h)
    summer    2024-06-02 .. 2024-09-05   (JJA inits + 120 h)

Station set: groups 1,2 (SwissMetNet). The group filter is honoured on data
queries but silently ignored on --meta-info queries, so membership is taken from
resources/smn_grid_points.csv (the 145-station sample) and the returned rows are
filtered to it afterwards.

Parameters are the four that define the 145-station sample. Wind is fetched as
speed and direction, which is what SMN measures; U and V are derived later.

    tre200s0  T_2M        tde200s0  TD_2M
    fkl010z0  wind speed  dkl010z0  wind direction

Output: one CSV per block under output/obs/, plus a combined tidy CSV.

    uv run workflow/scripts/fetch_obs.py
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("fetch_obs")

BLOCKS = {
    "winter-a": (datetime(2024, 1, 1), datetime(2024, 3, 5)),
    "winter-b": (datetime(2024, 12, 1), datetime(2025, 1, 5)),
    "summer": (datetime(2024, 6, 2), datetime(2024, 9, 5)),
}
PARAMS = ["tre200s0", "tde200s0", "fkl010z0", "dkl010z0"]
CHUNK_DAYS = 5  # one DWH request per chunk; whole blocks exceed the 600 s timeout
OUT = Path("output/obs")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    # Load jretrieve.py directly by path. Importing it as data_input.jretrieve
    # would run the package __init__, which pulls in earthkit.data and the whole
    # GRIB stack; this script needs neither, only numpy and pandas.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "jretrieve", Path("src/data_input/jretrieve.py"))
    jr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jr)

    jr.check_prerequisites("prod")

    want = set(pd.read_csv("resources/smn_grid_points.csv")["station_id"])  # DWH ids

    # One request for a whole block (65 days x ~190 stations x 4 params) exceeds
    # the DWH's 600 s timeout. Chunk it: each chunk is its own file, so a failure
    # costs one chunk and a rerun resumes.
    frames = []
    for name, (start, end) in BLOCKS.items():
        edges = pd.date_range(start, end, freq=f"{CHUNK_DAYS}D").to_pydatetime().tolist()
        if edges[-1] < end:
            edges.append(end)
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            f = OUT / f"obs_{name}_{i:02d}.csv"
            if f.exists():
                frames.append(pd.read_csv(f))
                continue
            LOG.info("fetching %s chunk %d: %s .. %s", name, i, a, b)
            df = jr.fetch_data(stations={"group": "1,2"}, params=PARAMS,
                               start=a, end=b, increment_minutes=60)
            df.to_csv(f, index=False)
            LOG.info("  wrote %s (%d rows)", f, len(df))
            frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    LOG.info("combined: %d rows, %d stations", len(all_df), all_df["station"].nunique())
    all_df.to_csv(OUT / "obs_all.csv", index=False)
    print(f"\nwrote {OUT}/obs_all.csv  ({len(all_df)} rows)")
    print(f"stations returned: {all_df['station'].nunique()}, "
          f"sample wants {len(want)} (matched later by station_id)")


if __name__ == "__main__":
    main()
