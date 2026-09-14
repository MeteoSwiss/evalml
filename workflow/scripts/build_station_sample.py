"""Build the SMN station sample used for the temporal spectra of exp_plan_01.

The sample is the set of SwissMetNet stations (DWH station groups 1 and 2) that
report all four variables the spectral analysis needs: T_2M, TD_2M and the two
10m wind components.

Membership is determined from *data* queries, because the DWH ignores the
station-group filter on ``--meta-info`` queries (verified: a nonexistent group
id returns the full catalog from the metadata endpoint and zero rows from the
data endpoint). Coordinates come from a metadata query resolved through
``StationCatalog.from_meta``, so the sample matches how the rest of evalml
locates stations.

Stations are probed at ``N_TERMINS`` timestamps drawn at random from the two
2024 sample periods of ``exp_plan_01.md`` (winter: Jan, Feb, Dec; summer: JJA),
with a fixed seed so the selection is reproducible. Two criteria are reported:

  union  -- reports at *any* probed timestamp (permissive; a station counts as
            present even if mostly down)
  strict -- reports at *every* probed timestamp (a coarse availability screen)

The sample written to disk uses the union criterion (``CRITERION``). A probe at
ten isolated hours cannot separate a station having one bad day from one with
systematic gaps, so it is not used to screen availability; that is left to a
proper gap analysis over the contiguous 48h windows the spectra need.

Retrieval is skipped for snapshots already on disk, so re-running is cheap.
Requires DWH credentials (a ``.env`` in the project root) and jretrievedwh.py.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from src.data_input.jretrieve import StationCatalog, _resolve_binary  # noqa: E402

OUT = ROOT / "output" / "station_selection"
GROUPS = "stn_group_id,1,2"  # SwissMetNet
SEED = 20240914
N_TERMINS = 10
CRITERION = "union"  # "union" or "strict"; see module docstring

# Variables defining the sample, and the wider set reported for context.
SAMPLE_PARAMS = ("tre200s0", "tde200s0", "fkl010z0", "dkl010z0")
CONTEXT_PARAMS = ("pp0qffs0", "prestas0", "rre150h0", "rre006i0")
LABELS = {
    "tre200s0": "T_2M",
    "tde200s0": "TD_2M",
    "fkl010z0": "wind speed 10m",
    "dkl010z0": "wind dir 10m",
    "pp0qffs0": "PMSL",
    "prestas0": "PS",
    "rre150h0": "precip 1h",
    "rre006i0": "precip 6h",
}

# The 2024 sample periods of exp_plan_01.md.
PERIODS = (("2024-01-01", "2024-02-29"), ("2024-12-01", "2024-12-31"),
           ("2024-06-01", "2024-08-31"))


def draw_termins() -> list[str]:
    """Random hourly timestamps from the sample periods, evenly split."""
    rng = np.random.default_rng(SEED)
    hours = np.concatenate(
        [pd.date_range(a, b + " 23:00", freq="h").to_numpy() for a, b in PERIODS]
    )
    picked = rng.choice(hours, size=N_TERMINS, replace=False)
    return sorted(pd.Timestamp(t).strftime("%Y%m%d%H%M") for t in picked)


def fetch(param: str, termin: str) -> Path:
    path = OUT / f"snap_{termin}_{param}.csv"
    if path.exists():
        return path
    argv = [_resolve_binary(), "-s", "surface", "-n", param,
            "-t", f"{termin},{termin}", "--format", "csv",
            "--use-limitation", "40", "-a", GROUPS]
    env = {**__import__("os").environ,
           "JRETRIEVE_CONF_DIR": str(ROOT),
           "JRETRIEVE_CONF_NAME": ".jretrievedwh-conf.prod.py"}
    proc = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0 or proc.stdout.lstrip().startswith("ERROR"):
        raise RuntimeError(f"{param}@{termin}: {proc.stderr.strip()[:200]}"
                           f"{proc.stdout.strip()[:200]}")
    path.write_text(proc.stdout)
    return path


def reporting(param: str, termins: list[str]) -> dict[str, set[int]]:
    """Per-timestamp sets of stations that delivered a value."""
    out = {}
    for t in termins:
        d = pd.read_csv(fetch(param, t), sep=";")
        out[t] = set(d.loc[d[param].notna(), "station"]) if len(d) else set()
    return out


def main() -> None:
    termins = draw_termins()
    print(f"{N_TERMINS} random timestamps (seed {SEED}):")
    for t in termins:
        print(f"  {pd.Timestamp(t).strftime('%Y-%m-%d %H:%M UTC')}")

    per_param = {p: reporting(p, termins)
                 for p in SAMPLE_PARAMS + CONTEXT_PARAMS}

    print(f"\n{'variable':<16} {'union':>6} {'strict':>7}   per-timestamp")
    union, strict = {}, {}
    for p in SAMPLE_PARAMS + CONTEXT_PARAMS:
        sets = list(per_param[p].values())
        union[p] = set.union(*sets)
        strict[p] = set.intersection(*sets)
        counts = " ".join(f"{len(s):>3}" for s in sets)
        print(f"{LABELS[p]:<16} {len(union[p]):>6} {len(strict[p]):>7}   {counts}")

    u = set.intersection(*(union[p] for p in SAMPLE_PARAMS))
    s = set.intersection(*(strict[p] for p in SAMPLE_PARAMS))
    print(f"\nsample, union criterion : {len(u)}")
    print(f"sample, strict criterion: {len(s)}")
    print(f"dropped by strict       : {len(u - s)}")

    sample = {"union": u, "strict": s}[CRITERION]
    print(f"writing the {CRITERION} sample")

    meta = pd.read_csv(OUT / "meta_core_params.csv", sep=";")
    cat = StationCatalog.from_meta(meta[meta["station"].isin(sample)])
    if missing := sample - set(cat.station_id):
        print(f"WARNING: {len(missing)} stations absent from metadata: {sorted(missing)}")

    df = pd.DataFrame({
        "nat_abbr": cat.nat_abbr, "station_id": cat.station_id, "name": cat.name,
        "latitude": cat.latitude, "longitude": cat.longitude,
        "elevation": cat.elevation,
    })
    df.to_csv(OUT / "station_sample.csv", index=False)
    (OUT / "station_sample_abbr.txt").write_text(",".join(df["nat_abbr"]) + "\n")

    e = df["elevation"]
    print(f"\n{len(df)} stations written to {OUT/'station_sample.csv'}")
    print(f"elevation: min {e.min():.0f}  p25 {e.quantile(.25):.0f}  "
          f"median {e.median():.0f}  p75 {e.quantile(.75):.0f}  max {e.max():.0f}")
    for lo, hi in ((0, 600), (600, 1000), (1000, 1500), (1500, 2000), (2000, 5000)):
        print(f"  {lo:>4}-{hi:<4} m: {((e >= lo) & (e < hi)).sum():>3}")
    print(f"lat {df.latitude.min():.2f}-{df.latitude.max():.2f}  "
          f"lon {df.longitude.min():.2f}-{df.longitude.max():.2f}")


if __name__ == "__main__":
    main()
