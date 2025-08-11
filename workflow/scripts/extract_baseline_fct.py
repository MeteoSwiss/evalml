from argparse import ArgumentParser, Namespace
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import sys
import tarfile

definition_path = Path(sys.prefix) / "share/eccodes-cosmo-resources/definitions"
os.environ["ECCODES_DEFINITION_PATH"] = str(definition_path)

import earthkit.data as ekd  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from earthkit.data.sources.stream import StreamFieldList  # noqa: E402

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def reftime_from_tarfile(tarfile: Path, suffix: str | None = None) -> datetime:
    """Extract reftime from tarfile name."""
    suffix = tarfile.stem[-4:] if suffix is None else suffix
    return datetime.strptime(tarfile.stem.removesuffix(suffix), "%y%m%d%H")


def check_reftime_consistency(tarfiles: list[Path], delta_h: int = 12):
    """Check that all reftimes are available and every delta_h hours."""

    # note the lower case y in the format string, it's for 2-digit years

    first_reftime = reftime_from_tarfile(tarfiles[0])
    expected_reftime = first_reftime
    for file in tarfiles:
        reftime = reftime_from_tarfile(file)
        if reftime != expected_reftime:
            raise ValueError(f"Expected reftime {expected_reftime} but got {reftime}.")
        expected_reftime += timedelta(hours=delta_h)
    return first_reftime, expected_reftime - timedelta(hours=delta_h)


def extract(
    tar: Path, lead_time: list[int], run_id: str, params: list[str]
) -> xr.Dataset:
    LOG.info(f"Extracting fields from {tar}.")
    reftime = reftime_from_tarfile(tar)
    if "COSMO-E" in tar.parts:
        gribname = "ceffsurf"
    elif "COSMO-1E" in tar.parts:
        gribname = "c1effsurf"
    else:
        raise ValueError("Currently only COSMO-E and COSMO-1E are supported.")
    tar_archive = tarfile.open(tar, mode="r:*")
    out = ekd.SimpleFieldList()
    for lt in lead_time:
        filename = f"{tar.stem}/grib/{gribname}{lt:03}_{run_id}"
        LOG.info(f"Extracting {filename}.")
        stream = tar_archive.extractfile(filename)

        # LOG.info(f"Reading fields...")
        streamfieldlist: StreamFieldList = ekd.from_source("stream", stream)
        for field in streamfieldlist:
            shortname = field.metadata("shortName")
            if shortname in params:
                out.append(field)
        stream.close()
    tar_archive.close()

    out = out.to_xarray(profile="grib")
    out = out.expand_dims(
        forecast_reference_time=[np.array(reftime, dtype="datetime64[ns]")], axis=0
    )

    return out


class ScriptConfig(Namespace):
    archive_dir: Path
    output_store: Path
    lead_time: int
    run_id: str
    params: list[str]


def _parse_lead_time(lead_time: str) -> int:
    # check that lead_time is in the format "start/stop/step"
    if "/" not in lead_time:
        raise ValueError(
            f"Expected lead_time in format 'start/stop/step', got '{lead_time}'"
        )
    if len(lead_time.split("/")) != 3:
        raise ValueError(
            f"Expected lead_time in format 'start/stop/step', got '{lead_time}'"
        )

    return list(range(*map(int, lead_time.split("/"))))


def main(cfg: ScriptConfig):
    tarfiles = sorted(cfg.archive_dir.glob("*.tar"))
    delta_h = 12
    if "COSMO-1E" in tarfiles[0].parts:
        delta_h = 3
    first_reftime, last_reftime = check_reftime_consistency(tarfiles, delta_h)
    LOG.info(
        f"Found {len(tarfiles)} tar archives from {first_reftime} to {last_reftime}."
    )

    reftimes = np.array([reftime_from_tarfile(f) for f in tarfiles], dtype="datetime64")
    missing = reftimes
    if not cfg.overwrite:  # only check dataset when we want to append as this is slow
        existing_reftimes = np.array([])
        try:
            with xr.open_dataset(cfg.output_store) as ds:
                existing_reftimes = ds.forecast_reference_time
        except FileNotFoundError:
            LOG.info("Dataset doesn't exist yet.")

        if existing_reftimes.size > 0 and reftimes[0] == existing_reftimes[0]:
            missing = np.setdiff1d(reftimes, existing_reftimes)
            LOG.info("Dataset already exists, missing reftimes will be appended")
            LOG.info(
                f"{existing_reftimes.size} reftimes of {reftimes.size} already ingested."
            )

    _, indices, _ = np.intersect1d(reftimes, missing, return_indices=True)

    for i in indices:
        file = tarfiles[i]
        ds = extract(file, cfg.lead_time, cfg.run_id, cfg.params)

        LOG.info(f"Extracted: {ds}")

        # remove GRIB message from attrs (not serializable)
        for v in ds.data_vars:
            ds[v].attrs.pop("_earthkit")

        # Write to zarr, appending if store exists
        ds = ds.chunk({"forecast_reference_time": 1})
        zarr_encoding = {
            "forecast_reference_time": {"units": "nanoseconds since 1970-01-01"}
        }
        if i == 0:
            ds.to_zarr(cfg.output_store, mode="w", encoding=zarr_encoding)
        else:
            ds.to_zarr(cfg.output_store, mode="a", append_dim="forecast_reference_time")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--archive_dir", type=Path, default="/archive/mch/msopr/osm/COSMO-E/FCST20/"
    )

    parser.add_argument(
        "--output_store",
        type=Path,
        help="Path to the output zarr store.",
    )

    parser.add_argument("--lead_time", type=_parse_lead_time, default="0/126/6")

    parser.add_argument("--run_id", type=str, default="000")

    parser.add_argument(
        "--params",
        type=lambda x: x.split(","),
        default=["T_2M", "TD_2M", "U_10M", "V_10M", "PS", "PMSL", "TOT_PREC"],
    )

    parser.add_argument(
        "--overwrite",
        type=bool,
        default=False,
        help="Should existing dataset be overwritten?",
    )

    args = parser.parse_args()

    main(args)

"""
Example usage:
python workflow/scripts/extract_baseline_fct.py \
    --archive_dir /archive/mch/msopr/osm/COSMO-E/FCST20 \
    --output_store /store_new/mch/msopr/ml/COSMO-E/FCST20.zarr \
    --lead_time 0/126/6

python workflow/scripts/extract_baseline_fct.py \
    --archive_dir /archive/mch/s83/osm/from_GPFS/COSMO-1E/FCST20 \
    --output_store /store_new/mch/msopr/ml/COSMO-1E/FCST20.zarr \
    --lead_time 0/34/1
"""
