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


def get_input(root: Path) -> list[Path]:
    """Get list of tarfiles or directories in root directory."""
    input_files = sorted(root.glob("*.tar"))
    if not input_files:
        gribfiles = sorted(root.glob("*_*/grib/i?eff00000000_000"))
        input_files = [f.parent.parent for f in gribfiles]
    if not input_files:
        raise ValueError(f"No files found in {root}.")
    return input_files


def get_reftime(file: Path) -> datetime:
    if ".tar" in file.suffixes:
        return reftime_from_tarfile(file)
    else:
        return reftime_from_directory(file)


def reftime_from_directory(directory: Path) -> datetime:
    """Extract reftime from directory name."""
    dir_stem = directory.name.rsplit("_", 1)[0]
    return datetime.strptime(dir_stem, "%y%m%d%H")


def reftime_from_tarfile(tarfile: Path, suffix: str | None = None) -> datetime:
    """Extract reftime from tarfile name."""
    suffix = tarfile.stem[-4:] if suffix is None else suffix
    return datetime.strptime(tarfile.stem.removesuffix(suffix), "%y%m%d%H")


def check_reftime_consistency(input: list[Path], delta_h: int = 12):
    """Check that all reftimes are available and every delta_h hours."""

    # note the lower case y in the format string, it's for 2-digit years

    first_reftime = get_reftime(input[0])
    expected_reftime = first_reftime
    for file in input:
        reftime = get_reftime(file)
        if reftime != expected_reftime:
            raise ValueError(f"Expected reftime {expected_reftime} but got {reftime}.")
        expected_reftime += timedelta(hours=delta_h)
    return first_reftime, expected_reftime - timedelta(hours=delta_h)


def extract(
    file: Path, lead_times: list[int], run_id: str, params: list[str]
) -> xr.Dataset:
    LOG.info(f"Extracting fields from {file}.")
    reftime = reftime_from_tarfile(file)
    if "COSMO-E" in file.parts:
        gribname = "ceffsurf"
    elif "COSMO-1E" in file.parts:
        gribname = "c1effsurf"
    elif "ICON-CH1-EPS" in file.parts:
        gribname = "i1eff"
    elif "ICON-CH2-EPS" in file.parts:
        gribname = "i2eff"
    else:
        raise ValueError("Currently only COSMO-E/1E and ICON-CH1/2-EPS are supported.")
    out = ekd.SimpleFieldList()
    if ".tar" in file.suffixes:
        tar_archive = tarfile.open(file, mode="r:*")
        for lt in lead_times:
            filename = f"{file.stem}/grib/{gribname}{lt:03}_{run_id}"
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
    else:
        for lt in lead_times:
            lh = lt % 24
            ld = lt // 24
            filepath = file / "grib" / f"{gribname}{ld:02}{lh:02}0000_{run_id}"
            LOG.info(f"Extracting {filepath}.")
            fields = ekd.from_source("file", filepath)
            for field in fields:
                shortname = field.metadata("shortName")
                if shortname in params:
                    out.append(field)

    out = out.to_xarray(profile="grib")
    out = out.expand_dims(
        forecast_reference_time=[np.array(reftime, dtype="datetime64[ns]")], axis=0
    )

    return out


class ScriptConfig(Namespace):
    archive_dir: Path
    output_store: Path
    steps: list[int]
    run_id: str
    params: list[str]


def _parse_steps(steps: str) -> int:
    # check that steps is in the format "start/stop/step"
    if "/" not in steps:
        raise ValueError(f"Expected steps in format 'start/stop/step', got '{steps}'")
    if len(steps.split("/")) != 3:
        raise ValueError(f"Expected steps in format 'start/stop/step', got '{steps}'")
    start, end, step = map(int, steps.split("/"))
    return list(range(start, end + 1, step))


def main(cfg: ScriptConfig):
    input = get_input(cfg.archive_dir)
    delta_h = 12
    if "COSMO-1E" in input[0].parts or "ICON-CH1-EPS" in input[0].parts:
        delta_h = 3
    if "ICON-CH2-EPS" in input[0].parts:
        delta_h = 6
    first_reftime, last_reftime = check_reftime_consistency(input, delta_h)
    LOG.info(f"Found {len(input)} forecasts from {first_reftime} to {last_reftime}.")

    reftimes = np.array([get_reftime(f) for f in input], dtype="datetime64")
    missing = reftimes
    if not cfg.overwrite:  # only check dataset when we want to append as this is slow
        existing_reftimes = np.array([])
        data_vars = []
        try:
            with xr.open_dataset(cfg.output_store) as ds:
                existing_reftimes = ds.forecast_reference_time
                data_vars = ds.data_vars
        except FileNotFoundError:
            LOG.info("Dataset doesn't exist yet.")

        if (
            existing_reftimes.size > 0
            and reftimes[0] == existing_reftimes[0]
            and set(cfg.params) == set(data_vars)
        ):
            missing = np.setdiff1d(reftimes, existing_reftimes)
            LOG.info("Dataset already exists, missing reftimes will be appended")
            LOG.info(
                f"{existing_reftimes.size} reftimes of {reftimes.size} already ingested."
            )

    _, indices, _ = np.intersect1d(reftimes, missing, return_indices=True)

    for i in indices:
        file = input[i]
        ds = extract(file, cfg.steps, cfg.run_id, cfg.params)

        LOG.info(f"Extracted: {ds}")

        # remove GRIB message from attrs (not serializable)
        for v in ds.data_vars:
            ds[v].attrs.pop("_earthkit")

        # Write to zarr, appending if store exists
        ds = ds.chunk({"forecast_reference_time": 1})
        zarr_encoding = {
            "forecast_reference_time": {"units": "nanoseconds since 1970-01-01"}
        }
        cfg.output_store.parent.mkdir(parents=True, exist_ok=True)
        if i == 0:
            LOG.info(f"Creating new zarr store at {cfg.output_store}.")
            ds.to_zarr(cfg.output_store, mode="w", encoding=zarr_encoding)
        else:
            LOG.info(f"Appending to existing zarr store at {cfg.output_store}.")
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

    parser.add_argument("--steps", type=_parse_steps, default="0/120/6")

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

To submit as a batch job on compute nodes
sbatch --wrap "uv run python ..."

python workflow/scripts/extract_baseline.py \
    --archive_dir /archive/mch/msopr/osm/COSMO-E/FCST20 \
    --output_store /store_new/mch/msopr/ml/COSMO-E/FCST20.zarr \
    --steps 0/120/6

python workflow/scripts/extract_baseline.py \
    --archive_dir /archive/mch/s83/osm/from_GPFS/COSMO-1E/FCST20 \
    --output_store /store_new/mch/msopr/ml/COSMO-1E/FCST20.zarr \
    --steps 0/33/1

python workflow/scripts/extract_baseline.py \
    --archive_dir /store_new/mch/msopr/osm/ICON-CH1-EPS/FCST24 \
    --output_store /store_new/mch/msopr/ml/ICON-CH1-EPS/FCST24.zarr \
    --steps 0/33/1

python workflow/scripts/extract_baseline.py \
    --archive_dir /store_new/mch/msopr/osm/ICON-CH1-EPS/FCST25 \
    --output_store /store_new/mch/msopr/ml/ICON-CH1-EPS/FCST25.zarr \
    --steps 0/33/1
"""
