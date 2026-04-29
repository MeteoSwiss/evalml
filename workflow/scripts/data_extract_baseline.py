from argparse import ArgumentParser, Namespace
from datetime import datetime, timedelta, timezone
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

from data_input import parse_steps  # noqa: E402

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


def get_run_ids(file: Path, gribname: str) -> list[str]:
    """Discover all ensemble member run_ids available for a given file."""
    if ".tar" in file.suffixes:
        with tarfile.open(file, mode="r:*") as tar:
            prefix = f"{file.stem}/grib/{gribname}000_"
            return sorted(
                rid
                for m in tar.getmembers()
                if m.name.startswith(prefix)
                and (rid := m.name.removeprefix(prefix)).isdigit()
                and len(rid) == 3
            )
    else:
        pattern = f"{gribname}00000000_???"
        return sorted(p.name.rsplit("_", 1)[1] for p in (file / "grib").glob(pattern))


def _extract_member(
    file: Path, lead_times: list[int], run_id: str, params: list[str], gribname: str
) -> xr.Dataset:
    """Extract all lead times for a single ensemble member."""
    out = ekd.SimpleFieldList()
    if ".tar" in file.suffixes:
        tar_archive = tarfile.open(file, mode="r:*")
        for lt in lead_times:
            filename = f"{file.stem}/grib/{gribname}{lt:03}_{run_id}"
            LOG.info(f"Extracting {filename}.")
            stream = tar_archive.extractfile(filename)
            streamfieldlist: StreamFieldList = ekd.from_source("stream", stream)
            for field in streamfieldlist:
                if field.metadata("shortName") in params:
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
                if field.metadata("shortName") in params:
                    out.append(field)
    return out.to_xarray(profile="grib")


def extract(
    file: Path,
    lead_times: list[int],
    params: list[str],
    run_id: str | None = None,
    ensemble_mean: bool = False,
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

    if not ensemble_mean and run_id is None:
        raise ValueError("run_id must be provided when ensemble_mean=False.")

    if ensemble_mean:
        run_ids = get_run_ids(file, gribname)
        LOG.info(f"Computing ensemble mean over {len(run_ids)} members: {run_ids}")
        acc = None
        loaded = []
        for rid in run_ids:
            try:
                member = _extract_member(file, lead_times, rid, params, gribname)
                acc = member if acc is None else acc + member
                loaded.append(rid)
            except Exception as e:
                LOG.warning(f"Skipping member {rid}: {e}")
        if acc is None:
            raise ValueError(f"No ensemble members could be loaded from {file}.")
        out = acc / len(loaded)
        LOG.info(f"Ensemble mean computed over {len(loaded)} members: {loaded}")
        out.attrs["ensemble"] = f"mean over members {', '.join(loaded)}"
    else:
        out = _extract_member(file, lead_times, run_id, params, gribname)
        out.attrs["ensemble"] = f"member {run_id}"

    out.attrs["institution"] = "MeteoSwiss"
    out.attrs["extracted"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
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
    ensemble_mean: bool


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
        ds = extract(
            file, cfg.steps, cfg.params, run_id=cfg.run_id, ensemble_mean=cfg.ensemble_mean
        )

        LOG.info(f"Extracted: {ds}")

        # remove GRIB message from attrs (not serializable)
        for v in ds.data_vars:
            ds[v].attrs.pop("_earthkit", None)

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

    parser.add_argument("--steps", type=parse_steps, default="0/120/6")

    parser.add_argument("--run_id", type=str, default="000")

    parser.add_argument(
        "--ensemble_mean",
        action="store_true",
        default=False,
        help="Compute mean over all ensemble members. When set, --run_id is ignored.",
    )

    parser.add_argument(
        "--params",
        type=lambda x: x.split(","),
        default=[
            "T_2M", "TD_2M", "U_10M", "V_10M", "PS", "PMSL", "TOT_PREC",
            "CLCT", "CLCL", "CLCM", "CLCH",
            "ASWDIFD_S", "ASWDIR_S",
        ],
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

python workflow/scripts/data_extract_baseline.py \
    --archive_dir /store_new/mch/msopr/osm/ICON-CH1-EPS/FCST24 \
    --output_store /store_new/mch/msopr/ml/ICON-CH1-CTRL/FCST24.zarr \
    --steps 0/33/1

python workflow/scripts/data_extract_baseline.py \
    --archive_dir /store_new/mch/msopr/osm/ICON-CH2-EPS/FCST25 \
    --output_store /store_new/mch/msopr/ml/ICON-CH2-CTRL/FCST25.zarr \
    --steps 0/121/1

python workflow/scripts/data_extract_baseline.py \
    --archive_dir /store_new/mch/msopr/osm/ICON-CH1-EPS/FCST25 \
    --output_store /store_new/mch/msopr/ml/ICON-CH1-MEAN/FCST25.zarr \
    --steps 0/33/1 \
    --ensemble_mean
"""
