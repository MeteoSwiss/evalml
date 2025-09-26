from pathlib import Path
from argparse import ArgumentParser, Namespace
import logging
import sys
import os
from datetime import datetime
from typing import Iterable

eccodes_definition_path = Path(sys.prefix) / "share/eccodes-cosmo-resources/definitions"
os.environ["ECCODES_DEFINITION_PATH"] = str(eccodes_definition_path)

from meteodatalab import data_source, grib_decoder  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from src.verification import verify  # noqa: E402

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def load_analysis_data_from_zarr(
    analysis_zarr: Path, times: Iterable[datetime], params: list[str]
) -> xr.Dataset:
    """Load analysis data from an anemoi-generated Zarr dataset

    This function loads analysis data from a Zarr dataset, processing it to make it more
    xarray-friendly. It renames variables, sets the time index, and pivots the dataset.
    """
    PARAMS_MAP_COSMO2 = {
        "T_2M": "2t",
        "TD_2M": "2d",
        "U_10M": "10u",
        "V_10M": "10v",
        "PS": "sp",
        "PMSL": "msl",
        "TOT_PREC": "tp",
        "SKT": "skt",
        "T_G": "skt",  # This needs to come after SKT (takes precedence)
        "T": "t",
        "U": "u",
        "V": "v",
        "QV": "q",
        "FI": "z",
    }
    PARAMS_MAP_COSMO1 = {v: v.replace("TOT_PREC", "TOT_PREC_6H") for v in PARAMS_MAP_COSMO2.keys()}
    PARAMS_MAP = PARAMS_MAP_COSMO2 if "co2" in analysis_zarr.name else PARAMS_MAP_COSMO1

    ds = xr.open_zarr(analysis_zarr, consolidated=False)

    # rename "dates" to "time" and set it as index
    ds = ds.set_index(time="dates")

    # set 'variables' attr as dimension coordinate
    ds = ds.assign_coords({"variable": ds.attrs["variables"]})

    # select variables and valid time, squeeze ensemble dimension
    inverse_mapping = {v:k for k,v in PARAMS_MAP.items()}
    prefixes = tuple(PARAMS_MAP.values())
    vars_to_select = [v for v in ds["variable"].values if any(str(v).startswith(p) for p in prefixes) and inverse_mapping[v] in params]
    ds = ds.sel(variable=vars_to_select).squeeze("ensemble", drop=True)

    # recover original 2D shape
    if len(ds.attrs["field_shape"]) == 2:
        ny, nx = ds.attrs["field_shape"]
        y_idx, x_idx = np.unravel_index(np.arange(ny * nx), shape=(ny, nx))
        ds = ds.assign_coords({"y": ("cell", y_idx), "x": ("cell", x_idx)})
        ds = ds.set_index(cell=("y", "x"))
        ds = ds.unstack("cell")

    # set lat lon as coords (optional)
    if "latitudes" in ds and "longitudes" in ds:
        ds = ds.rename({"latitudes": "latitude", "longitudes": "longitude"})
    ds = ds.set_coords(["latitude", "longitude"])

    # pivot (use inverse of PARAMS_MAP)
    ds = (
        ds["data"]
        .to_dataset("variable")
        .rename({v: k for k, v in PARAMS_MAP.items() if v in ds["variable"].values})
    )

    # select valid times
    # (handle special case where some valid times are not in the dataset, e.g. at the end)
    times_included = times.isin(ds.time.values).values.ravel()
    print("TIMES", times, flush=True)
    if all(times_included):
        ds = ds.sel(time=times)

    elif 0 < np.sum(times_included) < len(times_included):
        LOG.warning(
            "Some valid times are not included in the dataset: \n%s",
            times[~times_included].values,
        )
        ds = ds.sel(time=times[times_included])
    else:
        raise ValueError(
            "Valid times are not included in the dataset. "
            "Please check the valid times and the dataset."
        )

    return ds


def load_fct_data_from_grib(
    grib_output_dir: Path, params: list[str], step: list[int]
) -> xr.Dataset:
    """Load forecast data from GRIB files for a specific valid time."""
    files = sorted(grib_output_dir.glob("20*.grib"))
    fds = data_source.FileDataSource(datafiles=files)
    ds = grib_decoder.load(fds, {"param": params, "step": step})
    for var, da in ds.items():
        if "z" in da.dims and da.sizes["z"] == 1:
            ds[var] = da.squeeze("z", drop=True)
        elif "z" in da.dims and da.sizes["z"] > 1:
            ds[var] = da.rename({"z": da.attrs["vcoord_type"]})
    ds = xr.merge([ds[p].rename(p) for p in ds], compat="no_conflicts")
    if "TOT_PREC" in ds.data_vars:
        LOG.info("Disaggregating precipitation")
        ds = ds.assign(
            TOT_PREC=lambda x: (
                x.TOT_PREC.fillna(0)
                .diff("lead_time")
                .pad(lead_time=(1, 0), constant_value=None)
                .clip(min=0.0)
            )
        )
    # make sure time coordinate is available, and valid_time is not
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    if "time" not in ds.coords:
        ds = ds.assign_coords(time=ds.ref_time + ds.lead_time)

    return ds


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


class ScriptConfig(Namespace):
    """Configuration for the script to verify forecast data."""

    archive_root: Path = None
    analysis_zarr: Path = None
    forecast_zarr: Path = None
    params: list[str]
    lead_time: list[int] = _parse_lead_time("0/126/6")


def program_summary_log(args):
    """Log a welcome message with the script information."""
    LOG.info("=" * 80)
    LOG.info("Running verification of ML model forecast data")
    LOG.info("=" * 80)
    LOG.info("GRIB output directory: %s", args.grib_output_dir)
    if args.analysis_zarr:
        LOG.info("Zarr dataset: %s", args.analysis_zarr)
    else:
        LOG.info("Archive root: %s", args.archive_root)
    LOG.info("Parameters: %s", args.params)
    LOG.info("Lead time: %s", args.lead_time)
    LOG.info("Output file: %s", args.output)
    LOG.info("=" * 80)


def main(args: ScriptConfig):
    """Main function to verify forecast data."""

    # get forecast data
    start = datetime.now()
    fct = load_fct_data_from_grib(
        grib_output_dir=args.grib_output_dir, params=args.params, step=args.lead_time
    )
    LOG.info(
        "Loaded forecast data from GRIB files in %.2f seconds: \n%s",
        (datetime.now() - start).total_seconds(),
        fct,
    )
    print("FCT TIMES", fct.time.values, flush=True)
    # get truth data (aka analysis)
    start = datetime.now()
    if args.analysis_zarr:
        analysis = (
            load_analysis_data_from_zarr(
                analysis_zarr=args.analysis_zarr,
                times=fct.time,
                params=args.params,
            )
            .compute()
            .chunk({"y": -1, "x": -1})
        )
    else:
        raise ValueError("--analysis_zarr must be provided.")
    LOG.info(
        "Loaded analysis data from zarr dataset in %.2f seconds: \n%s",
        (datetime.now() - start).total_seconds(),
        analysis,
    )

    # compute metrics and statistics
    results = verify(fct, analysis, args.fcst_label, args.analysis_label)

    # # save results to CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_netcdf(args.output)
    LOG.info("Saved results to %s", args.output)
    LOG.info("Verification completed successfully.")


if __name__ == "__main__":
    # TODO: temporarily we must specify all params because it's currently required in the
    # request with meteodatalab.grib_decoder.load. This will be fixed in a future version.
    PARAMS = ["T_2M", "TD_2M", "U_10M", "V_10M", "PS", "PMSL", "TOT_PREC", "T_G"]
    # PARAMS += ["T", "U", "V","QV","FI"]

    parser = ArgumentParser(description="Verify forecast data.")

    parser.add_argument(
        "--grib_output_dir",
        type=Path,
        required=True,
        help="Path to the Zarr dataset containing forecast data.",
    )
    parser.add_argument(
        "--analysis_zarr",
        type=Path,
        required=True,
        help="Path to the Zarr dataset containing analysis data.",
    )
    parser.add_argument(
        "--params",
        type=lambda x: x.split(","),
        required=False,
        default=PARAMS,
        help="Comma-separated list of parameters to verify.",
    )
    parser.add_argument(
        "--lead_time",
        type=_parse_lead_time,
        default="0/126/6",
        help="Lead time in the format 'start/stop/step'.",
    )
    parser.add_argument(
        "--fcst_label",
        type=str,
        help="Label for the forecast data.",
    )
    parser.add_argument(
        "--analysis_label",
        type=str,
        help="Label for the analysis data (default: COSMO KENDA).",
        default="COSMO KENDA",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="verif.nc",
        help="Output file to save the verification results.",
    )
    args = parser.parse_args()

    main(args)

# run examples
# uv run workflow/scripts/verif_from_grib.py \
#     --analysis_zarr /scratch/mch/fzanetta/data/anemoi/datasets/mch-co2-an-archive-0p02-2015-2020-6h-v3-pl13.zarr \
#     --reftime 202006011200 \
#     --output debug_verif_zarr.csv
# uv run workflow/scripts/verif_from_grib.py \
#     --grib_output_dir /users/fzanetta/projects/mch-anemoi-evaluation/output/7c58e59d24e949c9ade3df635bbd37e2/202001050600/grib \
#     --analysis_zarr /scratch/mch/fzanetta/data/anemoi/datasets/mch-co2-an-archive-0p02-2015-2020-6h-v3-pl13.zarr \
#     --reftime 202006011200 \
#     --output debug_verif_grib.csv
