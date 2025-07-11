from pathlib import Path
from argparse import ArgumentParser, Namespace
import logging
import sys
import os
from datetime import datetime
from typing import Iterable

eccodes_definition_path = Path(sys.prefix) / "share/eccodes-cosmo-resources/definitions"
os.environ["ECCODES_DEFINITION_PATH"] = str(eccodes_definition_path)

from meteodatalab import data_source, grib_decoder
import numpy as np
import xarray as xr

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def load_kenda1_data_from_zarr(zarr_dataset: Path, valid_times: Iterable[datetime], params: list[str]) -> xr.Dataset:
    """Load KENDA-1 data from an anemoi-generated Zarr dataset

    This function loads KENDA-1 data from a Zarr dataset, processing it to make it more 
    xarray-friendly. It renames variables, sets the time index, and pivots the dataset.
    """
    PARAMS_MAP = {
        "T_2M": "2t",
        "TD_2M": "2d",
        "U_10M": "10u",
        "V_10M": "10v",
        "PS": "sp",
        "PMSL": "msl",
        "TOT_PREC": "tp",
        "SKT": "skt",  
        "T_G": "skt", # This needs to come after SKT (takes precedence)
        "T": "t",
        "U": "u",
        "V": "v",
        "QV": "q",
        "FI": "z",
    }

    ds = xr.open_zarr(zarr_dataset, consolidated=False)

    # rename "dates" to "time" and set it as index
    ds = ds.set_index(time="dates")

    # set 'variables' attr as dimension coordinate
    ds = ds.assign_coords({"variable": ds.attrs["variables"]})

    # select variables and valid time, squeeze ensemble dimension
    ds = ds.sel(variable=[PARAMS_MAP[p] for p in params]).squeeze("ensemble", drop=True)

    # recover original 2D shape
    if len(ds.attrs["field_shape"]) == 2:
        ny, nx = ds.attrs["field_shape"]
        y_idx, x_idx = np.unravel_index(np.arange(ny * nx), shape=(ny, nx))
        ds = ds.assign_coords({"y": ("cell", y_idx), "x": ("cell", x_idx)})
        ds = ds.set_index(cell=("y", "x"))
        ds = ds.unstack("cell")

    # set lat lon as coords (optional)
    ds = ds.set_coords(["latitudes", "longitudes"])

    # pivot (use inverse of PARAMS_MAP)
    ds = ds["data"].to_dataset("variable").rename(
        {v: k for k, v in PARAMS_MAP.items() if v in ds["variable"].values}
    )

    # select valid times
    ds = ds.sel(time=valid_times)
    
    return ds

# def load_kenda1_data_from_grib(archive_root: Path, valid_times: datetime, params: list[str]) -> xr.Dataset:
#     """Load KENDA1 data from GRIB files for a specific valid time."""

#     ACCUM_PARAMS = [
#         "TOT_PREC",
#     ]

#     archive_year = archive_root / str(valid_time.year)
#     files = [
#         archive_year / f"laf_sfc_{valid_time:%Y%m%d%H}.grib2", 
#         archive_year / f"lff_sfc_{valid_time:%Y%m%d%H}.grib2"
#     ]
#     accum_file = archive_year / f"lff_sfc_6haccum_{valid_time:%Y%m%d%H}.grib2"

#     # separate parameters into instantaneous and accumulated
#     # TODO: we need a more general way to handle this
#     accum_params = []
#     for param in params:
#         if param in ACCUM_PARAMS:
#             accum_params.append(param)
#             params.remove(param)

#     fds = data_source.FileDataSource(datafiles=files)
#     ds = grib_decoder.load(fds, {"param": params})
#     for var, da in ds.items():
#         if "z" in da.dims and da.sizes["z"] == 1:
#             ds[var] = da.squeeze("z", drop=True)
#         elif "z" in da.dims and da.sizes["z"] > 1:
#             ds[var] = da.rename({"z": da.attrs["vcoord_type"]})

#     ds = xr.Dataset(ds)

#     if accum_params:
#         fds = data_source.FileDataSource(datafiles=[str(accum_file)])
#         for var, da in grib_decoder.load(fds, {"param": accum_params}).items():
#             da = da.drop("z") if "z" in da.dims else da
#             ds[var] = da

#     return ds

def load_fct_data_from_grib(grib_output_dir: Path, params: list[str], step: list[int]) -> xr.Dataset:
    """Load COSMO-E forecast data from GRIB files for a specific valid time."""
    files = sorted(grib_output_dir.glob("*.grib"))
    fds = data_source.FileDataSource(datafiles=files)
    ds = grib_decoder.load(fds, {"param": params, "step": step})
    for var, da in ds.items():
        if "z" in da.dims and da.sizes["z"] == 1:
            ds[var] = da.squeeze("z", drop=True)
        elif "z" in da.dims and da.sizes["z"] > 1:
            ds[var] = da.rename({"z": da.attrs["vcoord_type"]})
    ds = xr.merge([ds[p].rename(p) for p in ds])
    return ds 

def _parse_lead_time(lead_time: str) -> int:
    # check that lead_time is in the format "start/stop/step"
    if "/" not in lead_time:
        raise ValueError(f"Expected lead_time in format 'start/stop/step', got '{lead_time}'")
    if len(lead_time.split("/")) != 3:
        raise ValueError(f"Expected lead_time in format 'start/stop/step', got '{lead_time}'")
    
    return list(range(*map(int, lead_time.split("/"))))


class ScriptConfig(Namespace):
    """Configuration for the script to verify COSMOe forecast data."""
    
    archive_root: Path = None
    zarr_dataset: Path = None
    cosmoe_zarr: Path = Path("/scratch/mch/fzanetta/data/COSMO-E/2020.zarr")
    reftime: datetime = None
    params: list[str]
    lead_time: list[int] = _parse_lead_time("0/126/6")

def main(args: ScriptConfig):
    """Main function to verify COSMOe forecast data."""
    
    
    # get forecast data
    start = datetime.now()
    fct = load_fct_data_from_grib(
        grib_output_dir=args.grib_output_dir,
        params=args.params,
        step=args.lead_time
    )
    LOG.info("Loaded forecast data from GRIB files in %.2f seconds", (datetime.now() - start).total_seconds())


    # get truth data (COSMO-2 analysis aka KENDA-1)
    if args.zarr_dataset:
        start = datetime.now()
        kenda = load_kenda1_data_from_zarr(
            zarr_dataset=args.zarr_dataset,
            valid_times=fct.valid_time,
            params=args.params
        ).compute().chunk({"y": -1, "x": -1})
        LOG.info("Loaded KENDA-1 data from Zarr dataset in %.2f seconds", (datetime.now() - start).total_seconds())
    elif args.archive_root:
        # kenda = load_kenda1_data_from_grib(
        #     archive_root=args.archive_root,
        #     valid_times=fct.valid_time,
        #     params=args.params
        # )
        pass
    else:
        raise ValueError("Either --archive_root or --zarr_dataset must be provided.")

    # compute metrics and statistics
    start = datetime.now()
    error = fct - kenda
    results = {}
    results["BIAS"] = error.mean(["y", "x"])
    results["RMSE"] = np.sqrt((error ** 2).mean(["y", "x"]))
    results["MAE"] = abs(error).mean(["y", "x"])
    results["STD"] = error.std(["y", "x"])
    results["CORR"] = (corr := xr.Dataset({k: xr.corr(fct[k], kenda[k], dim=["y", "x"]) for k in fct.data_vars}))
    results["R2"] = corr ** 2
    results = xr.Dataset({k: v.to_array("param") for k, v in results.items()})
    results = results.to_array("metric").to_dataframe(name="value").reset_index()
    LOG.info("Computed metrics in %.2f seconds", (datetime.now() - start).total_seconds())

    # # save results to CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output)
    LOG.info("Saved results to %s", args.output)
    LOG.info("Verification completed successfully.")

if __name__ == "__main__":

    # TODO: temporarily we must specify all params because it's currently required in the
    # request with meteodatalab.grib_decoder.load. This will be fixed in a future version.
    PARAMS = ["T_2M", "TD_2M", "U_10M", "V_10M", "PS", "PMSL", "TOT_PREC", "T_G"]
    # PARAMS += ["T", "U", "V","QV","FI"]

    parser = ArgumentParser(description="Verify COSMO-E forecast data.")

    parser.add_argument("--grib_output_dir", type=Path, required=True,
                        help="Path to the Zarr dataset containing COSMO-E forecast data.")
    
    # truth data must be provided either as a GRIB archive or a anemoi-generated Zarr dataset
    truth_group = parser.add_mutually_exclusive_group(required=True)
    truth_group.add_argument("--archive_root", type=Path, required=False,
                        help="Root directory of the archive containing GRIB files.")
    truth_group.add_argument("--zarr_dataset", type=Path, required=False,
                        help="Path to the Zarr dataset containing COSMOe data.")
    
    
    parser.add_argument("--reftime", type=lambda s: datetime.strptime(s, "%Y%m%d%H%M"),
                        help="Valid time for the data in ISO format (default: 6 hours ago).")
    parser.add_argument("--params", type=lambda x: x.split(","), required=False,
                        default=PARAMS,
                        help="Comma-separated list of parameters to verify (default: T_2M,TD_2M,U_10M,V_10M).")
    parser.add_argument("--lead_time",
                        type=_parse_lead_time,
                        default="0/126/6",
                        help="Lead time in the format 'start/stop/step' (default: 0/126/6).")
    parser.add_argument("--output", type=Path, default="verif.csv",
                        help="Output file to save the verification results (default: verif.csv)."),
    args = parser.parse_args()
    
    main(args)

# run examples 
# uv run workflow/scripts/verif_from_grib.py --zarr_dataset /scratch/mch/fzanetta/data/anemoi/datasets/mch-co2-an-archive-0p02-2015-2020-6h-v3-pl13.zarr --reftime 202006011200 --output debug_verif_zarr.csv
# uv run workflow/scripts/verif_from_grib.py \
#     --grib_output_dir /users/fzanetta/projects/mch-anemoi-evaluation/output/7c58e59d24e949c9ade3df635bbd37e2/202001050600/grib \
#     --archive_root /scratch/mch/fzanetta/data/KENDA-1 \
#     --reftime 202006011200 \
#     --output debug_verif_grib.csv
#     --params T_2M,TD_2M,U_10M,V_10M,PS,PMSL,TOT_PREC



# uv run workflow/scripts/verif_from_grib.py \
#     --grib_output_dir /users/fzanetta/projects/mch-anemoi-evaluation/output/7c58e59d24e949c9ade3df635bbd37e2/202001050600/grib \
#     --zarr_dataset /scratch/mch/fzanetta/data/anemoi/datasets/mch-co2-an-archive-0p02-2015-2020-6h-v3-pl13.zarr \
#     --reftime 202006011200 \
#     --output debug_verif_grib.csv
