from pathlib import Path
import argparse
from glob import glob
from itertools import product
from datetime import datetime, timedelta
import sys
import os

DEFINITION_PATH = Path(sys.prefix) / "share/eccodes-cosmo-resources/definitions"
os.environ["ECCODES_DEFINITION_PATH"] = str(DEFINITION_PATH)

import ujson
from kerchunk.combine import merge_vars, MultiZarrToZarr
from kerchunk.grib2 import scan_grib
from kerchunk.df import refs_to_dataframe
from tqdm import tqdm
import numpy as np
import xarray as xr

def merged_var_refs(gribfile: Path, filter: dict):
    """Generate a json reference for a single file, merging all messages."""
    vars = scan_grib(gribfile, filter=filter)
    ref = merge_vars(vars)
    return ref


def create_references(files: list[Path], filter: dict) -> list[dict]:
    """Create references for a list of files."""
    return [merged_var_refs(f, filter=filter) for f in tqdm(files, desc="Creating references...")]


def find_files(glob_pattern: str, expand: dict | None = None):
    """Find files matching a glob pattern, optionally expanding the pattern with a dict of values."""
    if expand is not None:
        for k, v in expand.items():
            if not isinstance(v, list):
                expand[k] = [v]
        expand = [
            dict(zip(expand.keys(), v)) for v in product(*expand.values())
        ]
        files = []
        for exp in expand:
            res = glob(glob_pattern.format(**exp))
            if not res:
                raise ValueError(f"No files found for pattern {glob_pattern} with values {exp}")
            files.extend(res)
        return files
    else:
        return sorted(glob(glob_pattern))


def create_combine_pl_refs(files, levelist: list[int] = [850, 925, 1000]):
    """Create and combine references for pressure level parameters."""
    
    refs = []
    for level in levelist:
        filter = {"levtype": "pl", "level": level}
        refs += create_references(files, filter=filter)

    refs = MultiZarrToZarr(
        refs,
        concat_dims=["forecast_reference_time", "step", "isobaricInhPa"],
        identical_dims=["latitude", "longitude"],
        preprocess=_pl_mzz_pre_process,
        postprocess=_pl_mzz_post_process,
    ).translate()

    return refs


def create_combine_sfc_refs(files):
    """Create and combine references for surface parameters."""
    
    filter = {"levtype": "sfc"}
    refs = create_references(files, filter=filter)

    refs = MultiZarrToZarr(
        refs,
        concat_dims=["forecast_reference_time", "step"],
        identical_dims=["latitude", "longitude"],
        preprocess=_sfc_mzz_pre_process,
        postprocess=_sfc_mzz_post_process,
    ).translate()

    return refs

# -----------------------------------------------------------------------------
# Pre- and post-processing functions for the MultiZarrToZarr class
# these are hardcoded for now but could be made more flexible
# -----------------------------------------------------------------------------

def _sfc_mzz_pre_process(refs: dict[str, dict]):
    DROP_KEYS = ("heightAboveGround", "surface")
    # DROP_KEYS = ("heightAboveGround", "surface", "valid_time")
    for k in list(refs):
        if k.startswith(DROP_KEYS):
            refs.pop(k)
    for g in [".zattrs", ".zarray", "0"]:
        if f"time/{g}" in refs:
            refs[f"forecast_reference_time/{g}"] = refs.pop(f"time/{g}") 
    return refs


def _sfc_mzz_post_process(refs: dict[str, dict]):
    CF_SHORTNAME_MAP = {"t2m": "T_2M", "d2m": "TD_2M", "u10": "U_10M", "v10": "V_10M"}
    new_refs = {}
    for k in refs:
        if k.split("/")[0] in tuple(CF_SHORTNAME_MAP.keys()):
            _k, _v = k.split("/")
            new_refs[f"{CF_SHORTNAME_MAP[_k]}/{_v}"] = refs[k]
        else:
            new_refs[k] = refs[k]
    return new_refs


def _pl_mzz_pre_process(refs: dict[str, dict]):
    DROP_KEYS = ("valid_time")
    for k in list(refs):
        if k.startswith(DROP_KEYS):
            refs.pop(k)
    for g in [".zattrs", ".zarray", "0"]:
        if f"time/{g}" in refs:
            refs[f"forecast_reference_time/{g}"] = refs.pop(f"time/{g}") 
    return refs


def _pl_mzz_post_process(refs: dict[str, dict]):
    CF_SHORTNAME_MAP = {"h": "HEIGHT"}
    new_refs = {}
    for k in refs:
        if k.startswith(tuple(CF_SHORTNAME_MAP.keys())):
            _k, _v = k.split("/")
            new_refs[f"{CF_SHORTNAME_MAP[_k]}/{_v}"] = refs[k]
        else:
            new_refs[k] = refs[k]
    return new_refs

# -----------------------------------------------------------------------------

def cmd_generate(args):
    """Command to generate references from GRIB2 files."""
    files = find_files(args.template, expand=dict(args.values))
    if not files:
        raise ValueError(f"No files found for template {args.template} with values {args.values}")

    sfc_refs = create_combine_sfc_refs(files)
    pl_refs = create_combine_pl_refs(files, levelist=args.levels)
    refs = merge_vars([sfc_refs, pl_refs])

    with open(args.output, "w") as f:
        ujson.dump(refs, f, indent=4)

    if args.inspect:
        ds = xr.open_dataset(refs, engine="kerchunk", decode_timedelta=True)
        print(ds)

def cmd_combine(args):
    """Command to combine existing reference JSON files."""

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    step = timedelta(hours=args.step)
    current = start
    refs = []
    while current <= end:
        print(f"Processing {current}...")
        timestamp = current.strftime("%Y%m%d%H%M")
        ref_file = args.experiment_root / timestamp / f"references.json"
        if not ref_file.exists():
            raise FileNotFoundError(f"Reference file {ref_file} does not exist.")
        refs.append(ref_file.as_posix())
        current += step

    refs = MultiZarrToZarr(
        refs,
        concat_dims=["forecast_reference_time"],
        identical_dims=["latitude", "longitude"],
    ).translate()

    if args.inspect:
        # import fsspec
        # fs = fsspec.filesystem("reference", fo=refs, remote_options={'anon':True}).get_mapper("")
        # ds = xr.open_zarr(fs, decode_timedelta=True, consolidated=False)
        ds = xr.open_dataset(refs, engine="kerchunk", decode_timedelta=True, chunks={"forecast_reference_time": "auto", "step": 1})
        # print(ds)
        # print(ds["T_2M"][:].mean(["forecast_reference_time","step"]).compute(num_workers=8))
        # print(ds.isel(forecast_reference_time=slice(None, 50)).persist(num_workers=128).mean().compute(num_workers=16))
        # print(ds["T_2M"].persist(num_workers=128).mean().compute(num_workers=16))
        # exit()
        breakpoint()

    refs_to_dataframe(refs, str(args.experiment_root / args.output))


def parse_kv(s: str) -> tuple[str, int | list[int] | str]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"Expected key=value, got '{s}'")
    key, val = s.split("=", 1)
    if "," in val:
        return key, [_cast_int(x) for x in val.split(",")]
    if "/" in val:
        start, stop, step = map(int, val.split("/"))
        return key, list(range(start, stop, step))
    return key, _cast_int(val)

def _cast_int(v: str) -> int | str:
    return int(v) if v.isdigit() else v


def main():

    # breakpoint()
    # show defaults
    parser = argparse.ArgumentParser(
        prog="kerchunk-tool",
        description="Tool to generate and combine kerchunk references for GRIB2 files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Generate subcommand
    gen = sub.add_parser("generate", 
                        help="Scan GRIB2 files and create kerchunk references for model run.", 
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    gen.add_argument("template", help="Filename template without '*'.")
    gen.add_argument("values", nargs="*", type=parse_kv,
                     help="Template expansions as key=value.",)
    
    gen.add_argument("--levels", nargs="*", type=int, default=[850, 925, 1000],
                     help="Pressure levels to include.")
    gen.add_argument( "--output", default="references.json",
                     help="Output JSON path.")
    gen.add_argument("--inspect", action="store_true", help="Open and print the xarray Dataset.")
    gen.set_defaults(func=cmd_generate)

    # Combine subcommand
    cmb = sub.add_parser("combine",
                         help="Combine references of multiple runs into one parquet-based reference.",
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmb.add_argument("experiment_root", type=Path,
                    help="Root directory of the experiment containing reference files.")
    cmb.add_argument("--start", type=str, default="2020-01-01T06:00",
                     help="Start time of the experiment in YYYY-MM-DDTHH:MM format.")
    cmb.add_argument("--end", type=str, default="2020-04-30T18:00",
                        help="End time of the experiment in YYYY-MM-DDTHH:MM format.")
    cmb.add_argument("--step", type=int, default="6",
                        help="Step size in hours for the experiment.")

    cmb.add_argument("--output", default="referenced.vzarr", help="Parquet-based virtual zarr output path.")
    cmb.add_argument("--inspect", action="store_true", help="Open and print the xarray Dataset.")
    cmb.set_defaults(func=cmd_combine)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":

    main()