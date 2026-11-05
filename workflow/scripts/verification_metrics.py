import json
import logging
from argparse import ArgumentParser
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from verification import (  # noqa: E402
    CIRCULAR_PARAMS,
    _apply_masks,
    _compute_statistics,
    _region_and_station_masks,
    _too_many_missing,
    apply_lapse_rate_correction_inplace,
    verify,
    verify_ensemble,
)
from verification.spatial import map_forecast_to_truth  # noqa: E402
from verification.valid_mask import apply_valid_mask, combine_valid_masks  # noqa: E402
from data_input import (
    parse_steps,
    load_forecast_data,
    load_truth_data,
    truth_type,
)  # noqa: E402

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class ScriptConfig(Namespace):
    """Configuration for the script to verify baseline forecast data."""

    archive_root: Path = None
    truth: Path = None
    baseline_zarr: Path = None
    reftime: datetime = None
    params: list[str] = ["T_2M", "TD_2M", "U_10M", "V_10M"]
    steps: list[int] = parse_steps("0/120/6")


def load_holdout_stations(path: Path) -> list[str]:
    """Read the holdout station nat_abbr from the CSV written by rule station_holdout."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)["nat_abbr"].tolist()


def select_holdout_stations(
    requested: list[str], truth: xr.Dataset, truth_root: Path
) -> list[str] | None:
    """Holdout stations for the station_group stratification, or None to skip it.

    The stratification only applies to station truth (jretrieve observations),
    whose ``values`` dimension holds the station nat_abbr. Gridded truth may
    also have a ``values`` dimension (unstructured grids), so the truth source
    is checked instead. Stations missing from the truth are dropped.
    """
    if truth_type(truth_root) != "jretrieve":
        LOG.warning(
            "Holdout stations given, but the truth is gridded data, not station "
            "observations; station_group stratification skipped."
        )
        return None
    truth_stations = set(truth["values"].values)  # station nat_abbr
    holdout_stations = [s for s in requested if s in truth_stations]
    if not holdout_stations:
        LOG.warning(
            "None of the %d holdout stations are in the station truth; "
            "station_group stratification skipped.",
            len(requested),
        )
        return None
    LOG.info(
        "Station holdout: %d / %d truth stations in holdout group",
        len(holdout_stations),
        len(truth_stations),
    )
    return holdout_stations


def program_summary_log(args):
    """Log a welcome message with the script information."""
    LOG.info("=" * 80)
    LOG.info("Running verification of baseline forecast data")
    LOG.info("=" * 80)
    LOG.info("Baseline dataset: %s", args.baseline_zarr)
    LOG.info("Truth dataset: %s", args.truth)
    LOG.info("Reference time: %s", args.reftime)
    LOG.info("Parameters to verify: %s", args.params)
    LOG.info("Lead time: %s", args.lead_time)
    LOG.info("Thresholds to verify: %s", args.threshold_dict)
    LOG.info("Output file: %s", args.output)
    LOG.info("=" * 80)


def load_ensemble_forecast(
    root: Path, reftime: datetime, steps: list[int], params: list[str], members: int
) -> xr.Dataset:
    """Load the members ``root/001 .. root/{members:03d}`` of an ensemble run.

    Each member is loaded with ``load_forecast_data`` (same derived and period
    parameters as a deterministic run) and the members are concatenated on a
    ``member`` dimension (1..N). A missing member is an error. All members get
    member 001's ``elevation`` (model orography, the same for every member), so
    the lapse-rate correction uses one elevation for the whole ensemble.
    """
    datasets = []
    for m in range(1, members + 1):
        member_dir = Path(root) / f"{m:03d}"
        if not any(member_dir.glob("*.grib")):
            raise FileNotFoundError(
                f"Ensemble member {m:03d} of {members}: no GRIB files in {member_dir}"
            )
        datasets.append(load_forecast_data(member_dir, reftime, steps, params))
    elevation = datasets[0].coords.get("elevation")
    datasets = [ds.drop_vars("elevation", errors="ignore") for ds in datasets]
    fcst = xr.concat(
        datasets,
        dim=pd.Index(range(1, members + 1), name="member"),
        coords="minimal",
        compat="override",
        join="exact",
    )
    if elevation is not None:
        fcst = fcst.assign_coords(elevation=elevation)
    return fcst


def load_aligned_data(args, params: list[str]) -> tuple[xr.Dataset, xr.Dataset]:
    """Load forecast and truth for ``params``, with the forecast mapped to the truth
    points and the truth selected at the forecast valid times.

    With ``args.members`` set, ``args.forecast`` is the ``grib/`` directory of an
    ensemble run and the forecast has a ``member`` dimension (load_ensemble_forecast).
    """

    # get baseline forecast data
    now = datetime.now()

    if getattr(args, "members", None):
        fcst = load_ensemble_forecast(
            args.forecast, args.reftime, args.steps, params, args.members
        )
    else:
        fcst = load_forecast_data(
            args.forecast, args.reftime, args.steps, params, member=args.member
        )

    LOG.info(
        "Loaded forecast data in %s seconds: \n%s",
        (datetime.now() - now).total_seconds(),
        fcst,
    )

    # get truth data
    now = datetime.now()
    truth = load_truth_data(args.truth, args.reftime, args.steps, params)
    LOG.info(
        "Loaded truth data in %s seconds: \n%s",
        (datetime.now() - now).total_seconds(),
        truth,
    )

    # align forecast and truth data spatially and temporally
    now = datetime.now()
    fcst = map_forecast_to_truth(fcst, truth)
    # map_forecast_to_truth uses fancy indexing which collapses the spatial
    # dimension into one monolithic dask chunk; rechunk by step so that
    # verify() can parallelise over time steps rather than materialising the
    # full (regions × steps × values) array at once.
    fcst = fcst.chunk({"step": 1})
    truth = truth.sel(time=fcst["valid_time"])
    LOG.info(
        "Aligned forecast and truth in %s seconds",
        (datetime.now() - now).total_seconds(),
    )
    return fcst, truth


def member_mean_statistics(
    fcst: xr.Dataset,
    truth: xr.Dataset,
    regions: list[dict],
    holdout_stations: list[str] | None = None,
    params: list[str] | None = None,
) -> xr.Dataset:
    """Forecast statistics ``{p}.{mean,var,min,max}`` of an ensemble, averaged over
    the members (without a ``source`` dim).

    Each member is masked as ``verify`` masks the forecast (regions, station groups,
    NaN where too many forecast values are missing; a point is missing if any member
    is). The ensemble mean is smoother than any member, so its own variance and
    extremes would not describe the forecast fields.
    """
    dim = ["x", "y"] if "x" in fcst.dims and "y" in fcst.dims else ["values"]
    fcst, truth = xr.align(fcst, truth, join="inner", copy=False)
    masks, station_masks = _region_and_station_masks(truth, regions, holdout_stations)
    stats = []
    for param in params or list(fcst.data_vars):
        if param not in truth.data_vars:
            continue
        members = fcst[param].where(fcst[param].notnull().all("member"))
        too_many_missing = _too_many_missing(
            members.isnull().any("member"),
            _apply_masks(truth[param], masks, station_masks),
            dim,
            0.0,
        )
        member_stats = _compute_statistics(
            _apply_masks(members, masks, station_masks),
            prefix=param + ".",
            dim=dim,
            circular=param in CIRCULAR_PARAMS,
        )
        stats.append(
            member_stats.mean("member")
            .squeeze("source", drop=True)
            .where(~too_many_missing)
        )
    return xr.merge(stats).compute()


def verify_ensemble_run(
    fcst: xr.Dataset, truth: xr.Dataset, source_id: str, truth_source_id: str, **kwargs
) -> xr.Dataset:
    """Scores of an ensemble run (``fcst`` with a ``member`` dimension).

    The deterministic scores are those of the ensemble mean (``verify``), under the
    run's own ``source``; the ensemble scores (``verify_ensemble``: CRPS, ENS_VAR,
    MEMBER_MSE, rank_hist, prob_table) are merged next to them. A point where any
    member is missing is missing in the ensemble mean; the ensemble mean of a circular
    parameter (wind direction) is the direction of the mean unit vector. The run's forecast statistics
    are averaged over the members (``member_mean_statistics``), except for circular
    parameters (wind direction), which keep those of the ensemble mean.
    """
    n_members = fcst.sizes["member"]
    LOG.info("Verifying an ensemble of %d members", n_members)
    ens_mean = fcst.mean("member", skipna=False)
    for param in CIRCULAR_PARAMS.intersection(fcst.data_vars):
        # wind direction: direction of the mean unit vector (350 and 10 deg -> 0 deg)
        rad = np.deg2rad(fcst[param])
        ens_mean[param] = np.mod(
            np.degrees(
                np.arctan2(
                    np.sin(rad).mean("member", skipna=False),
                    np.cos(rad).mean("member", skipna=False),
                )
            ),
            360.0,
        ).assign_attrs(fcst[param].attrs)
    results = verify(ens_mean, truth, source_id, truth_source_id, **kwargs)

    stat_params = [p for p in fcst.data_vars if p not in CIRCULAR_PARAMS]
    member_stats = member_mean_statistics(
        fcst,
        truth,
        kwargs["regions"],
        kwargs.get("holdout_stations"),
        params=stat_params,
    )
    for name, value in member_stats.data_vars.items():
        current = results[name]
        results[name] = (
            xr.where(current["source"] == source_id, value, current)
            .transpose(*current.dims)
            .assign_attrs(current.attrs)
        )

    ens_scores = verify_ensemble(fcst, truth, source_id, truth_source_id, **kwargs)
    out = xr.merge([results, ens_scores], join="outer", compat="no_conflicts")
    out.attrs = {**results.attrs, "n_members": n_members}
    return out


def main(args: ScriptConfig):
    """Main function to verify baseline forecast data."""

    fcst, truth = load_aligned_data(args, args.params)

    if args.valid_masks:
        common_mask = combine_valid_masks(
            [xr.open_dataset(f) for f in args.valid_masks]
        )
        fcst, truth = apply_valid_mask(fcst, truth, common_mask)

    if args.lapse_rate_correction:
        apply_lapse_rate_correction_inplace(fcst, truth, args.params)

    holdout_stations = (
        select_holdout_stations(
            load_holdout_stations(args.holdout_stations_file), truth, args.truth
        )
        if args.holdout_stations_file
        else None
    )

    # compute metrics and statistics
    now = datetime.now()
    verify_kwargs = dict(
        regions=args.regions,
        threshold_dict=args.threshold_dict,
        holdout_stations=holdout_stations,
    )
    if "member" in fcst.dims:
        results = verify_ensemble_run(
            fcst, truth, args.source_id, args.truth_source_id, **verify_kwargs
        )
    else:
        results = verify(
            fcst, truth, args.source_id, args.truth_source_id, **verify_kwargs
        )
    LOG.info(
        "Computed verification metrics in %s seconds",
        (datetime.now() - now).total_seconds(),
    )

    # save results to NetCDF
    now = datetime.now()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Strip any non-NetCDF-serializable attrs (e.g. dicts set by compute_derived)
    # before writing; those attrs are only meaningful for in-memory use (e.g. plot_meteogram).
    _netcdf_types = (str, bytes, int, float, list, tuple)
    results.attrs = {
        k: v for k, v in results.attrs.items() if isinstance(v, _netcdf_types)
    }
    for _var in results.data_vars.values():
        _var.attrs = {
            k: v for k, v in _var.attrs.items() if isinstance(v, _netcdf_types)
        }
    results.earthkit.to_netcdf(args.output)
    LOG.info(
        "Saved verification results to %s in %s seconds",
        args.output,
        (datetime.now() - now).total_seconds(),
    )

    LOG.info("Program completed successfully.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Verify forecast or baseline data.")

    parser.add_argument(
        "--forecast",
        type=Path,
        required=True,
        default="/store_new/mch/msopr/ml/COSMO-E/FCST20.zarr",
        help="Path to the directory containing the grib forecast or to the zarr dataset containing baseline data.",
    )
    parser.add_argument(
        "--truth",
        type=Path,
        required=True,
        help="Path to the truth data.",
    )
    parser.add_argument(
        "--reftime",
        type=lambda s: datetime.strptime(s, "%Y%m%d%H%M"),
        default="202010010000",
        help="Valid time for the data in ISO format.",
    )
    parser.add_argument(
        "--params",
        type=lambda x: x.split(","),
        default=["T_2M", "TD_2M", "U_10M", "V_10M", "PS", "PMSL", "TOT_PREC"],
    )
    parser.add_argument(
        "--steps",
        type=parse_steps,
        default="0/120/6",
        help="Forecast steps in the format 'start/stop/step' (default: 0/120/6).",
    )
    parser.add_argument(
        "--source_id",
        type=str,
        required=True,
        help="Stable identifier for the forecast or baseline source (e.g. run_id or baseline_id).",
    )
    parser.add_argument(
        "--truth_source_id",
        type=str,
        required=True,
        help="Stable identifier for the truth source (e.g. truth_<TRUTH_HASH>).",
    )
    parser.add_argument(
        "--regions",
        type=json.loads,
        help=(
            "JSON list of region specs in config order. "
            'Each entry is {"type": "bbox", "name": ..., "bbox": [...]} '
            'or {"type": "shp", "name": ..., "path": ...}.'
        ),
        default="[]",
    )
    parser.add_argument(
        "--threshold_dict",
        type=lambda x: eval(x),
        help="Dictionary of thresholds for each parameter in the format '{param: [threshold1, threshold2, ...]}' (default: None).",
        default=None,
    )
    parser.add_argument(
        "--holdout_stations_file",
        type=Path,
        default=None,
        help=(
            "CSV file with a nat_abbr column listing the holdout stations. When set, adds "
            "station_group stratification (all/holdout/holdin) to all models and "
            "baselines using the truth dataset's station list."
        ),
    )
    parser.add_argument(
        "--member",
        type=str,
        default="000",
        help="Ensemble member to load: '000' for control, 'median' for the pre-computed median, 'mean' to average all members, or any 3-digit member ID.",
    )
    parser.add_argument(
        "--members",
        type=int,
        default=None,
        help=(
            "Number of members of an ensemble run. --forecast is then the run's grib/ "
            "directory holding one sub-directory per member (001, 002, ...); the "
            "ensemble mean is scored as the run's deterministic forecast and the "
            "ensemble scores are added."
        ),
    )
    parser.add_argument(
        "--valid_masks",
        type=Path,
        nargs="*",
        default=[],
        help=(
            "Valid-mask files (verification_valid_mask.py) of all experiment "
            "participants for this reference time. The parameters they hold are "
            "scored only where every participant has a value."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="verif.nc",
        help="Output file to save the verification results (default: verif.nc).",
    )
    parser.add_argument(
        "--lapse_rate_correction",
        action="store_true",
        default=False,
        help="Apply lapse-rate correction to T_2M and PS.",
    )
    args = parser.parse_args()

    main(args)
