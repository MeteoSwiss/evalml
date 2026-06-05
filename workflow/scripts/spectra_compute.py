"""Compute power spectra for one source (run, baseline, or truth) at one init."""

import logging
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import pandas as pd

from data_input import load_forecast_data, load_truth_data, parse_steps
from spectra import io
from spectra.compute import compute_source_spectra

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main():
    ap = ArgumentParser()
    ap.add_argument("--forecast", help="Forecast root (grib dir or baseline archive).")
    ap.add_argument("--truth", help="Truth zarr root (use instead of --forecast).")
    ap.add_argument("--reftime", required=True)
    ap.add_argument("--steps", required=True, help="start/stop/step in hours.")
    ap.add_argument("--lead_times", required=True, help="comma-separated hours.")
    ap.add_argument("--variables", required=True, help="comma-separated names.")
    ap.add_argument("--method", default="dct")
    ap.add_argument("--label", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if bool(args.forecast) == bool(args.truth):
        ap.error("Exactly one of --forecast / --truth must be supplied.")

    reftime = datetime.strptime(args.reftime, "%Y%m%d%H%M")
    all_steps = parse_steps(args.steps)
    lead_times = [int(s) for s in args.lead_times.split(",")]
    missing = [s for s in lead_times if s not in all_steps]
    if missing:
        LOG.warning("Requested lead times %s not in source steps; dropping.", missing)
    lead_times = [s for s in lead_times if s in all_steps]
    if not lead_times:
        LOG.error(
            "None of the requested lead times are available in source steps %s; aborting.",
            all_steps,
        )
        raise SystemExit(1)
    variables = args.variables.split(",")
    params = io.required_params(variables)

    if args.truth:
        ds = load_truth_data(Path(args.truth), reftime, all_steps, params)
        if "time" in ds.dims and "step" not in ds.dims:
            valid = pd.to_datetime(ds["time"].values)
            steps_h = (
                ((valid - pd.Timestamp(reftime)) / pd.Timedelta(hours=1))
                .round()
                .astype(int)
            )
            if pd.Series(steps_h).duplicated().any():
                raise ValueError(
                    f"Truth timestamps rounded to integer hours produced duplicate "
                    f"steps: {steps_h.tolist()}. Sub-hourly truth is unsupported for spectra."
                )
            ds = ds.assign_coords(step=("time", steps_h)).swap_dims({"time": "step"})
    else:
        ds = load_forecast_data(Path(args.forecast), reftime, all_steps, params)

    result = compute_source_spectra(ds, variables, lead_times, args.method, args.label)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_netcdf(out)
    LOG.info("Saved spectra to %s", out)


if __name__ == "__main__":
    main()
