import argparse
import json as _json
import logging
import math
import sys as _sys
from pathlib import Path

import jinja2
import xarray as xr

_sys.path.append(str(Path(__file__).parent))
from verification_plot_metrics import _ensure_unique_lead_time, _select_best_sources
from verification import decode_metric

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def program_summary_log(args):
    """Log a welcome message with the script and template information."""
    LOG.info("=" * 80)
    LOG.info("Generating experiment verification dashboard")
    LOG.info("=" * 80)
    LOG.info("Verification files: \n%s", "\n".join(str(f) for f in args.verif_files))
    LOG.info("Template: %s", args.template)
    LOG.info("Script: %s", args.script)
    LOG.info("Output: %s", args.output)
    LOG.info("=" * 80)


def main(args):
    program_summary_log(args)

    # Load, de-duplicate lead_time, and keep best provider per source (same logic as verif_plot_metrics)
    dfs = [xr.open_dataset(f) for f in args.verif_files]
    dfs = [_ensure_unique_lead_time(d) for d in dfs]
    dfs = _select_best_sources(dfs)
    ds = xr.concat(dfs, dim="source", join="outer")
    LOG.info("Loaded verification netcdf: \n%s", ds)

    # extract only  non-spatial variables to pd.DataFrame
    nonspatial_vars = [d for d in ds.data_vars if "spatial" not in d]
    df = ds[nonspatial_vars].to_array("stack").to_dataframe(name="value").reset_index()
    df[["param", "metric"]] = df["stack"].str.split(".", n=1, expand=True)
    df["metric"] = df.metric.apply(decode_metric)
    df.drop(columns=["stack"], inplace=True)
    df["lead_time"] = df["lead_time"].dt.total_seconds() / 3600
    # convert numeric column init_hour to string in format HH:00 UTC and replace -999 with "all"
    df["init_hour"] = df["init_hour"].astype(str).str.zfill(2) + ":00 UTC"
    df["init_hour"] = df["init_hour"].where(df["init_hour"] != "-999:00 UTC", "all")

    # retain only rows relevant for the active stratifications
    stratification = args.stratification
    if "region" not in stratification:
        df = df[df["region"] == "all"]
    if "season" not in stratification:
        df = df[df["season"] == "all"]
    if "init_hour" not in stratification:
        df = df[df["init_hour"] == "all"]

    # create a new column for line styles and shapes in dashboard
    df.dropna(inplace=True)
    LOG.info("Loaded verification data frame: \n%s", df)

    # get unique sources and params
    sources = df["source"].unique()
    params = df["param"].unique()
    metrics = df["metric"].unique()
    regions = df["region"].unique() if "region" in stratification else []
    seasons = df["season"].unique() if "season" in stratification else []
    init_hours = df["init_hour"].unique() if "init_hour" in stratification else []

    # Columnar JSON: store columns + data array (no repeated keys per row).
    # region_season_init is a derived column — computed in JS at parse time.
    # Round float values to 6 significant digits to avoid unnecessary precision.
    def _round_sig(x, sig=6):
        if not math.isfinite(x) or x == 0:
            return None  # Infinity/NaN → null in JSON
        d = math.ceil(math.log10(abs(x)))
        power = sig - d
        factor = 10 ** power
        return round(x * factor) / factor

    export_cols = ["source", "param", "metric", "lead_time", "value", "region", "season", "init_hour"]
    df_export = df[export_cols].copy()
    df_export["value"] = df_export["value"].apply(lambda v: _round_sig(float(v), 6) if v is not None else v)

    def _sanitize(v):
        """Replace non-finite floats (NaN/Inf) with None so JSON stays valid."""
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v

    df_json = _json.dumps({
        "columns": export_cols,
        "data": [[_sanitize(v) for v in row] for row in df_export.values.tolist()],
    })

    # compute number of bytes in the JSON string
    json_size = len(df_json.encode("utf-8"))
    LOG.info("Size of embedded JSON data: %d bytes", json_size)

    # read script
    with open(args.script, "r") as f:
        js_src = f.read()

    # generate HTML from Jinja2 template
    environment = jinja2.Environment(
        loader=jinja2.FileSystemLoader(args.template.parent)
    )
    template = environment.get_template(args.template.name)
    html = template.render(
        verif_data=df_json,
        js_src=js_src,
        sources=sources,
        params=params,
        metrics=metrics,
        regions=regions,
        seasons=seasons,
        init_hours=init_hours,
        stratification=stratification,
        header_text=args.header_text,
        configfile_content=open(args.configfile, "r").read()
        if args.configfile.is_file()
        else "",
    )
    LOG.info("Size of generated HTML: %d bytes", len(html.encode("utf-8")))

    output = Path(args.output) / "dashboard.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        f.write(html)

    LOG.info("Dashboard generated and saved to %s", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a dashboard for experiment verification data."
    )
    parser.add_argument(
        "--verif_files",
        type=Path,
        nargs="+",
        default=[],
        help="Paths to verification data files (not used in this mock).",
    )
    parser.add_argument(
        "--template", type=Path, required=True, help="Path to the Jinja2 template file."
    )
    parser.add_argument(
        "--script",
        type=Path,
        required=True,
        help="Path to the JavaScript source file for the dashboard.",
    )
    parser.add_argument(
        "--header_text",
        type=str,
        default="",
        help="Text to display in the header of the dashboard.",
    )
    parser.add_argument(
        "--stratification",
        nargs="*",
        default=["region", "season", "init_hour"],
        help="Stratification dimensions to include in the dashboard (any of region, season, init_hour).",
    )
    parser.add_argument(
        "--configfile",
        type=Path,
        help="Path to config file for the evalml run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dashboard.html"),
        help="Path to save the generated HTML dashboard file.",
    )
    args = parser.parse_args()

    main(args)

"""
Example usage:
python report_experiment_dashboard.py \
    --verif_files output/data/*/*/verif_aggregated.nc \
    --template resources/report/dashboard/template.html.jinja2 \
    --script resources/report/dashboard/script.js
"""
