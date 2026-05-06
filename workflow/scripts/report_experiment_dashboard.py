import argparse
import json
import logging
import sys as _sys
from pathlib import Path

import jinja2
import xarray as xr

from diagnostics import melt_for_dashboard

_sys.path.append(str(Path(__file__).parent))
from verification_plot_metrics import _ensure_unique_lead_time
from verification_plot_metrics import _select_best_sources

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def _load_sysmetrics(sysmetrics_file: Path) -> tuple[str, list[str], list[str]]:
    """Load system metrics JSON and melt to long format for Vega-Lite."""
    if not sysmetrics_file or not sysmetrics_file.is_file():
        return "[]", [], []
    with open(sysmetrics_file) as fh:
        records = json.load(fh)
    sysmetrics_json, sources, model_types = melt_for_dashboard(records)
    LOG.info(
        "Loaded system metrics for %d source(s), %d model type(s)",
        len(sources),
        len(model_types),
    )
    return sysmetrics_json, sources, model_types


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
    df.drop(columns=["stack"], inplace=True)
    df["lead_time"] = df["lead_time"].dt.total_seconds() / 3600
    # convert numeric column init_hour to string in format HH:00 UTC and replace -999 with "all"
    df["init_hour"] = df["init_hour"].astype(str).str.zfill(2) + ":00 UTC"
    df["init_hour"] = df["init_hour"].where(df["init_hour"] != "-999:00 UTC", "all")
    # create a new column for line styles and shapes in dashboard
    df["region_season_init"] = (
        "Region: "
        + df["region"].astype(str)
        + ", Season: "
        + df["season"].astype(str)
        + ", Init: "
        + df["init_hour"].astype(str)
    )

    df.dropna(inplace=True)
    LOG.info("Loaded verification data frame: \n%s", df)

    # get unique sources and params
    sources = df["source"].unique()
    params = df["param"].unique()
    metrics = df["metric"].unique()
    regions = df["region"].unique()
    seasons = df["season"].unique()
    init_hours = df["init_hour"].unique()

    # get json string to embed in the HTML
    df_json = df.to_json(orient="records", lines=False)

    # compute number of bytes in the JSON string
    json_size = len(df_json.encode("utf-8"))
    LOG.info("Size of embedded JSON data: %d bytes", json_size)

    # load system metrics
    sysmetrics_json, sysmetrics_sources, sysmetrics_model_types = _load_sysmetrics(
        args.sysmetrics_file
    )

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
        header_text=args.header_text,
        configfile_content=open(args.configfile, "r").read()
        if args.configfile.is_file()
        else "",
        sysmetrics_data=sysmetrics_json,
        sysmetrics_sources=sysmetrics_sources,
        sysmetrics_model_types=sysmetrics_model_types,
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
        "--configfile",
        type=Path,
        help="Path to config file for the evalml run.",
    )
    parser.add_argument(
        "--sysmetrics_file",
        type=Path,
        default=None,
        help="Path to system metrics JSON produced by parse_inference_logs.py.",
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
