import argparse
import logging
import sys as _sys
from pathlib import Path

import jinja2
import xarray as xr

_sys.path.append(str(Path(__file__).parent))
from verif_plot_metrics import _ensure_unique_lead_time
from verif_plot_metrics import _select_best_sources

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
    df.drop(columns=["stack"], inplace=True)
    df["lead_time"] = df["lead_time"].dt.total_seconds() / 3600
    # select only results for all seasons and init_hours (for now)
    df = df[(df["season"] == "all") & (df["init_hour"] == -999)]
    df.dropna(inplace=True)
    LOG.info("Loaded verification data frame: \n%s", df)

    # get unique sources and params
    sources = df["source"].unique()
    params = df["param"].unique()
    metrics = df["metric"].unique()
    regions = df["region"].unique()

    # get json string to embed in the HTML
    df_json = df.to_json(orient="records", lines=False)

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
        header_text=args.header_text,
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
