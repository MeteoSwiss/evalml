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


def dataset_metrics_to_dataframe(
    ds: xr.Dataset,
    forbidden_dims=("values",),
    metric_dims=("source", "season", "init_hour", "region", "lead_time", "eps"),
):
    """
    Convert a verification xarray.Dataset to a pandas.DataFrame compatible with
    the old `to_array("stack").to_dataframe()` workflow.

    Returns columns:
      - metric_dims...
      - stack   (e.g. "T_2M.MAE")
      - value
    """

    # keep only non-spatial metric variables
    metric_vars = []
    for v in ds.data_vars:
        da = ds[v]

        # skip spatial metrics
        if "spatial" in v:
            continue

        # skip forbidden dimensions
        if any(d in da.dims for d in forbidden_dims):
            continue

        # only keep vars whose dims are a subset of expected metric dims
        if not set(da.dims).issubset(metric_dims):
            continue

        metric_vars.append(v)

    if not metric_vars:
        raise ValueError("No compatible metric variables found in dataset")

    # stack variables exactly like the original code
    df = (
        ds[metric_vars]
        .to_array("stack")
        .to_dataframe(name="value")
        .reset_index()
    )

    return df

def main(args):
    program_summary_log(args)

    # Load, de-duplicate lead_time, and keep best provider per source (same logic as verif_plot_metrics)
    drop_variables = ["TOT_PREC.MAE.spatial", "TOT_PREC.RMSE.spatial", "TOT_PREC.BIAS.spatial", 
                      "PMSL.MAE.spatial", "PMSL.RMSE.spatial", "PMSL.BIAS.spatial", 
                      "PS.MAE.spatial", "PS.RMSE.spatial", "PS.BIAS.spatial", 
                      "V_10M.MAE.spatial", "V_10M.RMSE.spatial", "V_10M.BIAS.spatial", 
                      "U_10M.MAE.spatial", "U_10M.RMSE.spatial", "U_10M.BIAS.spatial", 
                      "TD_2M.MAE.spatial", "TD_2M.RMSE.spatial", "TD_2M.BIAS.spatial", 
                      "T_2M.MAE.spatial", "T_2M.RMSE.spatial", "T_2M.BIAS.spatial"]
    
    dfs = [xr.open_dataset(f, drop_variables = drop_variables) for f in args.verif_files]
    dfs = [_ensure_unique_lead_time(d) for d in dfs]
    dfs = _select_best_sources(dfs)
    ds = xr.concat(dfs, dim="source", join="outer")
    LOG.info("Loaded verification netcdf: \n%s", ds)

    # extract only  non-spatial variables to pd.DataFrame
    # nonspatial_vars = [d for d in ds.data_vars if "spatial" not in d]
    # df = ds[nonspatial_vars].to_array("stack").to_dataframe(name="value").reset_index()

    df = dataset_metrics_to_dataframe(
        ds,
        forbidden_dims=("values",),  # critical!
        metric_dims=("source", "season", "init_hour", "region", "lead_time", "eps"),
    )

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
