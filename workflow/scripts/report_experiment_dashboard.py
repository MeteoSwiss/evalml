from pathlib import Path
import argparse
import logging

import pandas as pd
import jinja2

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def read_verif_file(Path: str) -> pd.DataFrame:
    """Read a verification file and return it as a DataFrame."""
    df = pd.read_csv(
        Path,
        dtype={"lead_time": str},
    )
    df["lead_time"] = pd.to_timedelta(df["lead_time"])
    df["lead_time"] = df["lead_time"].dt.total_seconds() / 3600  # convert to hours
    return df


def combine_verif_files(verif_files: Path) -> pd.DataFrame:
    """
    Combine multiple verification files into a single DataFrame.
    Each file is expected to have a 'model' column indicating the model name.
    """
    df = pd.DataFrame()
    for i, file in enumerate(verif_files):
        _df = read_verif_file(file)
        df = pd.concat([df, _df])
    subset_cols = [c for c in df.columns if c not in ["value"]]
    df = df.drop_duplicates(subset=subset_cols)
    df.rename(columns={"label": "model"}, inplace=True)
    df = df.reset_index(drop=True)

    return df


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

    # load and combine verification data
    df = combine_verif_files(args.verif_files)

    # TODO: remove this when we have the logic to handle these groups
    df = df[
        (df["hour"] == "all") & (df["season"] == "all") & (df["init_hour"] == "all")
    ]
    LOG.info("Loaded verification data: \n%s", df)

    # get unique models and params
    models = df["model"].unique()
    params = df["param"].unique()
    metrics = df["metric"].unique()

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
        models=models,
        params=params,
        metrics=metrics,
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
    --verif_files output/data/*/*/verif_aggregated.csv \
    --template resources/report/dashboard/template.html.jinja2 \
    --script resources/report/dashboard/script.js
"""
