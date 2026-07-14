r"""CSV table of where a model's analysis (lead time 0) sits between two baselines.

This is a standalone script: it is not part of the Snakemake workflow and is run
by hand. It only reads the aggregated verification files the pipeline has already
written, so produce those first — that is the heavy step, it runs the inference
and the verification on SLURM — and then build the table from the same config:

    evalml make config/varda-single-1.0.yaml experiment_all

    uv run workflow/scripts/report_analysis_verification_table.py \
        --config config/varda-single-1.0.yaml

Rows are metrics, columns are regions, and each cell is

    position = (run − INCA) / (ICON − INCA) × 100  [%]

which says only how close the run scored to each baseline: 0% means it scored the
same as INCA, 100% the same as ICON, and a value outside that range means it went
past one of them. A cell reads the same way for a metric where a low score is
better (RMSE) and for one where a high score is better (R2).

What the config must provide:
  - a truth of station observations (`truth: {root: jretrievedwh:...}`). Against a
    KENDA-based truth the table is meaningless: the model reproduces KENDA at lead
    time 0, so every error would be zero.
  - the two baselines used as anchors, INCA and ICON-CH1-CTRL by default
    (see --lower / --upper).
  - enough dates: with only a handful of init times most ETS thresholds are never
    reached and the table comes out full of NaN.
"""

import hashlib
import json
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from evalml.config import ConfigModel
from verification import decode_metric

DEFAULT_VARIABLES = {
    "U_10M": ["RMSE", "R2", "ETS"],
    "V_10M": ["RMSE", "R2", "ETS"],
    "T_2M": ["RMSE", "R2", "ETS"],
}
# The "aggregate over all" value of every stratification dimension other than
# region (same convention as report_scorecard.py).
_STRAT_ALL = {"season": "all", "init_hour": -999}


# ── Locating the three input files ────────────────────────────────────────────
# The pipeline names its outputs after content hashes of the config, so the paths
# are computed rather than searched for. The hashing mirrors the one in
# workflow/rules/common.smk; if that ever changes, the computed paths will not
# exist and the script fails instead of reading the wrong files.
_HASH_LENGTH = 4
_HASH_EXCLUDE = {"label"}  # BASELINE_HASH_EXCLUDE / TRUTH_HASH_EXCLUDE in common.smk


def _hash(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:_HASH_LENGTH]


def _load_config(path: str) -> dict:
    """The config as the Snakefile sees it: validated, with pydantic defaults.

    The defaults are hashed along with the rest, so hashing the raw YAML would
    produce different (wrong) ids.
    """
    raw = yaml.safe_load(Path(path).read_text())
    return ConfigModel.model_validate(raw).model_dump(mode="json")


def _resolve_sources(cfg: dict, lower: str, upper: str, run_source: str | None) -> list:
    """Return [(name, file, source_id)] for the run and the two anchor baselines."""
    data = Path(cfg.get("locations", {}).get("output_root", "output/")) / "data"
    truth = {k: v for k, v in cfg["truth"].items() if k not in _HASH_EXCLUDE}
    verif = {"lapse_rate_correction": cfg.get("lapse_rate_correction", True)}
    vhash = _hash({"truth": truth, "verif": verif})

    # A run id (env_id/run_hash) comes from a recursive registration of the nested
    # forecaster, environment and checkpoint, which is not reproduced here — the
    # run file is globbed instead, at the exact verif hash.
    runs = {
        "/".join(p.parent.relative_to(data / "runs").parts): p
        for p in data.glob(f"runs/**/verif_aggregated_{vhash}.nc")
    }
    if not runs:
        raise SystemExit(
            f"No run verification file under {data / 'runs'} matches hash {vhash}.\n"
            "Produce it with `evalml make <config> experiment_all`."
        )
    if run_source is None and len(runs) > 1:
        raise SystemExit(
            "Several candidate runs; pick one with --run-source:\n  "
            + "\n  ".join(sorted(runs))
        )
    if run_source is not None and run_source not in runs:
        raise SystemExit(
            f"--run-source {run_source!r} not found. Available:\n  "
            + "\n  ".join(sorted(runs))
        )
    source_id = run_source or next(iter(runs))
    sources = [("run", runs[source_id], source_id)]

    baseline_ids = {
        b["label"]: "baseline-"
        + _hash({k: v for k, v in b.items() if k not in _HASH_EXCLUDE})
        for entry in cfg["runs"]
        if (b := entry.get("baseline"))
    }
    for name, label in ((f"{lower} (0%)", lower), (f"{upper} (100%)", upper)):
        if label not in baseline_ids:
            raise SystemExit(
                f"Baseline {label!r} is not in the config. Available: {sorted(baseline_ids)}"
            )
        path = data / "baselines" / baseline_ids[label] / f"verif_aggregated_{vhash}.nc"
        if not path.exists():
            raise SystemExit(
                f"No verification file for {label}: {path}\n"
                "Produce it with `evalml make <config> experiment_all`; if it exists "
                "under another name, the hashing in common.smk has changed."
            )
        sources.append((name, path, baseline_ids[label]))
    return sources


# ── Loading ───────────────────────────────────────────────────────────────────
def _load_source(path: Path, source: str, lead_time_h: int) -> xr.Dataset:
    """One source at one lead time, reduced to a region-only dataset."""
    with xr.open_dataset(path) as ds:
        try:
            out = ds.sel(
                source=source, step=pd.Timedelta(hours=lead_time_h), **_STRAT_ALL
            )
        except KeyError:
            hours = [
                int(pd.Timedelta(s).total_seconds() / 3600) for s in ds["step"].values
            ]
            raise SystemExit(
                f"Cannot select source={source!r} at lead time {lead_time_h}h from {path}.\n"
                f"The file carries sources {np.atleast_1d(ds['source'].values).tolist()} "
                f"at lead times (h) {hours}."
            ) from None
        return out.squeeze(drop=True).load()


def _check_same_dates(sources: list) -> None:
    """A position only means something if the three sources cover the same dates."""
    counts = {
        name: int(ds["n_samples"].item()) for name, ds in sources if "n_samples" in ds
    }
    if len(set(counts.values())) > 1:
        raise SystemExit(
            f"The sources cover a different number of init times: {counts}.\n"
            "Re-run the verification so all three cover the same dates."
        )


# ── Table ─────────────────────────────────────────────────────────────────────
def _parse_var_metrics(spec: str) -> tuple[str, list[str]]:
    var, _, metrics = spec.partition(":")
    return var.strip(), [m.strip() for m in metrics.split(",") if m.strip()]


def _resolve_metric(var: str, metric: str, data_vars) -> list[str]:
    """Names matching ``var.metric``, or the threshold-suffixed ``var.metric_*``."""
    exact = f"{var}.{metric}"
    if exact in data_vars:
        return [exact]
    return [v for v in data_vars if v.startswith(f"{var}.{metric}_")]


def build_table(sources: list, variables: dict) -> pd.DataFrame:
    (_, run_ds), (_, lower_ds), (_, upper_ds) = sources
    regions = run_ds["region"].values.tolist()
    regions = (["all"] if "all" in regions else []) + [r for r in regions if r != "all"]

    index, rows = [], []
    for var, metrics in variables.items():
        for metric in metrics:
            for name in _resolve_metric(var, metric, run_ds.data_vars):
                # Positioning needs a score from all three sources. An anchor may not
                # carry the variable at all (INCA has no PMSL), or may have no data at
                # this lead time (TOT_PREC is accumulated, so at lead time 0 the
                # forecasts have no window yet while observation-based INCA does).
                blank = [
                    src_name
                    for src_name, ds in sources
                    if name not in ds.data_vars or not np.isfinite(ds[name]).any()
                ]
                if blank:
                    print(f"skipped {name}: no data from {', '.join(blank)}")
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    pos = (
                        (run_ds[name] - lower_ds[name])
                        / (upper_ds[name] - lower_ds[name])
                        * 100
                    )
                pos = pos.where(np.isfinite(pos)).reindex(region=regions)
                var_group, raw_metric = name.split(".", 1)
                index.append((var_group, decode_metric(raw_metric)))
                rows.append(np.round(pos.values, 1))

    if not rows:
        raise SystemExit("None of the requested metrics could be positioned.")
    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(index, names=["variable", "metric"]),
        columns=regions,
    )


def main(args) -> None:
    cfg = _load_config(args.config)
    files = _resolve_sources(cfg, args.lower, args.upper, args.run_source)

    print("Using verification files:")
    for name, path, source_id in files:
        print(f"  {name:22s} source={source_id}\n{'':24s}{path}")

    sources = [
        (name, _load_source(path, sid, args.lead_time_h)) for name, path, sid in files
    ]
    _check_same_dates(sources)

    variables = (
        dict(_parse_var_metrics(s) for s in args.variable)
        if args.variable
        else DEFAULT_VARIABLES
    )
    table = build_table(sources, variables)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output)
    print(f"\nsaved: {args.output}\n")
    print(table.to_string())


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    p.add_argument(
        "--config",
        required=True,
        help="The evalml config used to produce the verification files.",
    )
    p.add_argument("--lower", default="INCA", help="Baseline label for the 0%% anchor.")
    p.add_argument(
        "--upper", default="ICON-CH1-CTRL", help="Baseline label for the 100%% anchor."
    )
    p.add_argument(
        "--run-source",
        default=None,
        help="Run id, only needed when the config has several candidate runs.",
    )
    p.add_argument(
        "--lead-time-h",
        type=int,
        default=0,
        help="Lead time to read, in hours (default 0, the analysis).",
    )
    p.add_argument(
        "--variable",
        action="append",
        default=None,
        help="VAR:M1,M2 (repeatable). Default: U_10M, V_10M and T_2M, "
        "each with RMSE, R2 and ETS.",
    )
    p.add_argument(
        "--output",
        default="output/results/analysis_verification_table_lt0.csv",
        help="Output CSV path.",
    )
    main(p.parse_args())
