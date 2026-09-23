"""Script to prepare configuration and working directory for inference runs."""

import logging
import yaml
import shutil
from pathlib import Path

from evalml.helpers import setup_logger


def prepare_config(
    default_config_path: str,
    output_config_path: str,
    params: dict,
    station_holdout_cfg: dict | None = None,
):
    """Prepare the configuration file for the inference run.

    Overrides default configuration parameters with those provided in params
    and writes the updated configuration to output_config_path.

    Parameters
    ----------
    default_config_path : str
        Path to the default configuration file.
    output_config_path : str
        Path where the updated configuration file will be written.
    params : dict
        Dictionary of parameters to override in the default configuration.
    station_holdout_cfg : dict, optional
        The experiment's ``experiment.station_holdout`` settings. Only used if
        the config contains a ``forward_transform_filter:
        nudge_toward_observation`` block, in which case its
        ``exclude_stations``/``holdout_fraction``/``holdout_seed`` are
        injected into that block — see ``_inject_nudging_station_holdout``.
        Configs without a nudging filter are left untouched.
    """

    with open(default_config_path, "r") as f:
        config = yaml.safe_load(f)

    config = _override_recursive(config, params)
    nudging_filter = _find_nudging_filter(config)
    if nudging_filter is not None:
        _inject_nudging_station_holdout(nudging_filter, station_holdout_cfg or {})

    with open(output_config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def prepare_workdir(workdir: Path, resources_root: Path):
    """Prepare the working directory for the inference run.

    Creates necessary subdirectories and copies resource files.

    Parameters
    ----------
    workdir : Path
        Path to the working directory.
    resources_root : Path
        Path to the root directory containing resource files.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "grib").mkdir(parents=True, exist_ok=True)
    (workdir / "resources").mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        resources_root / "templates", workdir / "resources", dirs_exist_ok=True
    )
    shutil.copytree(
        resources_root / "metadata", workdir / "resources", dirs_exist_ok=True
    )


def prepare_temporal_downscaler(smk):
    """Prepare the temporal downscaler for the inference run.

    Required steps:
    - prepare working directory
    - prepare forecaster directory
    - prepare config
    """
    LOG = _setup_logger(smk)

    # prepare working directory
    workdir = _get_workdir(smk)
    prepare_workdir(workdir, smk.params.resources_root)
    LOG.info("Prepared working directory at %s", workdir)
    res_list = "\n".join([str(fn) for fn in Path(workdir / "resources").rglob("*")])
    LOG.info("Resources: \n%s", res_list)

    # prepare forecaster directory
    fct_run_id = smk.params.forecaster_run_id
    if fct_run_id != "null":
        fct_workdir = (
            smk.params.output_root / "runs" / fct_run_id / smk.wildcards.init_time
        )
        (workdir / "forecaster").symlink_to(fct_workdir / "grib")
        LOG.info(
            "Created symlink to forecaster grib directory at %s", workdir / "forecaster"
        )
    else:
        (workdir / "forecaster").mkdir(parents=True, exist_ok=True)
        (workdir / "forecaster/.dataset").touch()
        LOG.info(
            "No forecaster run ID provided; using dataset placeholder at %s",
            workdir / "forecaster/.dataset",
        )

    # prepare config
    overrides = _overrides_from_params(smk)
    station_holdout_cfg = getattr(smk.params, "station_holdout_cfg", None)
    prepare_config(
        smk.input.config,
        smk.output.config,
        overrides,
        station_holdout_cfg=station_holdout_cfg,
    )

    LOG.info("Wrote config file at %s", smk.output.config)
    with open(smk.output.config, "r") as f:
        config_content = f.read()
    LOG.info("Config: \n%s", config_content)

    okfile = Path(smk.output.okfile)
    okfile.parent.mkdir(parents=True, exist_ok=True)
    okfile.touch()
    LOG.info("Interpolator preparation complete.")


def prepare_forecaster(smk):
    """Prepare the forecaster for the inference run.

    Required steps:
    - prepare working directory
    - prepare config
    """
    LOG = _setup_logger(smk)

    workdir = _get_workdir(smk)
    prepare_workdir(workdir, smk.params.resources_root)
    LOG.info("Prepared working directory at %s", workdir)
    res_list = "\n".join([str(fn) for fn in Path(workdir / "resources").rglob("*")])
    LOG.info("Resources: \n%s", res_list)

    overrides = _overrides_from_params(smk)
    prepare_config(smk.input.config, smk.output.config, overrides)
    LOG.info("Wrote config file at %s", smk.output.config)
    with open(smk.output.config, "r") as f:
        config_content = f.read()
    LOG.info("Config: \n%s", config_content)

    okfile = Path(smk.output.okfile)
    okfile.parent.mkdir(parents=True, exist_ok=True)
    okfile.touch()
    LOG.info("Forecaster preparation complete.")


# TODO: just pass a dictionary of config overrides to the rule's params
def _overrides_from_params(smk) -> dict:
    return {
        "checkpoint": str(Path(smk.input.checkpoint).resolve()),
        "date": smk.params.reftime_to_iso,
        "lead_time": smk.params.lead_time,
    }


def _get_workdir(smk) -> Path:
    run_id = smk.wildcards.run_id
    init_time = smk.wildcards.init_time
    return smk.params.output_root / "runs" / run_id / init_time


def _setup_logger(smk) -> logging.Logger:
    run_id = smk.wildcards.run_id
    init_time = smk.wildcards.init_time
    logger_name = f"{smk.rule}_{run_id}_{init_time}"
    LOG = setup_logger(logger_name, log_file=smk.log[0])
    return LOG


def _override_recursive(original: dict, updates: dict) -> dict:
    """Recursively override values in the original dictionary with those from the updates dictionary."""
    for key, value in updates.items():
        if (
            isinstance(value, dict)
            and key in original
            and isinstance(original[key], dict)
        ):
            original[key] = _override_recursive(original[key], value)
        else:
            original[key] = value
    return original


def _find_nudging_filter(config: dict) -> dict | None:
    """Return the ``forward_transform_filter: nudge_toward_observation`` block
    of config, or None if there is none. By design an inference config
    declares at most one nudging filter; more than one raises ValueError.
    Unlike _override_recursive (dict-into-dict only), this walks into lists
    too, since the filter typically sits inside a pre_processors list.
    """
    found = []

    def _walk(node):
        if isinstance(node, dict):
            transform_filter = node.get("forward_transform_filter")
            if isinstance(transform_filter, dict) and isinstance(
                transform_filter.get("nudge_toward_observation"), dict
            ):
                found.append(transform_filter["nudge_toward_observation"])
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(config)
    if len(found) > 1:
        raise ValueError(
            f"Found {len(found)} nudge_toward_observation filters in the "
            "inference config; at most one is supported."
        )
    return found[0] if found else None


def _inject_nudging_station_holdout(
    nudging_filter: dict, station_holdout_cfg: dict
) -> None:
    """Set exclude_stations/holdout_fraction/holdout_seed on the
    nudge_toward_observation block nudging_filter, in place, from the
    experiment's station_holdout settings — so the holdout station set is
    defined once, in the experiment config, rather than hand-maintained in
    the inference config as well.

    A no-op if station_holdout_cfg has neither exclude_stations nor
    holdout_fraction set (e.g. station holdout isn't configured for this
    experiment) — any hand-written value already in the config is then left
    untouched.
    """
    exclude_stations = station_holdout_cfg.get("exclude_stations")
    holdout_fraction = station_holdout_cfg.get("holdout_fraction")
    if exclude_stations is None and holdout_fraction is None:
        return
    holdout_seed = station_holdout_cfg.get("holdout_seed", 42)

    # Mutually exclusive in NudgeTowardObservation itself — clear both
    # before setting the one station_holdout_cfg actually specifies,
    # so a stale hand-written value of the other can never linger.
    nudging_filter.pop("exclude_stations", None)
    nudging_filter.pop("holdout_fraction", None)
    if exclude_stations is not None:
        nudging_filter["exclude_stations"] = list(exclude_stations)
    else:
        nudging_filter["holdout_fraction"] = holdout_fraction
        nudging_filter["holdout_seed"] = holdout_seed


def main(smk):
    """Main function to run the Snakemake workflow."""
    if smk.rule == "inference_prepare_forecaster":
        prepare_forecaster(smk)
    elif smk.rule == "inference_prepare_temporal_downscaler":
        prepare_temporal_downscaler(smk)
    else:
        raise ValueError(f"Unknown rule: {smk.rule}")


if __name__ == "__main__":
    snakemake = snakemake  # type: ignore # noqa: F821
    raise SystemExit(main(snakemake))
