"""Script to prepare configuration and working directory for inference runs."""

import logging
import yaml
import shutil
from pathlib import Path

from evalml.helpers import setup_logger


def prepare_config(default_config_path: str, output_config_path: str, params: dict):
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
    """

    with open(default_config_path, "r") as f:
        config = yaml.safe_load(f)

    config = _override_recursive(config, params)

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
    shutil.copytree(resources_root / "templates", workdir / "resources")


def prepare_interpolator(smk):
    """Prepare the interpolator for the inference run.

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
    prepare_config(smk.input.config, smk.output.config, overrides)
    LOG.info("Wrote config file at %s", smk.output.config)
    with open(smk.output.config, "r") as f:
        config_content = f.read()
    LOG.info("Config: \n%s", config_content)

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

    LOG.info("Forecaster preparation complete.")


# TODO: just pass a dictionary of config overrides to the rule's params
def _overrides_from_params(smk) -> dict:
    return {
        "checkpoint": f"{smk.params.checkpoint_path}",
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


def main(smk):
    """Main function to run the Snakemake workflow."""
    if smk.rule == "prepare_inference_forecaster":
        prepare_forecaster(smk)
    elif smk.rule == "prepare_inference_interpolator":
        prepare_interpolator(smk)
    else:
        raise ValueError(f"Unknown rule: {smk.rule}")


if __name__ == "__main__":
    snakemake = snakemake  # type: ignore # noqa: F821
    raise SystemExit(main(snakemake))
