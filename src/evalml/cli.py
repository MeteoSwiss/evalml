import subprocess
from pathlib import Path
from typing import Any

import yaml
import click

from evalml.config import ExperimentConfig


def run_command(command: list[str]) -> int:
    """Execute a shell command, optionally as dry-run."""
    click.echo("Launching: " + " ".join(command))
    return subprocess.run(command).returncode


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def common_options(func):
    func = click.option(
        "--dry-run", "-n", is_flag=True, help="Do not execute anything, and display what would be done."
    )(func)
    func = click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")(
        func
    )
    return func


@click.group(help="Evaluation workflows for ML experiments.")
def cli():
    pass


@cli.command(help="Launch an experiment defined by a config YAML file.")
@click.argument(
    "configfile", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--cores",
    "-c",
    default=4,
    type=int,
    help="Number of cores to use for local execution.",
)
@common_options
def experiment(
    configfile: Path,
    cores: int | None = None,
    verbose: bool = False,
    dry_run: bool = False,
):
    """Run an ML experiment defined in the given config file."""
    config = load_yaml(configfile)

    # Validate the config against the ExperimentConfig model
    config = ExperimentConfig.model_validate(config)

    command = ["snakemake"]
    command += config.profile.parsable()
    command += ["--configfile", str(configfile)]
    command += ["--cores", str(cores)]

    # Execute dry snakemake run if set
    if dry_run:
        command.append("--dry-run")

    # Add global options if set
    if verbose:
        command += ["--printshellcmds"]

    raise SystemExit(run_command(command))
