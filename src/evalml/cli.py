import subprocess
from pathlib import Path
from typing import Any

import yaml
import click

from evalml.config import ConfigModel


def run_command(command: list[str]) -> int:
    """Execute a shell command, optionally as dry-run."""
    click.echo("Launching: " + " ".join(command))
    return subprocess.run(command).returncode


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def workflow_options(func):
    """Decorator to apply common CLI options."""
    func = click.option(
        "--dry-run", "-n", is_flag=True, help="Do not execute anything."
    )(func)
    func = click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")(
        func
    )
    func = click.option(
        "--cores", "-c", default=4, type=int, help="Number of cores to use."
    )(func)
    func = click.option(
        "--report",
        default=None,
        required=False,
        metavar="FILE",
        type=click.Path(path_type=Path),
        help="Create a self-contained HTML report.",
    )(func)
    return func


def execute_workflow(
    configfile: Path,
    target: str,
    cores: int,
    verbose: bool,
    dry_run: bool,
    report: Path | None,
):
    config = ConfigModel.model_validate(load_yaml(configfile))

    command = ["snakemake"]
    command += config.profile.parsable()
    command += ["--configfile", str(configfile)]
    command += ["--cores", str(cores)]

    if dry_run:
        command.append("--dry-run")
    if verbose:
        command.append("--printshellcmds")
    if report and not dry_run:
        command += ["--report-after-run", "--report", str(report)]

    command.append(target)
    raise SystemExit(run_command(command))


@click.group(help="Evaluation workflows for ML experiments.")
def cli():
    pass


@cli.command(help="Launch an experiment defined by a config YAML file.")
@click.argument(
    "configfile", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@workflow_options
def experiment(configfile, cores, verbose, dry_run, report):
    execute_workflow(configfile, "experiment_all", cores, verbose, dry_run, report)


@cli.command(help="Obtain showcase material as defined by a config YAML file.")
@click.argument(
    "configfile", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@workflow_options
def showcase(configfile, cores, verbose, dry_run, report):
    execute_workflow(configfile, "showcase_all", cores, verbose, dry_run, report)
