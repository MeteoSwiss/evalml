import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml
import click


def run_command(command: list[str], dry_run: bool = False) -> int:
    """Execute a shell command, optionally as dry-run."""
    if dry_run:
        click.echo("[dry-run] " + " ".join(command))
        return 0
    else:
        click.echo("Launching: " + " ".join(command))
        return subprocess.run(command).returncode

def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


@click.group(help="Evaluation workflows for ML experiments.")
@click.option("--dry-run", "-n", is_flag=True, help="Only print the Snakemake command.")
@click.option("--cores", "-c", type=int, help="Number of cores to use for local execution.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
@click.pass_context
def cli(ctx, dry_run, cores, verbose):
    ctx.ensure_object(dict)
    ctx.obj["dry_run"] = dry_run
    ctx.obj["cores"] = cores
    ctx.obj["verbose"] = verbose


@cli.group(help="Launch experiments and pipelines.")
@click.pass_context
def launch(ctx):
    pass


@launch.command(help="Launch an experiment defined by a config YAML file.")
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.pass_context
def experiment(ctx, config: Path):
    """Run an ML experiment defined in the given config file."""
    config_data = load_yaml(config)
    profile_config = config_data.get("profile", {})
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_profile:
        yaml.safe_dump(profile_config, temp_profile)
        temp_profile_path = temp_profile.name
    
    command = ["snakemake", "--configfile", str(config), "--profile", temp_profile_path]

    # Add global options if set
    if ctx.obj["verbose"]:
        command += ["--printshellcmds", "--reason"]

    raise SystemExit(run_command(command, dry_run=ctx.obj["dry_run"]))
