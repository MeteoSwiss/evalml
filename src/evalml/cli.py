import base64
import shlex
import subprocess
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any

import yaml
import click

from evalml.config import ConfigModel


def run_command(command: list[str]) -> int:
    """Execute a shell command, optionally as dry-run."""
    Path(".evalml_snakemake_cmd.txt").write_text(shlex.join(command) + "\n")
    return subprocess.run(command).returncode


def _base_snakemake_command(
    config: ConfigModel, configfile: Path, cores: int
) -> list[str]:
    command = ["snakemake"]
    command += config.profile.parsable()
    command += ["--configfile", str(configfile)]
    command += ["--cores", str(cores)]
    return command


def _dot_to_svg(dot_content: str) -> bytes:
    compressed = zlib.compress(dot_content.encode(), 9)
    encoded = base64.urlsafe_b64encode(compressed).decode()
    req = urllib.request.Request(
        f"https://kroki.io/graphviz/svg/{encoded}",
        headers={"User-Agent": "curl/7.68.0"},
    )
    try:
        with urllib.request.urlopen(req) as response:
            return response.read()
    except urllib.error.HTTPError as e:
        raise click.ClickException(
            f"kroki.io request failed: {e.code} {e.reason}"
        ) from e
    except urllib.error.URLError as e:
        raise click.ClickException(f"kroki.io request failed: {e.reason}") from e


def generate_graph(
    configfile: Path,
    target: str,
    graph_type: str,
    cores: int,
    extra_smk_args: tuple[str, ...] = (),
) -> None:
    config = ConfigModel.model_validate(load_yaml(configfile))
    command = _base_snakemake_command(config, configfile, cores)
    command += [f"--{graph_type}", "dot", target]
    command += list(extra_smk_args)

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        click.echo(result.stderr, err=True)
        raise SystemExit(result.returncode)

    output_file = Path(f"{graph_type}.svg")
    output_file.write_bytes(_dot_to_svg(result.stdout))
    click.echo(f"Graph saved to {output_file}")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def workflow_options(func):
    """Decorator to apply common CLI options."""

    command_name = func.__name__

    func = click.option(
        "--dry-run", "-n", is_flag=True, help="Do not execute anything."
    )(func)
    func = click.option(
        "--unlock", is_flag=True, help="Remove a lock on the working directory."
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
        is_flag=False,
        flag_value=f"{command_name}_report.html",
    )(func)
    func = click.option(
        "--dag", is_flag=True, help="Generate a DAG and save as dag.svg."
    )(func)
    func = click.option(
        "--rulegraph",
        is_flag=True,
        help="Generate a rule graph and save as rulegraph.svg.",
    )(func)
    func = click.argument(
        "extra_smk_args",
        nargs=-1,
        type=click.UNPROCESSED,
        metavar="-- [EXTRA_SMK_ARGS]",
    )(func)
    return func


def execute_workflow(
    configfile: Path,
    target: str,
    cores: int,
    verbose: bool,
    dry_run: bool,
    unlock: bool,
    report: Path | None,
    dag: bool = False,
    rulegraph: bool = False,
    extra_smk_args: tuple[str, ...] = (),
):
    if dag or rulegraph:
        generate_graph(
            configfile, target, "dag" if dag else "rulegraph", cores, extra_smk_args
        )
        return

    config = ConfigModel.model_validate(load_yaml(configfile))
    command = _base_snakemake_command(config, configfile, cores)

    if dry_run:
        command.append("--dry-run")
    if unlock:
        command.append("--unlock")
    if report and not dry_run:
        command += ["--report-after-run", "--report", str(report)]

    command += [target]
    command += list(extra_smk_args)
    if not verbose:
        command += ["--quiet", "rules"]  # reduce verobosity of snakemake output

    return run_command(command)


@click.group(help="Evaluation workflows for ML experiments.")
def cli():
    pass


@cli.command(help="Launch an experiment defined by a config YAML file.")
@click.argument(
    "configfile", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--mec",
    is_flag=True,
    default=False,
    help="Run MEC for Feedback file production.",
)
@click.option(
    "--ffv2", is_flag=True, default=False, help="Run the FFV2 scorefile production."
)
@workflow_options
def experiment(
    configfile,
    mec,
    ffv2,
    cores,
    verbose,
    dry_run,
    unlock,
    report,
    dag,
    rulegraph,
    extra_smk_args,
):
    if mec or ffv2:
        raw = load_yaml(configfile)
        if mec and raw.get("mec") is None:
            raise click.UsageError("--mec requires a 'mec' block in the config file.")
        if ffv2 and raw.get("ffv2") is None:
            raise click.UsageError("--ffv2 requires a 'ffv2' block in the config file.")

    targets = []
    if mec or ffv2:
        if mec:
            targets.append("mec_all")
        if ffv2:
            targets.append("ffv2_all")
    else:
        targets.append("experiment_all")

    for target in targets:
        rc = execute_workflow(
            configfile,
            target,
            cores,
            verbose,
            dry_run,
            unlock,
            report,
            dag,
            rulegraph,
            extra_smk_args,
        )
        if rc:
            raise SystemExit(rc)
    raise SystemExit(0)


@cli.command(help="Obtain showcase material as defined by a config YAML file.")
@click.argument(
    "configfile", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@workflow_options
def showcase(
    configfile, cores, verbose, dry_run, unlock, report, dag, rulegraph, extra_smk_args
):
    raise SystemExit(
        execute_workflow(
            configfile,
            "showcase_all",
            cores,
            verbose,
            dry_run,
            unlock,
            report,
            dag,
            rulegraph,
            extra_smk_args,
        )
    )


@cli.command(
    "capture-fixture",
    help="Snapshot inference GRIB output into a test fixture directory for replay.",
)
@click.argument(
    "configfile", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument("fixture_root", type=click.Path(file_okay=False, path_type=Path))
def capture_fixture_cmd(configfile: Path, fixture_root: Path) -> None:
    from datetime import datetime

    from evalml.fixtures import (
        capture_fixture,
        config_init_times,
        expected_run_prefixes,
        write_manifest,
    )

    def _evalml_commit() -> str | None:
        try:
            out = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parent,
                capture_output=True,
                text=True,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except OSError:
            return None

    # Load and validate via the same path the workflow uses, so the CLI and the
    # workflow cannot disagree on output_root.
    raw = load_yaml(configfile)
    model = ConfigModel.model_validate(raw)
    output_root = model.locations.output_root

    # Scope capture to this config's dates and its own run directories so an
    # unrelated experiment sharing the same output/ tree is not swept into the
    # fixture.
    init_times = config_init_times(raw["dates"])
    run_prefixes = expected_run_prefixes(raw.get("runs", []))

    copied = capture_fixture(
        output_root, fixture_root, init_times=init_times, run_prefixes=run_prefixes
    )
    if not copied:
        raise click.ClickException(
            f"No inference GRIB dirs found under {output_root}/data/runs for the "
            f"config's dates ({', '.join(sorted(init_times))}). "
            "Run the pipeline once (with a GPU) before capturing."
        )

    # Record every checkpoint the config references, including the nested
    # forecaster.checkpoint of a temporal_downscaler run.
    checkpoints: list[str] = []
    for entry in raw.get("runs", []):
        for kind, run in entry.items():
            if kind == "baseline" or not isinstance(run, dict):
                continue
            if run.get("checkpoint"):
                checkpoints.append(run["checkpoint"])
            nested = run.get("forecaster")
            if isinstance(nested, dict) and nested.get("checkpoint"):
                checkpoints.append(nested["checkpoint"])

    write_manifest(
        fixture_root,
        config_label=model.config_label,
        checkpoints=checkpoints,
        captured_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        grib_dirs=copied,
        dates=init_times,
        evalml_commit=_evalml_commit(),
    )
    click.echo(f"Captured {len(copied)} GRIB dir(s) into {fixture_root}")


@cli.command(help="Generate a sandbox for inference runs in the YAML config file.")
@click.argument(
    "configfile", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@workflow_options
def sandbox(
    configfile, cores, verbose, dry_run, unlock, report, dag, rulegraph, extra_smk_args
):
    raise SystemExit(
        execute_workflow(
            configfile,
            "sandbox_all",
            cores,
            verbose,
            dry_run,
            unlock,
            report,
            dag,
            rulegraph,
            extra_smk_args,
        )
    )


@cli.command(help="Make a specific file from a workflow defined in the YAML file.")
@click.argument(
    "configfile", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument("target", type=str)
@workflow_options
def make(
    configfile,
    target,
    cores,
    verbose,
    dry_run,
    unlock,
    report,
    dag,
    rulegraph,
    extra_smk_args,
):
    raise SystemExit(
        execute_workflow(
            configfile,
            target,
            cores,
            verbose,
            dry_run,
            unlock,
            report,
            dag,
            rulegraph,
            extra_smk_args,
        )
    )
