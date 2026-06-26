"""Standalone CLI for rendering publication figures outside Snakemake.

``python -m evalml.publication <figure>`` loads the manifest, resolves all data
paths and labels (no hashes typed by the user), validates the request, then
invokes the existing figure scripts in ``workflow/scripts/`` with their current
flags. This is the *same* resolution the thin Snakemake wrapper rules use, so
interactive and reproducible runs stay in lock-step.
"""

import functools
import os
import subprocess
import sys
from pathlib import Path

import click

from evalml.config import PROJECT_ROOT
from evalml.publication.manifest import default_manifest_path, load_manifest
from evalml.publication.resolver import ResolutionError

SCRIPTS = PROJECT_ROOT / "workflow" / "scripts"


def _friendly_errors(func):
    """Turn resolution/lookup errors into clean CLI messages (no traceback)."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ResolutionError, FileNotFoundError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc

    return wrapper


def _load(manifest):
    return load_manifest(manifest)


def _run(cmd, env=None) -> None:
    """Run a figure script, streaming output; abort the CLI on failure."""
    click.echo("→ " + " ".join(str(c) for c in cmd))
    result = subprocess.run([str(c) for c in cmd], env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _eccodes_env() -> dict:
    """Replicate the meteogram rule's ECCODES_DEFINITION_PATH export, if present."""
    env = dict(os.environ)
    defs = Path(".venv/share/eccodes-cosmo-resources/definitions")
    if defs.exists():
        env["ECCODES_DEFINITION_PATH"] = str(defs.resolve())
    return env


manifest_option = click.option(
    "--manifest",
    default=None,
    metavar="PATH",
    help="Manifest path (default: $EVALML_MANIFEST or output/publication/manifest.json).",
)


@click.group(help="Render publication figures from a manifest, outside Snakemake.")
def publication():
    pass


@publication.command("list")
@manifest_option
@_friendly_errors
def list_(manifest):
    """Show participants, truth and configured figures — no hashes needed."""
    m = _load(manifest)
    t = m.truth
    click.echo(f"Manifest: {manifest or default_manifest_path()}")
    click.echo(f"Truth:    {t.get('label')} ({t.get('type')}, hash {t.get('hash')})")
    click.echo("Participants:")
    for p in m.participants():
        flag = "candidate" if p.role == "candidate" else "baseline "
        member = f" member={p.member}" if p.member else ""
        click.echo(f"  [{flag}] {p.label:<22} steps={p.steps}{member}")
    pub = m.publication
    if pub.get("meteogram"):
        mg = pub["meteogram"]
        click.echo(
            f"Meteogram case: station={mg.get('station')} init_time={mg.get('init_time')}"
        )
    sm = pub.get("scoremaps")
    if sm and sm.get("enabled"):
        ok = "ok" if t.get("type") == "zarr" else "INVALID (needs zarr truth)"
        lts = sm.get("leadtimes") or (
            [sm["leadtime"]] if sm.get("leadtime") is not None else [24]
        )
        click.echo(
            f"Scoremaps: baseline={sm.get('baseline_label')} leadtimes={lts}h [{ok}]"
        )


@publication.command()
@manifest_option
@click.option("--output", default="figures/leadtime", help="Output directory.")
@_friendly_errors
def figures(manifest, output):
    """Lead-time metric curves across all participants."""
    m = _load(manifest)
    m.validate_request("figures")
    pairs = m.verif_paths()
    verif = " ".join(path for path, _ in pairs)
    sources = ",".join(label for _, label in pairs)
    _run(
        [
            sys.executable,
            SCRIPTS / "publication_figures.py",
            "--verif_files",
            verif,
            "--sources",
            sources,
            "--output",
            output,
        ]
    )


@publication.command()
@manifest_option
@click.option("--candidate", default=None, help="Candidate label (required if >1).")
@click.option("--init-time", "init_time", default=None, help="Override init_time.")
@click.option("--station", default=None, help="Override station.")
@click.option("--params", default=None, help="Comma-separated display params.")
@click.option("--output", default="figures/meteogram", help="Output directory.")
@_friendly_errors
def meteogram(manifest, candidate, init_time, station, params, output):
    """Time-series meteogram at a station for one case."""
    m = _load(manifest)
    cfg = m.publication.get("meteogram") or {}
    init_time = init_time or cfg.get("init_time")
    station = station or cfg.get("station")
    if params is None:
        params = ",".join(cfg.get("params", ["T_2M", "TOT_PREC", "SP_10M", "DD_10M"]))
    if not init_time or not station:
        raise click.ClickException(
            "init_time and station are required (from --init-time/--station or the "
            "manifest's publication.meteogram block)."
        )
    m.validate_request("meteogram", init_time=init_time, candidate=candidate)
    cand = m.get_candidate(candidate)
    cmd = [
        sys.executable,
        SCRIPTS / "publication_meteogram.py",
        "--forecast",
        m.grib_dir(cand, init_time),
        "--forecast_steps",
        cand.steps,
        "--forecast_label",
        cand.label,
        "--date",
        init_time,
        "--station",
        station,
        "--params",
        params,
        "--output",
        output,
    ]
    baselines = m.meteogram_baseline_specs()
    if baselines:
        cmd += ["--baseline", baselines]
    _run(cmd, env=_eccodes_env())


@publication.command()
@manifest_option
@click.option("--candidate", default=None, help="Candidate label (required if >1).")
@click.option("--baseline", "baseline", default=None, help="Baseline label.")
@click.option("--params", default=None, help="Comma-separated params (one row each).")
@click.option("--scores", default=None, help="Comma-separated derived metrics.")
@click.option(
    "--leadtime",
    "leadtimes",
    type=int,
    multiple=True,
    help="Lead time in hours (repeat for several; one figure each).",
)
@click.option("--season", default=None)
@click.option("--region", default=None)
@click.option("--output", default="figures/scoremaps", help="Output directory.")
@_friendly_errors
def scoremaps(
    manifest, candidate, baseline, params, scores, leadtimes, season, region, output
):
    """Spatial skill-score map panel (candidate vs baseline)."""
    m = _load(manifest)
    cfg = m.publication.get("scoremaps") or {}
    baseline = baseline or cfg.get("baseline_label", "ICON-CH1-CTRL")
    if leadtimes:
        leadtime_list = list(leadtimes)
    elif cfg.get("leadtimes"):
        leadtime_list = list(cfg["leadtimes"])
    elif cfg.get("leadtime") is not None:
        leadtime_list = [cfg["leadtime"]]
    else:
        leadtime_list = [24]
    season = season or cfg.get("season", "all")
    region = region or cfg.get("region", "switzerland")
    param_list = (
        [p.strip() for p in params.split(",")]
        if params
        else cfg.get("params", ["T_2M", "SP_10M"])
    )
    score_str = scores or ",".join(cfg.get("scores", ["MSE_SKILL", "BIAS_CONTRIB"]))

    cand = m.get_candidate(candidate)
    base = m.resolve_baseline(baseline)
    for lt in leadtime_list:
        m.validate_request(
            "scoremaps", candidate=candidate, baseline=baseline, leadtime=lt
        )
    # Files ordered leadtime-major so the script slices them by n_params.
    cand_files = [
        m.scoremap_path(cand, p, lt) for lt in leadtime_list for p in param_list
    ]
    base_files = [
        m.scoremap_path(base, p, lt) for lt in leadtime_list for p in param_list
    ]
    _run(
        [
            sys.executable,
            SCRIPTS / "publication_scoremaps.py",
            "--candidate_files",
            *cand_files,
            "--baseline_files",
            *base_files,
            "--params",
            ",".join(param_list),
            "--scores",
            score_str,
            "--candidate_label",
            cand.label,
            "--baseline_label",
            base.label,
            "--leadtimes",
            *[str(lt) for lt in leadtime_list],
            "--season",
            season,
            "--region",
            region,
            "--output",
            output,
        ]
    )


if __name__ == "__main__":
    publication()
