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
from evalml.publication.manifest import (
    default_manifest_path,
    load_manifest,
    truth_slug,
)
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


def _load(manifest, truth=None):
    return load_manifest(manifest, truth=truth)


def _fig_dir(m, name):
    """Default standalone output dir, namespaced by truth slug: figures/<slug>/<name>."""
    slug = m.truth.get("slug") or truth_slug(m.truth.get("label", ""))
    return f"figures/{slug}/{name}"


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
    help="Manifest path (overrides --truth / discovery).",
)
truth_option = click.option(
    "--truth",
    default=None,
    metavar="LABEL",
    help="Truth label selecting which manifest to use (e.g. KENDA-CH1). "
    "Auto-detected when only one exists.",
)


@click.group(help="Render publication figures from a manifest, outside Snakemake.")
def publication():
    pass


@publication.command("list")
@manifest_option
@truth_option
@_friendly_errors
def list_(manifest, truth):
    """Show participants, truth and configured figures — no hashes needed."""
    m = _load(manifest, truth)
    t = m.truth
    click.echo(f"Manifest: {manifest or default_manifest_path(truth=truth)}")
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
        lts = sm.get("steps") or [24]
        click.echo(
            f"Scoremaps: baseline={sm.get('baseline_label')} steps={lts}h [{ok}]"
        )


@publication.command()
@manifest_option
@truth_option
@click.option(
    "--output",
    default=None,
    help="Output directory (default figures/<truth>/leadtime).",
)
@_friendly_errors
def figures(manifest, truth, output):
    """Lead-time metric curves across all participants."""
    m = _load(manifest, truth)
    output = output or _fig_dir(m, "leadtime")
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
@truth_option
@click.option("--candidate", default=None, help="Candidate label (required if >1).")
@click.option("--init-time", "init_time", default=None, help="Override init_time.")
@click.option("--station", default=None, help="Override station.")
@click.option("--params", default=None, help="Comma-separated display params.")
@click.option(
    "--output",
    default=None,
    help="Output directory (default figures/<truth>/meteogram).",
)
@_friendly_errors
def meteogram(manifest, truth, candidate, init_time, station, params, output):
    """Time-series meteogram at a station for one case."""
    m = _load(manifest, truth)
    output = output or _fig_dir(m, "meteogram")
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
@truth_option
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
@click.option(
    "--output",
    default=None,
    help="Output directory (default figures/<truth>/scoremaps).",
)
@_friendly_errors
def scoremaps(
    manifest,
    truth,
    candidate,
    baseline,
    params,
    scores,
    leadtimes,
    season,
    region,
    output,
):
    """Spatial skill-score map panel (candidate vs baseline)."""
    m = _load(manifest, truth)
    output = output or _fig_dir(m, "scoremaps")
    cfg = m.publication.get("scoremaps") or {}
    baseline = baseline or cfg.get("baseline_label", "ICON-CH1-CTRL")
    if leadtimes:
        leadtime_list = list(leadtimes)
    elif cfg.get("steps"):
        leadtime_list = list(cfg["steps"])
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


@publication.command("sal-scatter")
@manifest_option
@truth_option
@click.option("--candidate", default=None, help="Candidate label (required if >1).")
@click.option(
    "--param", default=None, help="Accumulated precip param (e.g. TOT_PREC6)."
)
@click.option(
    "--leadtime",
    "leadtimes",
    type=int,
    multiple=True,
    help="Lead time (hours) to aggregate per init (repeat; default = all SAL lead times).",
)
@click.option("--baseline", "baselines", multiple=True, help="Baseline label (repeat).")
@click.option(
    "--season-split", default=None, help="Named season split (e.g. jja-novdec)."
)
@click.option("--min-truth-mm", type=float, default=None, help="Wet-case filter (mm).")
@click.option(
    "--output",
    default=None,
    help="Output directory (default figures/<truth>/sal_scatter).",
)
@_friendly_errors
def sal_scatter(
    manifest,
    truth,
    candidate,
    param,
    leadtimes,
    baselines,
    season_split,
    min_truth_mm,
    output,
):
    """SAL Structure–Amplitude scatter (candidate + baselines, coloured by season)."""
    m = _load(manifest, truth)
    output = output or _fig_dir(m, "sal_scatter")
    cfg = m.publication.get("sal_scatter") or {}
    param = param or cfg.get("param", "TOT_PREC6")
    lt_list = (
        list(leadtimes)
        or cfg.get("leadtimes")
        or m.experiment_sal.get("leadtimes")
        or [6, 12, 18, 24, 30]
    )
    base_labels = list(baselines) or cfg.get(
        "baselines", ["ICON-CH1-CTRL", "ICON-CH2-CTRL"]
    )
    season_split = season_split or cfg.get("season_split", "jja-novdec")
    if min_truth_mm is None:
        min_truth_mm = cfg.get("min_truth_mm", 2.0)
    annotate = cfg.get("annotate", {})

    m.validate_request(
        "sal_scatter", candidate=candidate, baselines=base_labels, leadtimes=lt_list
    )
    cand = m.get_candidate(candidate)
    bases = [m.resolve_baseline(b) for b in base_labels]

    cmd = [
        sys.executable,
        SCRIPTS / "publication_sal_scatter.py",
        "--param",
        param,
        "--leadtimes",
        ",".join(str(x) for x in lt_list),
        "--season-split",
        season_split,
        "--min-truth-mm",
        str(min_truth_mm),
        "--output",
        output,
    ]
    # Candidate panel first (annotations apply to it), then baseline panels.
    for p in [cand, *bases]:
        csvs = ",".join(m.sal_path(p, param, lt) for lt in lt_list)
        cmd += ["--participant", p.label, csvs]
    if annotate:
        cmd += ["--annotate", ",".join(f"{k}={v}" for k, v in annotate.items())]
    _run(cmd)


@publication.command("case-snapshots")
@manifest_option
@truth_option
@click.option("--candidate", default=None, help="Candidate label (required if >1).")
@click.option(
    "--output",
    default=None,
    help="Output directory (default figures/<truth>/case_snapshots).",
)
@_friendly_errors
def case_snapshots(manifest, truth, candidate, output):
    """Precipitation map snapshots (candidate vs truth) for hand-picked cases."""
    m = _load(manifest, truth)
    output = output or _fig_dir(m, "case_snapshots")
    cfg = m.publication.get("case_snapshots") or {}
    cases = list(cfg.get("cases", []))
    if not cases:
        raise click.ClickException(
            "No cases configured (publication.case_snapshots.cases is empty)."
        )
    param = cfg.get("param", "TOT_PREC")
    leadtime = int(cfg.get("leadtime", 12))
    accumulation = int(cfg.get("accumulation", 6))
    domain = cfg.get("domain", "centraleurope")
    sal_param = f"TOT_PREC{accumulation}"

    m.validate_request(
        "case_snapshots", candidate=candidate, leadtime=leadtime, cases=cases
    )
    cand = m.get_candidate(candidate)

    cmd = [
        sys.executable,
        SCRIPTS / "publication_case_snapshots.py",
        "--candidate-label",
        cand.label,
        "--truth",
        m.truth.get("root"),
        "--truth-label",
        m.truth.get("label"),
        "--param",
        param,
        "--leadtime",
        str(leadtime),
        "--accumulation",
        str(accumulation),
        "--domain",
        domain,
        "--output",
        output,
    ]
    for init in cases:
        cmd += [
            "--case",
            init,
            m.grib_dir(cand, init),
            m.sal_path(cand, sal_param, leadtime),
        ]
    _run(cmd, env=_eccodes_env())


if __name__ == "__main__":
    publication()
