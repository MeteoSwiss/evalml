"""Produce and consume frozen inference GRIB fixtures for tests/dev.

The fixture mirrors the pipeline's own output layout
``<root>/data/runs/<run_id>/<init_time>/grib`` so that capture (writing the
fixture from a real run) and replay (reading it back) agree by construction.
"""

import hashlib
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import yaml

_DATETIME_FORMAT = "%Y-%m-%dT%H:%M"
_INIT_TIME_FORMAT = "%Y%m%d%H%M"
_CHUNK = 1 << 20  # 1 MiB


def grib_checksum(grib_dir) -> str:
    """Deterministic SHA-256 over a GRIB directory's file names + contents.

    Order-independent (files are sorted) and content-sensitive, so it detects a
    corrupted, partial, or hand-modified fixture. Files are read in chunks so a
    multi-GB GRIB dir is not loaded into memory.
    """
    grib_dir = Path(grib_dir)
    h = hashlib.sha256()
    for f in sorted(p for p in grib_dir.rglob("*") if p.is_file()):
        h.update(f.relative_to(grib_dir).as_posix().encode())
        h.update(b"\0")
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(_CHUNK), b""):
                h.update(chunk)
    return h.hexdigest()


def verify_fixture(fixture_root, grib_dir) -> None:
    """Raise ``ValueError`` if ``grib_dir`` disagrees with the manifest checksum.

    A no-op when there is no ``MANIFEST.yaml`` or no recorded checksum for
    ``grib_dir`` (fixtures captured before checksums existed, or built by hand),
    so it never blocks a legitimate replay — it only fails a fixture that was
    checksummed at capture and has since drifted.
    """
    fixture_root = Path(fixture_root)
    manifest_path = fixture_root / "MANIFEST.yaml"
    if not manifest_path.exists():
        return
    manifest = yaml.safe_load(manifest_path.read_text()) or {}
    checksums = manifest.get("grib_checksums") or {}
    rel = Path(grib_dir).resolve().relative_to(fixture_root.resolve()).as_posix()
    expected = checksums.get(rel)
    if expected is None:
        return
    actual = grib_checksum(grib_dir)
    if actual != expected:
        raise ValueError(
            f"Fixture GRIB at {grib_dir} does not match its recorded checksum "
            f"(expected {expected[:12]}…, got {actual[:12]}…). The fixture is "
            "corrupted or stale; re-capture it with `evalml capture-fixture`."
        )


def _parse_timedelta(td: str) -> timedelta:
    magnitude, unit = int(td[:-1]), td[-1]
    if unit == "d":
        return timedelta(days=magnitude)
    if unit == "h":
        return timedelta(hours=magnitude)
    raise ValueError(f"Unsupported time unit: {unit!r} (only 'd' and 'h')")


def config_init_times(dates_cfg) -> set[str]:
    """Init-time path segments (``YYYYMMDDHHMM``) for a config's ``dates`` block.

    Accepts either an explicit list of ``YYYY-MM-DDTHH:MM`` strings or a
    ``{start, end, frequency, blacklist?}`` range, mirroring the workflow's
    reference-time parsing so that capture can be scoped to the dates the
    given config actually produces.
    """
    if isinstance(dates_cfg, list):
        times = [datetime.strptime(t, _DATETIME_FORMAT) for t in dates_cfg]
    else:
        start = datetime.strptime(dates_cfg["start"], _DATETIME_FORMAT)
        end = datetime.strptime(dates_cfg["end"], _DATETIME_FORMAT)
        freq = _parse_timedelta(dates_cfg["frequency"])
        blacklist = {
            datetime.strptime(t, _DATETIME_FORMAT)
            for t in dates_cfg.get("blacklist", [])
        }
        times = []
        t = start
        while t <= end:
            if t not in blacklist:
                times.append(t)
            t += freq
    return {t.strftime(_INIT_TIME_FORMAT) for t in times}


def fixture_grib_dir(fixture_root, run_id: str, init_time) -> Path:
    """Return the frozen GRIB directory for one run/init inside a fixture.

    ``run_id`` contains a '/' (``<env_id>/<run_hash>``) and is used verbatim.

    The result is absolute (``.resolve()``): the replay rule symlinks this path
    into a deep workdir, so a relative ``fixture_root`` would otherwise produce a
    dangling link (``exists()`` resolves against the launch cwd, ``ln -sfn``
    against the workdir).
    """
    return (
        Path(fixture_root) / "data" / "runs" / run_id / str(init_time) / "grib"
    ).resolve()


def iter_grib_dirs(output_root) -> list[Path]:
    """Every real ``grib/`` directory under ``<output_root>/data/runs``.

    Symlinked ``grib`` dirs are skipped: after a replay run, the workdir's
    ``grib`` is a symlink *into the fixture*, and capturing it would resolve
    back onto (and destroy) the fixture it points at. Only genuine inference
    output is ever a real directory here.
    """
    runs = Path(output_root) / "data" / "runs"
    if not runs.is_dir():
        return []
    return sorted(p for p in runs.rglob("grib") if p.is_dir() and not p.is_symlink())


def _checkpoint_model_id(checkpoint_uri: str) -> str:
    """Short checkpoint id, mirroring ``model_id()`` in workflow/rules/common.smk.

    Only used to recognize a run directory's name (``<model_type>-<model_id>-...``)
    as belonging to a given config's checkpoint; kept independent from the
    workflow's own hashing so fixture capture has no dependency on it.
    """
    parsed = urlparse(checkpoint_uri)
    if parsed.netloc in (
        "mlflow.ecmwf.int",
        "service.meteoswiss.ch",
        "servicedepl.meteoswiss.ch",
    ):
        fragment = checkpoint_uri.split("#")[-1]
        if "/models/" in fragment:
            parts = fragment.strip("/").split("/")
            if len(parts) >= 4 and parts[2] == "versions":
                return f"{parts[1]}-v{parts[3]}"[:4]
            return f"{parts[1]}-latest"[:4]
        return checkpoint_uri.split("/")[-1][:4]
    elif parsed.netloc == "huggingface.co":
        return checkpoint_uri.split("/")[-1].split(".")[0]
    else:
        return checkpoint_uri.split("/")[-2][:4]


def expected_run_prefixes(runs_cfg: list) -> set[str]:
    """Directory-name prefixes a config's ``runs:`` block should produce.

    Each ML run directory is named ``<model_type>-<model_id>-...`` (see
    ``register_run()`` in workflow/rules/common.smk); this returns the
    ``<model_type>-<model_id>-`` prefix for every checkpoint the config
    references, including a temporal_downscaler's nested forecaster.
    Baselines are skipped: they read from an archive, not a run directory.
    """
    prefixes = set()
    for run_entry in runs_cfg:
        model_type = next(iter(run_entry))
        if model_type == "baseline":
            continue
        run_config = run_entry[model_type]
        prefixes.add(f"{model_type}-{_checkpoint_model_id(run_config['checkpoint'])}-")
        nested = run_config.get("forecaster")
        if isinstance(nested, dict) and nested.get("checkpoint"):
            prefixes.add(f"forecaster-{_checkpoint_model_id(nested['checkpoint'])}-")
    return prefixes


def capture_fixture(
    output_root, fixture_root, *, init_times=None, run_prefixes=None
) -> list[Path]:
    """Copy inference GRIB dirs under ``output_root`` into ``fixture_root``.

    Preserves the relative path so the result is readable via
    :func:`fixture_grib_dir`. Overwrites any existing destination. Returns the
    list of destination directories.

    ``init_times``: when given (a set of ``YYYYMMDDHHMM`` strings, e.g. from
    :func:`config_init_times`), only GRIB dirs for those init times are
    captured.

    ``run_prefixes``: when given (from :func:`expected_run_prefixes`), only
    GRIB dirs whose run directory name starts with one of these prefixes are
    captured. Combined with ``init_times``, this scopes capture to one
    config's own runs so leftover output from an unrelated experiment sharing
    the same ``output/`` tree and init time is not swept in.
    """
    output_root = Path(output_root)
    fixture_root = Path(fixture_root)
    runs_root = output_root / "data" / "runs"
    copied: list[Path] = []
    for grib in iter_grib_dirs(output_root):
        if init_times is not None and grib.parent.name not in init_times:
            continue
        if run_prefixes is not None:
            env_id = grib.relative_to(runs_root).parts[0]
            if not any(env_id.startswith(p) for p in run_prefixes):
                continue
        dest = (fixture_root / grib.relative_to(output_root)).resolve()
        # Defensive: never delete-then-copy a source that already resolves to
        # its own destination (e.g. capturing onto the fixture itself), which
        # would wipe the source before the copy.
        if grib.resolve() == dest.resolve():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(grib, dest)
        # copytree preserves the source mtime, which would leave the fixture
        # older than the capture run's .ok file and make snakemake's rerun
        # decisions depend on capture history. Freshen to "now" for determinism.
        os.utime(dest, None)
        for child in dest.rglob("*"):
            os.utime(child, None)
        copied.append(dest)
    return copied


def write_manifest(
    fixture_root,
    *,
    config_label,
    checkpoints,
    captured_at,
    grib_dirs,
    dates=None,
    evalml_commit=None,
) -> Path:
    """Write MANIFEST.yaml: provenance plus per-GRIB checksums for replay-time
    integrity checks.

    ``grib_checksums`` maps each captured GRIB dir's path relative to
    ``fixture_root`` to its :func:`grib_checksum`, which :func:`verify_fixture`
    re-checks at replay. ``evalml_commit`` is recorded as provenance only.
    """
    fixture_root = Path(fixture_root)
    grib_checksums = {
        Path(d).resolve().relative_to(fixture_root.resolve()).as_posix(): grib_checksum(
            d
        )
        for d in grib_dirs
    }
    manifest = {
        "config_label": config_label,
        "checkpoints": list(checkpoints),
        "captured_at": captured_at,
        "evalml_commit": evalml_commit,
        "grib_dirs": [str(p) for p in grib_dirs],
        "grib_checksums": grib_checksums,
    }
    if dates is not None:
        manifest["dates"] = sorted(dates)
    path = fixture_root / "MANIFEST.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest, sort_keys=True))
    return path
