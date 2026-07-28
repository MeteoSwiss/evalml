"""Produce and consume frozen inference GRIB fixtures for tests/dev.

The fixture mirrors the pipeline's own output layout
``<root>/data/runs/<run_id>/<init_time>/grib`` so that capture (writing the
fixture from a real run) and replay (reading it back) agree by construction.
"""

import shutil
from pathlib import Path

import yaml


def fixture_grib_dir(fixture_root, run_id: str, init_time) -> Path:
    """Return the frozen GRIB directory for one run/init inside a fixture.

    ``run_id`` contains a '/' (``<env_id>/<run_hash>``) and is used verbatim.
    """
    return Path(fixture_root) / "data" / "runs" / run_id / str(init_time) / "grib"


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


def capture_fixture(output_root, fixture_root) -> list[Path]:
    """Copy every inference GRIB dir under ``output_root`` into ``fixture_root``.

    Preserves the relative path so the result is readable via
    :func:`fixture_grib_dir`. Overwrites any existing destination. Returns the
    list of destination directories.
    """
    output_root = Path(output_root)
    fixture_root = Path(fixture_root)
    copied: list[Path] = []
    for grib in iter_grib_dirs(output_root):
        dest = fixture_root / grib.relative_to(output_root)
        # Defensive: never delete-then-copy a source that already resolves to
        # its own destination (e.g. capturing onto the fixture itself), which
        # would wipe the source before the copy.
        if grib.resolve() == dest.resolve():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(grib, dest)
        copied.append(dest)
    return copied


def write_manifest(
    fixture_root, *, config_label, checkpoints, captured_at, grib_dirs
) -> Path:
    """Write MANIFEST.yaml recording what was frozen (provenance only)."""
    manifest = {
        "config_label": config_label,
        "checkpoints": list(checkpoints),
        "captured_at": captured_at,
        "grib_dirs": [str(p) for p in grib_dirs],
    }
    path = Path(fixture_root) / "MANIFEST.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest, sort_keys=True))
    return path
