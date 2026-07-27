"""Produce and consume frozen inference GRIB fixtures for tests/dev.

The fixture mirrors the pipeline's own output layout
``<root>/data/runs/<run_id>/<init_time>/grib`` so that capture (writing the
fixture from a real run) and replay (reading it back) agree by construction.
"""

from pathlib import Path


def fixture_grib_dir(fixture_root, run_id: str, init_time) -> Path:
    """Return the frozen GRIB directory for one run/init inside a fixture.

    ``run_id`` contains a '/' (``<env_id>/<run_hash>``) and is used verbatim.
    """
    return Path(fixture_root) / "data" / "runs" / run_id / str(init_time) / "grib"
