"""Warm the shared cartopy Natural Earth shapefile cache once, up front.

The scoremap plots draw land, coastlines and country borders via
earthkit-plots -> cartopy. cartopy downloads the Natural Earth shapefiles on
first use into its cache under ``$HOME`` and reuses them afterwards. When that
cache is cold and many plot jobs start in parallel, they all race to download
the same files at once and leave a truncated shapefile behind, which then
fails on read with ``struct.error: unpack requires a buffer of N bytes``.

In CI this happens on every pipeline because ``$HOME`` is recreated fresh each
run (see ``ci/cscs.yml``), so the cache is always cold. On a normal run with a
persistent ``$HOME`` the cache is already warm and this script is a no-op.

Running this once, serially, before the parallel plot fan-out downloads each
file exactly once so every plot job finds it already present. earthkit-plots
resolves shapefiles through the very same ``cartopy.io.shapereader.natural_earth``
call, so warming cartopy's cache is exactly what the plot jobs go on to read.
"""

import sys
import time

import cartopy.io.shapereader as shpreader

# (category, name) for each Natural Earth layer the scoremaps draw, at the 50m
# resolution earthkit-plots requests for these domains (the only resolution
# seen in the plot job logs: ne_50m_land / ne_50m_coastline /
# ne_50m_admin_0_boundary_lines_land).
LAYERS = [
    ("physical", "coastline"),
    ("physical", "land"),
    ("cultural", "admin_0_boundary_lines_land"),
]
RESOLUTION = "50m"
RETRIES = 3


def _warm(category: str, name: str) -> str:
    """Download one layer and read it back fully, so a truncated download
    fails here (in this single serial job) instead of in a parallel plot job."""
    last_err: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            path = shpreader.natural_earth(
                resolution=RESOLUTION, category=category, name=name
            )
            n = sum(1 for _ in shpreader.Reader(path).records())
            print(
                f"warmed {category}/{name} ({RESOLUTION}): {n} records -> {path}",
                flush=True,
            )
            return path
        except Exception as err:  # noqa: BLE001 - re-raised below after retries
            last_err = err
            print(
                f"attempt {attempt}/{RETRIES} for {category}/{name} failed: {err}",
                flush=True,
            )
            time.sleep(2 * attempt)
    raise RuntimeError(
        f"failed to warm {category}/{name} ({RESOLUTION}) after {RETRIES} attempts"
    ) from last_err


def main() -> int:
    for category, name in LAYERS:
        _warm(category, name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
