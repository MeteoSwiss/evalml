"""Paper plot: topography (z) of ICON-REA-L-CH1 with a fixed-longitude cross-section.

Loads the constant `z` field (geopotential at surface) from the realch1 anemoi
zarr, converts it to elevation [m], and renders:

  - Top: map of the elevation over the full realch1 domain, with a vertical
    line marking the cross-section longitude.
  - Bottom: elevation profile along that longitude vs latitude.

Uses the same earthkit-plots schema (Roboto font, soft grid, no axis spines,
StatePlotter + standard_layers) as `workflow/scripts/plot_forecast_frame.mo.py`.
"""

from pathlib import Path

import cartopy.crs as ccrs
import earthkit.plots as ekp  # noqa: F401 — import applies default schema (fonts, rcParams)
import numpy as np
import xarray as xr
from shapely.geometry import MultiPoint

from plotting import _PROJECTIONS, StatePlotter

REALCH1_ZARR = Path(
    "/store_new/mch/msopr/ml/datasets/"
    "mch-realch1-fdb-1km-2005-2025-1h-pl13-ifsnames-v1.0.zarr"
)
G = 9.80665  # standard gravity, m/s^2
CROSS_LON = 8.2  # fixed longitude (°E) for the cross-section through the Alps
LON_TOL = 0.005  # half-width [°] of the longitude band used for the profile
ELEVATION_LEVELS = list(range(0, 4001, 250))


def load_topography(zarr_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lon, lat, elevation [m]) for realch1 from the constant `z` field."""
    ds = xr.open_zarr(zarr_root, consolidated=False)
    variables = list(ds.attrs["variables"])
    z_idx = variables.index("z")
    geopotential = ds["data"][0, z_idx, 0, :].values  # constant_in_time -> use t=0
    elevation = geopotential / G
    lons = ds["longitudes"].values
    lats = ds["latitudes"].values
    lons = np.where(lons > 180, lons - 360, lons)
    return lons, lats, elevation


def extract_cross_section(
    lons: np.ndarray,
    lats: np.ndarray,
    field: np.ndarray,
    target_lon: float,
    lon_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick grid points within ±lon_tol of target_lon, sorted by latitude."""
    mask = np.abs(lons - target_lon) < lon_tol
    lat_slice = lats[mask]
    field_slice = field[mask]
    order = np.argsort(lat_slice)
    return lat_slice[order], field_slice[order]


def main(outfn: Path) -> None:
    lons, lats, elevation = load_topography(REALCH1_ZARR)

    lon_min, lon_max = float(np.nanmin(lons)), float(np.nanmax(lons))
    lat_min, lat_max = float(np.nanmin(lats)), float(np.nanmax(lats))
    bbox = [lon_min, lon_max, lat_min, lat_max]

    plotter = StatePlotter(lons, lats, outfn.parent)
    fig = plotter.init_geoaxes(
        nrows=2,
        ncols=1,
        projection=_PROJECTIONS["orthographic"],
        bbox=bbox,
        name="ICON-REA-L-CH1 topography",
        size=(8, 9),
    )
    # Replace the default equal-height gridspec so the map gets ~2x the height
    # of the cross-section panel. add_map() hasn't run yet, so this is safe.
    fig.gridspec = fig.fig.add_gridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.18)

    subplot = fig.add_map(row=0, column=0)
    style = ekp.styles.Style(
        colors="terrain",
        levels=ELEVATION_LEVELS,
        extend="max",
        units="m",
    )
    plotter.plot_field(subplot, elevation, style=style)

    # LAM envelope outline, as in the showcase.
    lam_hull = MultiPoint(list(zip(lons.tolist(), lats.tolist()))).convex_hull
    subplot.ax.add_geometries(
        [lam_hull],
        crs=ccrs.PlateCarree(),
        edgecolor="#333333",
        facecolor="none",
        linewidth=0.6,
    )

    # Vertical line marking the cross-section longitude.
    subplot.ax.plot(
        [CROSS_LON, CROSS_LON],
        [lat_min, lat_max],
        color="#C0392B",
        linewidth=1.4,
        linestyle="--",
        transform=ccrs.PlateCarree(),
        label=f"cross-section @ {CROSS_LON}°E",
    )
    subplot.ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    # Cross-section panel on row 1 of the same gridspec — inherits the
    # earthkit-plots schema (Roboto font, light grid, hidden spines).
    lat_cs, elev_cs = extract_cross_section(lons, lats, elevation, CROSS_LON, LON_TOL)
    ax_cs = fig.fig.add_subplot(fig.gridspec[1, 0])
    ax_cs.fill_between(lat_cs, 0, elev_cs, color="#7a7a7a", alpha=0.85)
    ax_cs.plot(lat_cs, elev_cs, color="#333333", linewidth=0.6)
    ax_cs.set_xlim(lat_min, lat_max)
    ax_cs.set_ylim(0, max(4500, float(np.nanmax(elev_cs)) * 1.05))
    ax_cs.set_xlabel("latitude [°N]")
    ax_cs.set_ylabel("elevation [m]")
    ax_cs.set_title(f"Cross-section at {CROSS_LON}°E")

    fig.title("ICON-REA-L-CH1 topography")
    fig.save(outfn, bbox_inches="tight", dpi=200)
    print(f"saved: {outfn}")


if __name__ == "__main__":
    main(Path("figures/realch1_topography_cross_section.png"))
