"""Paper plot: topography (z) of ICON-REA-L-CH1 with a fixed-longitude cross-section.

Loads the constant `z` field (geopotential at surface) from the realch1 anemoi
zarr, converts it to elevation [m], and renders:

  - Top: map of the elevation over the full realch1 domain, with a vertical
    line marking the cross-section longitude.
  - Bottom: elevation profile along that longitude vs latitude.

Uses the same earthkit-plots schema (Roboto font, soft grid, no axis spines)
as the showcase workflow. The map is drawn in the COSMO-CH rotated pole CRS
because that is the data's native grid — the domain envelope is then a
true rectangle, so the figure has no white slack around the data.
"""

from pathlib import Path

import cartopy.crs as ccrs
import earthkit.plots as ekp  # noqa: F401 — import applies default schema (fonts, rcParams)
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm
from matplotlib.tri import Triangulation
from shapely.geometry import MultiPoint

# COSMO-CH rotated pole — realch1 lives on this grid.
ROTATED_POLE = ccrs.RotatedPole(pole_longitude=-170.0, pole_latitude=43.0)

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
    lat_min, lat_max = float(np.nanmin(lats)), float(np.nanmax(lats))

    # Transform to the data's native rotated pole — in this CRS the realch1
    # domain is a perfect rectangle.
    xyz = ROTATED_POLE.transform_points(ccrs.PlateCarree(), lons, lats)
    x_rot, y_rot = xyz[:, 0], xyz[:, 1]
    valid = np.isfinite(x_rot) & np.isfinite(y_rot) & np.isfinite(elevation)
    triang = Triangulation(x_rot[valid], y_rot[valid])

    # earthkit-plots Figure: 2 rows, map on top, profile on bottom.
    fig = ekp.Figure(
        rows=2,
        columns=1,
        size=(8, 9),
        crs=ROTATED_POLE,
        height_ratios=[2.4, 1.0],
        hspace=0.18,
    )
    subplot = fig.add_map(row=0, column=0)
    ax = subplot.ax  # GeoAxes in the rotated pole CRS

    cmap = plt.get_cmap("terrain", len(ELEVATION_LEVELS)).copy()
    norm = BoundaryNorm(ELEVATION_LEVELS, cmap.N, extend="max")
    cf = ax.tricontourf(
        triang,
        elevation[valid],
        levels=ELEVATION_LEVELS,
        cmap=cmap,
        norm=norm,
        extend="max",
        transform=ROTATED_POLE,
    )

    # Tight extent in rotated coordinates — no projection slack at the corners.
    ax.set_extent(
        [x_rot[valid].min(), x_rot[valid].max(),
         y_rot[valid].min(), y_rot[valid].max()],
        crs=ROTATED_POLE,
    )

    # Showcase-style background layers, but skip gridlines (handled separately).
    subplot.land()
    subplot.coastlines()
    subplot.borders()

    # Lat/lon labels without drawing the grid lines themselves. Constrain
    # labels to the left/bottom edges so the diagonal rotated-pole meridians
    # don't drop labels in the middle of the map.
    gl = ax.gridlines(
        draw_labels=["left", "bottom"],
        linewidth=0,
        alpha=0,
        color="none",
        x_inline=False,
        y_inline=False,
    )

    # LAM envelope outline in lat/lon.
    lam_hull = MultiPoint(list(zip(lons.tolist(), lats.tolist()))).convex_hull
    ax.add_geometries(
        [lam_hull],
        crs=ccrs.PlateCarree(),
        edgecolor="#333333",
        facecolor="none",
        linewidth=0.6,
    )

    # Vertical line marking the cross-section longitude.
    ax.plot(
        [CROSS_LON, CROSS_LON],
        [lat_min, lat_max],
        color="#C0392B",
        linewidth=1.4,
        linestyle="--",
        transform=ccrs.PlateCarree(),
        label=f"cross-section @ {CROSS_LON}°E",
    )
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.set_title("ICON-REA-L-CH1 topography")

    cbar = fig.fig.colorbar(
        cf, ax=ax, orientation="horizontal", shrink=0.7, pad=0.08, extend="max"
    )
    cbar.set_label("elevation [m]")

    # Cross-section panel on row 1: fill drawn above the schema y-grid (zorder).
    lat_cs, elev_cs = extract_cross_section(lons, lats, elevation, CROSS_LON, LON_TOL)
    ax_cs = fig.fig.add_subplot(fig.gridspec[1, 0])
    ax_cs.fill_between(lat_cs, 0, elev_cs, color="#7a7a7a", alpha=0.95, zorder=3)
    ax_cs.plot(lat_cs, elev_cs, color="#333333", linewidth=0.6, zorder=4)
    ax_cs.set_xlim(lat_min, lat_max)
    ax_cs.set_ylim(0, max(4500, float(np.nanmax(elev_cs)) * 1.05))
    ax_cs.set_xlabel("latitude [°N]")
    ax_cs.set_ylabel("elevation [m]")
    ax_cs.set_title(f"Cross-section at {CROSS_LON}°E")

    outfn.parent.mkdir(parents=True, exist_ok=True)
    fig.save(outfn, bbox_inches="tight", dpi=200)
    print(f"saved: {outfn}")


if __name__ == "__main__":
    main(Path("figures/realch1_topography_cross_section.png"))
