"""Paper plot: topography (z) of ICON-REA-L-CH1 with a fixed-longitude cross-section.

Loads the constant `z` field (geopotential at surface) from the realch1 anemoi
zarr, converts it to elevation [m], and renders side-by-side:

  - Left: map of the elevation over the full realch1 domain.
  - Right: elevation profile along a fixed longitude, with latitude on the
    y-axis so it lines up with the map.

Uses the same earthkit-plots schema (Roboto font, soft grid, no axis spines)
and the project's `StatePlotter` as the showcase workflow. The map is drawn
in the COSMO-CH rotated pole CRS — realch1's native grid — so the data
domain is a true rectangle with no projection slack around it.
"""

from pathlib import Path

import cartopy.crs as ccrs
import earthkit.plots as ekp
import numpy as np
import xarray as xr

from plotting import StatePlotter

# COSMO-CH rotated pole — realch1 lives on this grid.
ROTATED_POLE = ccrs.RotatedPole(pole_longitude=-170.0, pole_latitude=43.0)

REALCH1_ZARR = Path(
    "/store_new/mch/msopr/ml/datasets/"
    "mch-realch1-fdb-1km-2005-2025-1h-pl13-ifsnames-v1.0.zarr"
)
G = 9.80665  # standard gravity, m/s^2
CROSS_LON = 9.0  # fixed longitude (°E) for the cross-section through the Alps
LON_TOL = 0.005  # half-width [°] of the longitude band used for the profile
ELEVATION_LEVELS = list(range(0, 3500, 250))


def load_topography(zarr_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lon, lat, elevation [m]) for realch1 from the constant `z` field."""
    ds = xr.open_zarr(zarr_root, consolidated=False)
    variables = list(ds.attrs["variables"])
    z_idx = variables.index("z")
    geopotential = ds["data"][0, z_idx, 0, :].values 
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

    plotter = StatePlotter(lons, lats, outfn.parent)
    fig = plotter.init_geoaxes(
        nrows=1,
        ncols=2,
        projection=ROTATED_POLE,
        bbox=[lon_min, lon_max, lat_min, lat_max],
        name="ICON-REA-L-CH1 topography",
        size=(13, 6),
    )
    # Override the equal-width gridspec so the map gets ~2.4x the width of the
    # cross-section panel. add_map() hasn't run yet, so this is safe.
    fig.gridspec = fig.fig.add_gridspec(1, 2, width_ratios=[2.4, 1.0], wspace=0.12)

    subplot = fig.add_map(row=0, column=0)
    style = ekp.styles.Style(
        colors="terrain",
        levels=ELEVATION_LEVELS,
        extend="max",
        units="m",
    )
    plotter.plot_field(subplot, elevation, style=style)

    # In rotated-pole coordinates the data envelope is a perfect rectangle.
    # Set the extent to the projected min/max so the map is a tight fit.
    xyz = ROTATED_POLE.transform_points(ccrs.PlateCarree(), lons, lats)
    x_rot, y_rot = xyz[:, 0], xyz[:, 1]
    finite = np.isfinite(x_rot) & np.isfinite(y_rot)
    subplot.ax.set_extent(
        [x_rot[finite].min(), x_rot[finite].max(),
         y_rot[finite].min(), y_rot[finite].max()],
        crs=ROTATED_POLE,
    )

    # standard_layers() (called inside plot_field) draws a cartopy graticule.
    # Replace it with a label-only one constrained to the left/bottom edges so
    # the diagonal rotated-pole meridians do not drop labels in the map body.
    from cartopy.mpl.gridliner import Gridliner

    for artist in list(subplot.ax.artists):
        if isinstance(artist, Gridliner):
            artist.remove()
    subplot.ax._gridliners = []
    subplot.ax.gridlines(
        draw_labels=["left", "bottom"],
        linewidth=0,
        alpha=0,
        color="none",
        x_inline=False,
        y_inline=False,
    )

    # Cross-section panel on column 1: latitude on y-axis to align with the map.
    lat_cs, elev_cs = extract_cross_section(lons, lats, elevation, CROSS_LON, LON_TOL)
    ax_cs = fig.fig.add_subplot(fig.gridspec[0, 1])
    ax_cs.fill_betweenx(lat_cs, 0, elev_cs, color="#7a7a7a", alpha=0.95, zorder=3)
    ax_cs.plot(elev_cs, lat_cs, color="#333333", linewidth=0.6, zorder=4)
    ax_cs.set_ylim(lat_min, lat_max)
    ax_cs.set_xlim(0, max(3500, float(np.nanmax(elev_cs)) * 1.05))
    ax_cs.set_ylabel("latitude [°N]")
    ax_cs.set_xlabel("[m]")
    ax_cs.set_title(f"Model elevation at {CROSS_LON}°E")

    fig.title("ICON-REA-L-CH1 topography")
    fig.save(outfn, bbox_inches="tight", dpi=200)
    print(f"saved: {outfn}")


if __name__ == "__main__":
    main(Path("figures/realch1_topography_cross_section.png"))
