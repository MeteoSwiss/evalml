"""Paper plot: 10 m maximum wind gust (VMAX_10M) from a realv2 inference output.

Loads the `VMAX_10M` field from a realv2 NetCDF file written by anemoi-inference
and renders it as a map over the realv2 / COSMO-CH domain.

realv2 lives on the COSMO-CH rotated pole grid (the same grid as realch1, see
`paper_plots.py`), so the map is drawn in that rotated pole CRS and the data
envelope is a true rectangle with no projection slack. The field is rendered
with a *continuous* colormap via gouraud-shaded `tripcolor`: the grid points are
projected into the rotated-pole CRS once and triangulated there, so matplotlib
draws directly (no per-call cartopy reprojection) and the result is both smooth
and fast. Importing `earthkit.plots` applies the project's paper styling
(Roboto font, soft grid, no axis spines) through matplotlib rcParams.
"""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import earthkit.plots  # noqa: F401  applies the paper styling via matplotlib rcParams
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import Normalize
from matplotlib.tri import Triangulation

# COSMO-CH rotated pole — realv2 (like realch1) lives on this grid.
ROTATED_POLE = ccrs.RotatedPole(pole_longitude=-170.0, pole_latitude=43.0)

REALV2_NC = Path(
    "/users/rradev/evalml/output/data/runs/forecaster-dadd-20dc/"
    "f9d8/202401010000/realv2.nc"
)
VARIABLE = "VMAX_10M"  # 10 m maximum wind gust [m/s]
TIME_INDEX = -1  # which forecast step to plot (-1 = last)
GUST_MAX = 36.0  # top of the continuous colour scale [m/s]
CMAP = "viridis"


def load_model_field(
    nc_path: Path, variable: str, time_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.datetime64]:
    """Return (lon, lat, field, valid_time) for one forecast step of `variable`."""
    ds = xr.open_dataset(nc_path)
    da = ds[variable].isel(time=time_index)
    lons = ds["longitude"].values
    lats = ds["latitude"].values
    lons = np.where(lons > 180, lons - 360, lons)
    return lons, lats, da.values, da["time"].values


def projected_triangulation(
    lons: np.ndarray, lats: np.ndarray
) -> tuple[Triangulation, np.ndarray]:
    """Triangulate the grid in ROTATED_POLE coordinates.

    Returns the triangulation and the boolean mask of points kept (those with a
    finite projected position). Plotting in the axes' own CRS means `tripcolor`
    never has to reproject, which is what makes a continuous render of ~1.1M
    points fast.
    """
    xyz = ROTATED_POLE.transform_points(ccrs.PlateCarree(), lons, lats)
    x, y = xyz[:, 0], xyz[:, 1]
    finite = np.isfinite(x) & np.isfinite(y)
    return Triangulation(x[finite], y[finite]), finite


def draw_field(ax, tri: Triangulation, field: np.ndarray, *, norm, cmap=CMAP):
    """Continuous (gouraud) tripcolor of `field` plus coastlines/borders.

    `field` must already be masked to the triangulation's points.
    """
    mappable = ax.tripcolor(
        tri, field, shading="gouraud", cmap=cmap, norm=norm, rasterized=True
    )
    ax.coastlines(resolution="50m", linewidth=0.4, color="0.25")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="0.4")
    ax.set_extent(
        [tri.x.min(), tri.x.max(), tri.y.min(), tri.y.max()], crs=ROTATED_POLE
    )
    return mappable


def add_gridlines(ax, *, left: bool = True, bottom: bool = True):
    """Light lon/lat graticule, labelled only on the requested edges."""
    edges = [e for e, on in (("left", left), ("bottom", bottom)) if on]
    ax.gridlines(
        draw_labels=edges or False,
        linewidth=0.4,
        color="0.8",
        alpha=0.7,
        x_inline=False,
        y_inline=False,
    )


def format_time(valid_time: np.datetime64) -> str:
    return str(valid_time)[:16].replace("T", " ")


def main(outfn: Path) -> None:
    outfn.parent.mkdir(parents=True, exist_ok=True)
    lons, lats, field, valid_time = load_model_field(REALV2_NC, VARIABLE, TIME_INDEX)
    tri, finite = projected_triangulation(lons, lats)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ROTATED_POLE)
    mappable = draw_field(ax, tri, field[finite], norm=Normalize(0, GUST_MAX))
    add_gridlines(ax)

    cbar = fig.colorbar(mappable, ax=ax, shrink=0.85, pad=0.02, extend="max")
    cbar.set_label("10 m max wind gust [m s$^{-1}$]")
    ax.set_title(f"ICON-REA-L-CH1 10 m max wind gust — {format_time(valid_time)}")

    fig.savefig(outfn, bbox_inches="tight", dpi=200)
    print(f"saved: {outfn}")


if __name__ == "__main__":
    main(Path("figures/realv2_vmax10m.png"))
