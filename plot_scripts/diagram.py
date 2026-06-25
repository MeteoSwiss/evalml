"""Reproduce the globe from ``paper/diagram/diagram.svg``.

An orthographic globe of 2 m temperature from the global O48 ERA5 analysis for
2021-01-01 00:00 UTC, drawn as a filled contour on the reduced-Gaussian grid with
coastlines and a bold outline of the Alpine LAM domain. Colours come from the
earthkit-plots Spectral_r temperature style.

The exact parameters were reverse-engineered from the original SVG:
  - orientation (14W / 40N) fitted by maximising coastline overlap;
  - 41 contour levels -> 42 Spectral_r bands (matching the 42 fill colours);
  - the date located by scanning every timestep of the O48 archives.
"""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import earthkit.plots as ekp
import matplotlib
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

# cmocean's "tempo" is not a matplotlib built-in, so register it by name once
# (earthkit resolves the colour scheme from the matplotlib registry).
if "tempo" not in matplotlib.colormaps:
    matplotlib.colormaps.register(cmocean.cm.tempo, name="tempo")
from matplotlib.tri import Triangulation

# --- data sources ----------------------------------------------------------
# Hourly lead times need the 1-hourly archive, which only covers 2020, so the
# base ("current") timestep lives in 2020 rather than the original's 2021 date.
O48_ZARR = Path(
    "/scratch/mch/rradev/datasets/aifs-ea-an-oper-0001-mars-o48-2020-2020-1h-v1.zarr"
)
# Only the coordinates of the LAM dataset are read, to draw the domain outline;
# its field is never plotted.
REALCH1_ZARR = Path(
    "/store_new/mch/msopr/ml/datasets/"
    "mch-realch1-fdb-1km-2005-2025-1h-pl13-ifsnames-v1.0.zarr"
)
VARIABLE = "2t"  # 2 m temperature
BASE_TIME = np.datetime64("2020-01-01T00:00")  # the "current" timestep (t+0)
LEAD_HOURS = [0, 6, 7, 8, 9, 10, 11, 12]  # frames to render, valid = base + lead
# The composite inset is purely illustrative, so it just uses 1 Jan of the next
# year (which the realch1 archive covers) regardless of the globe's date.
INSET_TIME = np.datetime64("2021-01-01T00:00")

# --- projection / style ----------------------------------------------------
ORTHO = ccrs.Orthographic(central_longitude=-14.0, central_latitude=40.0)
ROTATED_POLE = ccrs.RotatedPole(pole_longitude=-170.0, pole_latitude=43.0)
PLATE = ccrs.PlateCarree()
NORD_BLUE = "#5e81ac"  # the domain-box colour from the original
# 41 contour levels (K) -> 42 colour bands with extend="both".
GLOBAL_LEVELS = list(range(235, 316, 2))

# Two style variants, each rendered to its own folder. The tempo palette is dark
# so the box/inset borders switch to a light Nord tone; Spectral_r keeps the
# original Nord blue.
STYLES = {
    "spectral_r": {"cmap": "Spectral_r", "border": NORD_BLUE},
    "tempo": {"cmap": "tempo", "border": "#E5E9F0"},
}

# Each figure is placed at ~80 pt on the page; render it at print resolution for
# that size (so sharpness, not file size, is set by pixels-across-at-80-pt).
DISPLAY_PT = 80.0
PRINT_DPI = 300.0


def _dpi_for(fig_width_in: float) -> float:
    """Save dpi so a `fig_width_in`-wide figure is PRINT_DPI at DISPLAY_PT."""
    return PRINT_DPI * (DISPLAY_PT / 72) / fig_width_in

# Line weights measured from the original SVG, where the globe disk has radius
# 180 pt. They are scaled to our disk size so the lines keep the same thickness
# relative to the globe.
FIGSIZE_IN = 7.0
_ORIG_DISK_R = 180.0
_LW_SCALE = (FIGSIZE_IN / 2 * 72) / _ORIG_DISK_R  # our disk radius (pt) / theirs
BOX_LW = 6.5 * _LW_SCALE       # domain-box stroke (5 in the original, thickened)
COAST_LW = 1.1 * _LW_SCALE     # coastline stroke (1.1 in the original)

# --- with_inset composite layout (measured from with_inset.svg) -------------
# The original figure is 907.2 x 1300.94 pt; we reuse those exact dimensions so
# the SVG's point line-weights and axes positions reproduce 1:1.
INSET_W_PT, INSET_H_PT = 907.2, 1300.939733
INSET_FIGSIZE = (INSET_W_PT / 72, INSET_H_PT / 72)


def _frac(x: float, y: float) -> tuple[float, float]:
    """SVG point (y-down) -> matplotlib figure fraction (y-up)."""
    return (x / INSET_W_PT, 1.0 - y / INSET_H_PT)


# axes rectangles [left, bottom, width, height], from the SVG patch bboxes.
# Globe disk: centre (453.6, 183.6), radius 180 -> a 360 pt square at the top.
GLOBE_AXES_POS = (273.6 / INSET_W_PT, 1 - 363.6 / INSET_H_PT,
                  360.0 / INSET_W_PT, 360.0 / INSET_H_PT)
# Inset axes spans almost the full width across the lower ~47% of the canvas.
INSET_AXES_POS = (3.6 / INSET_W_PT, 1 - 1297.339733 / INSET_H_PT,
                  900.0 / INSET_W_PT, (1297.339733 - 687.6) / INSET_H_PT)
# Zoom-indicator anchors: the globe box's two bottom corners join the inset's
# two top corners; the inset border is the rectangle of the inset axes.
BOX_BL, BOX_BR = _frac(487.398507, 174.457914), _frac(521.56779, 164.626386)
INSET_TL, INSET_TR = _frac(3.6, 687.6), _frac(903.6, 687.6)
INSET_BL, INSET_BR = _frac(3.6, 1297.339733), _frac(903.6, 1297.339733)
# Line-weights straight from the SVG (the figure is its native size, so points
# map directly): 7 pt blue, 1.1 pt coastline.
INSET_BLUE_LW = 7.0
INSET_COAST_LW = 1.1


def load_field(
    zarr_root: Path, variable: str, target_time: np.datetime64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lon[-180,180], lat, field) for ``variable`` at the nearest time."""
    ds = xr.open_zarr(zarr_root, consolidated=False)
    var_idx = list(ds.attrs["variables"]).index(variable)
    # anemoi stores timestamps in `dates`; the `time` coordinate is just an index.
    dates = ds["dates"].values.astype("datetime64[m]")
    time_idx = int(np.abs(dates - target_time).argmin())
    field = ds["data"][time_idx, var_idx, 0, :].values
    lon = ds["longitudes"].values
    lat = ds["latitudes"].values
    lon = np.where(lon > 180, lon - 360, lon)
    return lon, lat, field


def domain_outline(zarr_root: Path, n: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Densified lon/lat outline of the LAM domain (a rotated-pole rectangle).

    Reads only the grid coordinates of the LAM dataset (never its field). The
    grid is a rectangle in the COSMO-CH rotated-pole CRS, so its true outline is
    curved when drawn on a globe.
    """
    ds = xr.open_zarr(zarr_root, consolidated=False)
    lon = ds["longitudes"].values
    lat = ds["latitudes"].values
    lon = np.where(lon > 180, lon - 360, lon)
    xyz = ROTATED_POLE.transform_points(PLATE, lon, lat)
    rlon, rlat = xyz[:, 0], xyz[:, 1]
    rlon_min, rlon_max = rlon.min(), rlon.max()
    rlat_min, rlat_max = rlat.min(), rlat.max()

    edge_lon = np.concatenate([
        np.linspace(rlon_min, rlon_max, n),
        np.full(n, rlon_max),
        np.linspace(rlon_max, rlon_min, n),
        np.full(n, rlon_min),
    ])
    edge_lat = np.concatenate([
        np.full(n, rlat_min),
        np.linspace(rlat_min, rlat_max, n),
        np.full(n, rlat_max),
        np.linspace(rlat_max, rlat_min, n),
    ])
    back = PLATE.transform_points(ROTATED_POLE, edge_lon, edge_lat)
    return back[:, 0], back[:, 1]


def plot_globe(
    ax: plt.Axes, lon, lat, field, box_lon, box_lat, cmap: str, border_color: str,
    box_lw: float = BOX_LW, coast_lw: float = COAST_LW,
) -> None:
    """Orthographic globe of the global field, with the LAM domain outlined."""
    ax.set_global()
    # Project to the orthographic CRS and drop points on the far side of the
    # globe (they become non-finite) before triangulating, so no triangles wrap
    # across the limb.
    xyz = ORTHO.transform_points(PLATE, lon, lat)
    x, y = xyz[:, 0], xyz[:, 1]
    visible = np.isfinite(x) & np.isfinite(y) & np.isfinite(field)
    tri = Triangulation(x[visible], y[visible])

    style = ekp.styles.Style(
        colors=cmap, levels=GLOBAL_LEVELS, extend="both", units="K"
    )
    kw = style.to_contourf_kwargs(field[visible])
    cs = ax.tricontourf(tri, field[visible], **kw)
    # Fill the antialiasing seams between filled bands (otherwise thin white
    # lines appear between contour levels).
    cs.set_edgecolor("face")

    ax.add_feature(cfeature.COASTLINE, linewidth=coast_lw, edgecolor="#333333", alpha=0.9)
    # The original draws the box as a round-jointed line, so its corners are
    # rounded rather than sharp mitre points.
    box = Polygon(
        np.column_stack([box_lon, box_lat]), closed=True, fill=False,
        edgecolor=border_color, linewidth=box_lw, joinstyle="round",
        capstyle="round", alpha=1.0, zorder=5, transform=PLATE,
    )
    ax.add_patch(box)
    ax.spines["geo"].set_visible(False)


def render_frame(
    outfn: Path, valid_time: np.datetime64, box_lon, box_lat,
    cmap: str, border_color: str,
) -> None:
    """Draw and save a single globe frame valid at ``valid_time``."""
    lon_g, lat_g, field_g = load_field(O48_ZARR, VARIABLE, valid_time)

    fig = plt.figure(figsize=(FIGSIZE_IN, FIGSIZE_IN))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection=ORTHO)
    plot_globe(ax, lon_g, lat_g, field_g, box_lon, box_lat, cmap, border_color)

    outfn.parent.mkdir(parents=True, exist_ok=True)
    # Rendered for an 80 pt on-page size at print resolution (~333 px square),
    # as a PNG to avoid the multi-MB vector file the tricontourf would produce.
    fig.savefig(outfn, bbox_inches="tight", dpi=_dpi_for(FIGSIZE_IN), transparent=True)
    plt.close(fig)
    print(f"saved: {outfn}")


def plot_inset(ax: plt.Axes, lon, lat, field, coast_lw: float, cmap: str) -> None:
    """Zoomed LAM field, drawn in the rotated-pole CRS as a tight rectangle."""
    # In the COSMO-CH rotated pole the LAM grid is a filled rectangle, so the
    # axes fills exactly with no projection slack (matching the SVG inset).
    xyz = ROTATED_POLE.transform_points(PLATE, lon, lat)
    rx, ry = xyz[:, 0], xyz[:, 1]
    ok = np.isfinite(rx) & np.isfinite(ry) & np.isfinite(field)
    tri = Triangulation(rx[ok], ry[ok])

    # Locally normalise the colour scale to the inset's own range (1-99th
    # percentile, so a few outliers don't wash it out) for vivid local contrast,
    # rather than reusing the globe's global 235-315 K levels.
    lo, hi = np.percentile(field[ok], [1, 99])
    local_levels = list(np.linspace(lo, hi, len(GLOBAL_LEVELS)))
    style = ekp.styles.Style(
        colors=cmap, levels=local_levels, extend="both", units="K"
    )
    kw = style.to_contourf_kwargs(field[ok])
    cs = ax.tricontourf(tri, field[ok], **kw)
    cs.set_edgecolor("face")

    ax.add_feature(cfeature.COASTLINE, linewidth=coast_lw, edgecolor="#333333", alpha=0.9)
    # The inset also carries country boundaries (as in the original).
    ax.add_feature(cfeature.BORDERS, linewidth=coast_lw, edgecolor="#333333", alpha=0.9)
    ax.set_extent([rx.min(), rx.max(), ry.min(), ry.max()], crs=ROTATED_POLE)
    ax.spines["geo"].set_visible(False)  # the blue border is drawn separately


def build_with_inset(
    outfn: Path, valid_time: np.datetime64, box_lon, box_lat,
    cmap: str, border_color: str,
) -> None:
    """Composite: globe with the LAM box, a zoomed inset, and the zoom lines."""
    lon_g, lat_g, field_g = load_field(O48_ZARR, VARIABLE, valid_time)
    lon_l, lat_l, field_l = load_field(REALCH1_ZARR, VARIABLE, INSET_TIME)

    fig = plt.figure(figsize=INSET_FIGSIZE)
    ax_globe = fig.add_axes(GLOBE_AXES_POS, projection=ORTHO)
    plot_globe(ax_globe, lon_g, lat_g, field_g, box_lon, box_lat, cmap, border_color,
               box_lw=INSET_BLUE_LW, coast_lw=INSET_COAST_LW)
    ax_inset = fig.add_axes(INSET_AXES_POS, projection=ROTATED_POLE)
    plot_inset(ax_inset, lon_l, lat_l, field_l, INSET_COAST_LW, cmap)

    # Inset border + zoom-indicator connectors, drawn in figure coordinates so
    # the lines can span from the globe box down to the inset corners.
    border = Polygon(
        [INSET_TL, INSET_TR, INSET_BR, INSET_BL], closed=True, fill=False,
        edgecolor=border_color, linewidth=INSET_BLUE_LW, joinstyle="miter",
        transform=fig.transFigure, figure=fig, zorder=6,
    )
    fig.add_artist(border)
    for start, end in [(BOX_BL, INSET_TL), (BOX_BR, INSET_TR)]:
        line = Line2D(
            [start[0], end[0]], [start[1], end[1]], color=border_color,
            linewidth=INSET_BLUE_LW, solid_capstyle="round",
            transform=fig.transFigure, figure=fig, zorder=6,
        )
        fig.add_artist(line)

    outfn.parent.mkdir(parents=True, exist_ok=True)
    # No bbox_inches="tight": the layout is measured against the full canvas, so
    # cropping margins would shift the relative positions. The composite is shown
    # larger than the small globe frames, so it keeps a higher fixed resolution.
    fig.savefig(outfn, dpi=120, transparent=True)
    plt.close(fig)
    print(f"saved: {outfn}")


def main(out_dir: Path) -> None:
    box_lon, box_lat = domain_outline(REALCH1_ZARR)
    for name, style in STYLES.items():
        style_dir = out_dir / name
        cmap, border = style["cmap"], style["border"]
        for lead in LEAD_HOURS:
            valid_time = BASE_TIME + np.timedelta64(lead, "h")
            outfn = style_dir / f"diagram_reproduced_t+{lead:02d}.png"
            render_frame(outfn, valid_time, box_lon, box_lat, cmap, border)

        # The composite is only needed for the first timestep.
        build_with_inset(
            style_dir / "with_inset_reproduced_t+00.png",
            BASE_TIME + np.timedelta64(LEAD_HOURS[0], "h"),
            box_lon, box_lat, cmap, border,
        )


if __name__ == "__main__":
    main(Path("paper/diagram"))
