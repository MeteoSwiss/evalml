"""Publication teaser ("hero") figure for the Varda-Single paper.

A single 2x2 snapshot of one forecast that tells the model's core story at a
glance: a **global** ML forecast with **regional high-resolution refinement**.

                     GLOBAL                    REGIONAL (Switzerland)
    T_2M      [full-globe orthographic]   [Alpine zoom, high-res + LAM box]
    SP_10M    [full-globe orthographic]   [Alpine zoom, high-res + LAM box]

Rows are the two variables (2 m temperature, 10 m wind speed); columns are two
views of the *same* forecast state. The global column is rendered from the
global model output and the regional column from the high-resolution LAM output
(each on its own mesh, so the coarse global grid never contaminates the regional
triangulation). The LAM envelope box is drawn on the global panels to connect
the global field to the refined region.

Styling is poster-oriented: white background, no gridlines/frames, continent
outlines only on the globe, zoom-callout lines from the globe to each regional
panel, and small captions. Temperature uses a diverging RdYlBu (wide range on
the globe, tighter in the zoom); wind uses cubehelix with thinned flow arrows.

Runs standalone (no Snakemake), reading the inference-sandbox GRIB layout::

    python workflow/scripts/publication_teaser.py \
        --input   <grib_dir> \
        --date    202505111200 \
        --leadtime 12 \
        --output  figures/teaser
"""

import logging
import sys
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import earthkit.meteo.wind as ekm_wind
import earthkit.plots as ekp
import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.tri as mtri
import numpy as np
import xarray as xr
from cartopy.mpl.gridliner import Gridliner
from earthkit.meteo.utils.convert import kelvin_to_celsius
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import ConnectionPatch
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint

_script_dir = Path(__file__).resolve().parent
sys.path.append(str(_script_dir))
sys.path.append(str(_script_dir.parent.parent / "src"))

plt.style.use(_script_dir / "publication.mplstyle")

from plotting import DOMAINS, StatePlotter  # noqa: E402
from plotting.compat import load_from_grib_file  # noqa: E402

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Map furniture. Globe panels get *continent outlines only* (coastlines), so the
# country-border layer is disabled globally here and re-added by hand on the
# regional panels. Light neutral lines read over both the dark wind field and
# the light temperature field.
OUTLINE_COLOR = "#e8e8e8"
CALLOUT_COLOR = "#4d4d4d"
CAPTION_COLOR = "#333333"
BG_COLOR = "white"  # figure/axes background (overridden by --dark)
DARK_BG = "black"  # background for the dark variant
GLOW = False  # soft halo around each panel (enabled by --dark)
ekp.schema.borders["edgecolor"] = "none"  # off everywhere; added manually on regional
ekp.schema.coastlines["resolution"] = "high"
ekp.schema.coastlines["edgecolor"] = OUTLINE_COLOR
ekp.schema.coastlines["linewidth"] = 0.8

GLOBAL_DOMAIN = "globe"
REGIONAL_DOMAIN = "switzerland"
# Deep nested zoom over Val Calanca (Graubünden, southern Switzerland); approx bbox.
VAL_CALANCA_EXTENT = [8.85, 9.42, 46.1, 46.55]
# Colorbar unit labels. Wind uses mathtext for the exponent so it doesn't depend
# on a superscript-minus glyph being present in the figure font.
UNIT_LABEL = {"°C": "(°C)", "m/s": "(m s$^{-1}$)"}

# GRIB variable rename maps: {grib_short_name: internal_key}. The regional LAM
# file uses COSMO names; the global IFS sibling uses ECMWF short names.
LAM_VARS = {"T_2M": "T_2M", "U_10M": "U_10M", "V_10M": "V_10M"}
GLOBAL_VARS = {"2t": "T_2M", "10u": "U_10M", "10v": "V_10M"}

# The two variables shown, top row to bottom row.
FIELDS = [
    {"key": "T_2M", "label": "2 m Temperature"},
    {"key": "SP_10M", "label": "10 m Wind Speed"},
]

# Field styles. Temperature uses a wide range on the global panel and a tighter
# ``regional_levels`` range so the Alpine detail gets full contrast in the zoom.
FIELD_STYLES = {
    "T_2M": dict(
        cmap="Spectral_r",
        # Global: Arctic cold to warm subtropics. Regional: fitted to the case
        # (a CH spring midday: sub-zero Alpine peaks to mild/warm lowlands).
        levels=list(range(-40, 31, 2)),
        regional_levels=list(range(-9, 25, 1)),
        units="°C",
        extend="both",
    ),
    # cubehelix: monotonic-luminance, calm dark -> strong light. Many fine levels
    # so tricontourf reads as a smooth gradient rather than blocky bands;
    # cmap_range trims the near-white tail. The globe uses a wider range (0-25)
    # to resolve ocean storms; the CH zoom uses 0-16.
    "SP_10M": dict(
        cmap="cubehelix",
        cmap_range=(0.0, 0.85),
        levels=list(np.round(np.arange(0.0, 25.0 + 1e-9, 0.5), 2)),
        regional_levels=list(np.round(np.arange(0.0, 16.0 + 1e-9, 0.25), 2)),
        units="m/s",
        extend="max",
    ),
}


def _find_grib(grib_dir: Path, lead_time: int) -> Path:
    """Locate the GRIB file for a lead time (mirrors plot_forecast_frame.py)."""
    matches = list(grib_dir.glob(f"2*_{lead_time}.grib"))
    if not matches:
        matches = list(grib_dir.glob(f"2*_{lead_time:03}.grib"))
    if not matches:
        raise FileNotFoundError(
            f"No GRIB file for lead_time={lead_time} in {grib_dir} "
            f"(patterns '2*_{lead_time}.grib' / '2*_{lead_time:03}.grib')."
        )
    return Path(matches[0])


def _load_state(grib_file: Path, rename: dict, *, wrap_lon: bool) -> dict:
    """Load one GRIB file into a {longitudes, latitudes, valid_time, fields} state."""
    ds = load_from_grib_file(grib_file, {"parameter.variable": list(rename)})
    lon = ds["longitude"].values.flatten()
    lat = ds["latitude"].values.flatten()
    if wrap_lon and lon.max() > 180:
        lon = ((lon + 180) % 360) - 180
    valid_time = datetime.fromtimestamp(
        ds["valid_time"].values.item() / 1e9, tz=timezone.utc
    )
    fields = {key: ds[var].values.flatten() for var, key in rename.items()}
    return dict(longitudes=lon, latitudes=lat, valid_time=valid_time, fields=fields)


def load_states(grib_dir: Path, lead_time: int) -> tuple[dict, dict]:
    """Return (global_state, lam_state) for the teaser.

    Like the evalml showcases, the global panel gets the high-resolution LAM data
    concatenated onto the coarse global background, so the refined region shows
    its detail on the globe. The regional (zoom) panel uses the LAM-only mesh, so
    the coarse global grid never contaminates the high-resolution triangulation.
    """
    grib_file = _find_grib(grib_dir, lead_time)
    LOG.info("Loading regional LAM GRIB %s", grib_file)
    lam = _load_state(grib_file, LAM_VARS, wrap_lon=False)

    lam_hull = MultiPoint(
        list(zip(lam["longitudes"].tolist(), lam["latitudes"].tolist()))
    ).convex_hull
    lam_envelope = gpd.GeoSeries([lam_hull], crs="EPSG:4326")
    lam["lam_envelope"] = lam_envelope

    global_file = grib_file.parent / f"ifs-{grib_file.stem}.grib"
    if global_file.exists():
        LOG.info("Loading global background %s", global_file)
        coarse = _load_state(global_file, GLOBAL_VARS, wrap_lon=True)
        global_state = dict(
            longitudes=np.concatenate([lam["longitudes"], coarse["longitudes"]]),
            latitudes=np.concatenate([lam["latitudes"], coarse["latitudes"]]),
            valid_time=lam["valid_time"],
            fields={
                k: np.concatenate([lam["fields"][k], coarse["fields"][k]])
                for k in LAM_VARS.values()
            },
        )
    else:
        LOG.warning("No global sibling %s; using LAM for global panels.", global_file)
        global_state = {
            k: lam[k] for k in ("longitudes", "latitudes", "valid_time", "fields")
        }
    global_state["lam_envelope"] = lam_envelope
    return global_state, lam


def _preprocess(key: str, fields: dict) -> np.ndarray:
    """Compute the display field. T_2M: K->degC; SP_10M: |(u, v)|."""
    if key == "T_2M":
        return kelvin_to_celsius(fields["T_2M"])
    if key == "SP_10M":
        return ekm_wind.speed(fields["U_10M"], fields["V_10M"])
    raise KeyError(key)


def _field_cmap(key: str):
    """Colormap for a field, with the optional near-white tail trimmed."""
    cfg = FIELD_STYLES[key]
    cmap = plt.get_cmap(cfg["cmap"])
    lo, hi = cfg.get("cmap_range", (0.0, 1.0))
    if (lo, hi) != (0.0, 1.0):
        cmap = LinearSegmentedColormap.from_list(
            f"{cfg['cmap']}_trunc", cmap(np.linspace(lo, hi, 256))
        )
    return cmap


def _field_levels(key: str, *, is_regional: bool) -> list:
    """Discrete levels for a field; zoom panels use the tighter regional range."""
    cfg = FIELD_STYLES[key]
    if is_regional and "regional_levels" in cfg:
        return cfg["regional_levels"]
    return cfg["levels"]


def _field_style(key: str, *, is_regional: bool) -> dict:
    """Return plot_field style kwargs (earthkit Style with discrete levels)."""
    cfg = FIELD_STYLES[key]
    levels = _field_levels(key, is_regional=is_regional)
    style = ekp.styles.Style(
        colors=_field_cmap(key),
        vmin=levels[0],
        vmax=levels[-1],
        extend=cfg["extend"],
        units=cfg["units"],
    )
    style._kwargs["levels"] = list(levels)
    return {"style": style, "norm": None}


def _round_ticks(key: str, lo: float, hi: float) -> list:
    """Round tick values within [lo, hi] for a continuous colorbar."""
    if key == "SP_10M":
        step = 5 if hi > 20 else 2
        start = 0
    else:
        step = 10 if (hi - lo) > 50 else 5
        start = int(np.ceil(lo / step) * step)
    return [t for t in range(start, int(hi) + 1, step) if lo <= t <= hi]


def _add_column_colorbar(mpl_fig, ax, key: str, *, is_regional: bool):
    """Add a continuous horizontal colorbar under one column (sized to its width).

    Continuous to match the smooth map colors; ticks come from ``_round_ticks``
    so colorbars for the same field range are identical.
    """
    cfg = FIELD_STYLES[key]
    levels = _field_levels(key, is_regional=is_regional)
    lo, hi = levels[0], levels[-1]
    sm = ScalarMappable(cmap=_field_cmap(key), norm=Normalize(vmin=lo, vmax=hi))
    sm.set_array([])
    cbar = mpl_fig.colorbar(
        sm,
        ax=ax,
        orientation="horizontal",
        location="bottom",
        fraction=0.05,
        pad=0.03,
        aspect=22,
        extend=cfg["extend"],
    )
    cbar.set_ticks(_round_ticks(key, lo, hi))
    cbar.set_label(
        UNIT_LABEL.get(cfg["units"], f"({cfg['units']})"), color=CAPTION_COLOR
    )
    cbar.ax.tick_params(colors=CAPTION_COLOR)
    cbar.outline.set_edgecolor(CAPTION_COLOR)
    return cbar


def load_icon_triangulation(grid_path: Path) -> mtri.Triangulation:
    """Build the true ICON cell mesh from the grid file.

    Vertices are ``vlon``/``vlat`` (radians); ``vertex_of_cell`` (1-based, shape
    (3, n_cells)) gives the triangle -> vertex connectivity. The resulting
    triangulation has one triangle per ICON cell, in cell order, so cell-centered
    field values map directly to ``facecolors``.
    """
    g = xr.open_dataset(grid_path)
    vlon = np.degrees(g["vlon"].values)
    vlat = np.degrees(g["vlat"].values)
    triangles = g["vertex_of_cell"].values.T - 1  # (n_cells, 3), 0-based
    return mtri.Triangulation(vlon, vlat, triangles)


def _tripcolor_panel(
    subplot, field, *, cmap, levels, icon_tri=None, plotter=None, extent=None
):
    """Flat-shaded tripcolor render exposing the native model mesh.

    Uses a continuous ``Normalize`` over the level range so the cell colors match
    the continuous colorbars and the smooth contourf panels.

    With ``icon_tri`` the true ICON cells are drawn (cell-centered values as
    ``facecolors``); otherwise it falls back to a Delaunay mesh of the points.
    When ``extent`` is given, the ICON mesh is subset to that bbox (+ margin)
    first, so only the visible cells are projected/drawn -- a huge speedup versus
    handing cartopy the whole ~1.1M-cell mesh for a small zoom window.
    """
    f = np.asarray(field)
    f = f[-1] if f.ndim == 2 else f.squeeze()
    norm = Normalize(vmin=levels[0], vmax=levels[-1])
    if icon_tri is not None:
        tri = icon_tri
        if extent is not None:
            lon0, lon1, lat0, lat1 = extent
            dlon, dlat = (lon1 - lon0) * 0.15, (lat1 - lat0) * 0.15
            cx = icon_tri.x[icon_tri.triangles].mean(axis=1)
            cy = icon_tri.y[icon_tri.triangles].mean(axis=1)
            m = (
                (cx >= lon0 - dlon)
                & (cx <= lon1 + dlon)
                & (cy >= lat0 - dlat)
                & (cy <= lat1 + dlat)
            )
            tri = mtri.Triangulation(icon_tri.x, icon_tri.y, icon_tri.triangles[m])
            f = f[m]
        mappable = subplot.ax.tripcolor(
            tri,
            facecolors=f,
            shading="flat",
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
            zorder=1,
            rasterized=True,
        )
    else:
        triang, mask = plotter._orthographic_tri
        fm = f[mask]
        mappable = subplot.ax.tripcolor(
            triang.x,
            triang.y,
            fm,
            triangles=triang.triangles,
            shading="flat",
            cmap=cmap,
            norm=norm,
            transform=subplot._crs,
            zorder=1,
            rasterized=True,
        )
    subplot.standard_layers()
    return mappable


def _strip_chrome(ax) -> None:
    """Remove gridlines, tick labels and the axis frame from a cartopy GeoAxes."""
    for child in getattr(ax, "_children", []) + getattr(ax, "_gridliners", []):
        if not isinstance(child, Gridliner):
            continue
        try:
            child.left_labels = child.right_labels = False
            child.top_labels = child.bottom_labels = False
            child.xlines = child.ylines = False
        except AttributeError:
            pass
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_facecolor(BG_COLOR)
    for spine in ax.spines.values():
        spine.set_visible(False)
    outline = getattr(ax, "outline_patch", None)
    if outline is not None:
        outline.set_visible(False)
    if GLOW:
        # Re-show the map boundary with a layered white stroke -> soft halo so the
        # panels glow against the black background.
        geo = ax.spines.get("geo")
        if geo is not None:
            geo.set_visible(True)
            geo.set_edgecolor("white")
            geo.set_linewidth(0.8)
            geo.set_path_effects(
                [
                    pe.Stroke(linewidth=9, foreground="white", alpha=0.06),
                    pe.Stroke(linewidth=6, foreground="white", alpha=0.10),
                    pe.Stroke(linewidth=3, foreground="white", alpha=0.20),
                    pe.Normal(),
                ]
            )


# Orthographic projection center (see plotting._PROJECTIONS["orthographic"]).
_ORTHO_CENTER = (5.0, 45.0)


def _angular_distance_deg(lon, lat, lon0, lat0):
    """Great-circle distance (deg) from each (lon, lat) to (lon0, lat0)."""
    lon, lat, lon0, lat0 = map(np.radians, (lon, lat, lon0, lat0))
    cos_d = np.sin(lat) * np.sin(lat0) + np.cos(lat) * np.cos(lat0) * np.cos(lon - lon0)
    return np.degrees(np.arccos(np.clip(cos_d, -1.0, 1.0)))


def _add_wind_arrows(
    ax,
    lon,
    lat,
    u,
    v,
    *,
    extent,
    nx,
    ny,
    center=None,
    radius_deg=70.0,
    pad=0.0,
    scale=95.0,
) -> None:
    """Overlay thinned wind vectors sampled onto a regular lon/lat grid.

    Vectors are the nearest data point to each grid node; ``transform`` rotates
    the east/north components into the map projection. ``center`` (for the global
    orthographic panel) drops grid nodes beyond ``radius_deg`` of the visible
    hemisphere so the far side / limb stays clean. ``pad`` extends the grid
    beyond ``extent`` (fraction per side, with proportionally more nodes) so a
    zoom panel that cartopy grows for aspect still has arrows to its edges; the
    overflow is clipped by the axes.
    """
    lon0, lon1, lat0, lat1 = extent
    if pad:
        dlon, dlat = (lon1 - lon0) * pad, (lat1 - lat0) * pad
        lon0, lon1, lat0, lat1 = lon0 - dlon, lon1 + dlon, lat0 - dlat, lat1 + dlat
        nx, ny = round(nx * (1 + 2 * pad)), round(ny * (1 + 2 * pad))
    gx, gy = np.meshgrid(np.linspace(lon0, lon1, nx), np.linspace(lat0, lat1, ny))
    px, py = gx.ravel(), gy.ravel()
    if center is not None:
        keep = _angular_distance_deg(px, py, *center) < radius_deg
        px, py = px[keep], py[keep]
    _, idx = cKDTree(np.column_stack([lon, lat])).query(np.column_stack([px, py]))
    ax.quiver(
        px,
        py,
        u[idx],
        v[idx],
        transform=ccrs.PlateCarree(),
        color="white",
        alpha=0.75,
        width=0.0032,
        scale=scale,
        scale_units="inches",
        headwidth=3.0,
        headlength=4.0,
        headaxislength=3.5,
        zorder=6,
    )


def _add_regional_borders(ax, *, color: str, linewidth: float) -> None:
    """Add country borders to a regional (zoom) panel (globe has coastlines only)."""
    ax.add_feature(
        cfeature.BORDERS.with_scale("10m"),
        edgecolor=color,
        facecolor="none",
        linewidth=linewidth,
    )


def _draw_extent_box(ax, extent, *, color: str, linewidth: float) -> None:
    """Draw the zoom rectangle (regional extent) on the global panel."""
    lon0, lon1, lat0, lat1 = extent
    ax.plot(
        [lon0, lon1, lon1, lon0, lon0],
        [lat0, lat0, lat1, lat1, lat0],
        transform=ccrs.PlateCarree(),
        color=color,
        linewidth=linewidth,
        zorder=8,
    )


def _add_zoom_callout(
    mpl_fig, global_ax, reg_ax, extent, *, color: str, linewidth: float
):
    """Connect the zoom box on the globe to the regional panel (magnifier effect)."""
    proj = global_ax.projection
    _, lon1, lat0, lat1 = extent
    # Right edge of the box -> left edge of the (right-hand) regional panel.
    top_right = proj.transform_point(lon1, lat1, ccrs.PlateCarree())
    bottom_right = proj.transform_point(lon1, lat0, ccrs.PlateCarree())
    for xy_a, xy_b in ((top_right, (0, 1)), (bottom_right, (0, 0))):
        cp = ConnectionPatch(
            xyA=xy_a,
            coordsA=global_ax.transData,
            xyB=xy_b,
            coordsB=reg_ax.transAxes,
            color=color,
            linewidth=linewidth,
            zorder=20,
            clip_on=False,
        )
        cp.set_in_layout(False)
        mpl_fig.add_artist(cp)


def make_teaser(
    global_state: dict, lam_state: dict, lead_time: int, output: Path, icon_tri=None
) -> Path:
    """Render the nested-zoom teaser figure and return the PNG path."""
    output.mkdir(parents=True, exist_ok=True)
    g_plotter = StatePlotter(
        global_state["longitudes"], global_state["latitudes"], output
    )
    l_plotter = StatePlotter(lam_state["longitudes"], lam_state["latitudes"], output)

    projection = DOMAINS[GLOBAL_DOMAIN]["projection"]  # orthographic for all columns
    switzerland_extent = DOMAINS[REGIONAL_DOMAIN]["extent"]

    # Progressive nested zoom, left to right. ``box`` is the next column's extent,
    # drawn on this panel and connected to the next panel by a callout.
    columns = [
        dict(
            label="Global",
            zoom=False,
            extent=None,
            box=switzerland_extent,
            # Full longitude span so the far-side longitudes visible around the
            # pole get arrows too, but cap the latitude below the pole where the
            # meridians converge and the arrows would bunch into a tight ring.
            arrows=dict(
                extent=[-180, 180, -40, 72],
                nx=60,
                ny=22,
                center=_ORTHO_CENTER,
                radius_deg=84,
                scale=160.0,
            ),
        ),
        dict(
            label="Switzerland",
            zoom=True,
            extent=switzerland_extent,
            box=VAL_CALANCA_EXTENT,
            arrows=dict(extent=switzerland_extent, nx=26, ny=19, pad=0.12),
        ),
        dict(
            label="Val Calanca",
            zoom=True,
            extent=VAL_CALANCA_EXTENT,
            box=None,
            arrows=dict(extent=VAL_CALANCA_EXTENT, nx=14, ny=16, pad=0.12),
            tripcolor=True,
        ),
    ]

    fig = g_plotter.init_geoaxes(
        projection=projection,
        bbox=None,  # start global; zoom panels crop via set_extent
        nrows=len(FIELDS),
        ncols=len(columns),
        name="teaser",
        # Width sized so each column is ~as wide as the globe is tall, otherwise
        # the height-limited globe letterboxes and leaves wide inter-column gaps.
        size=(13.5, 10),
    )
    mpl_fig = fig.fig
    mpl_fig.set_facecolor(BG_COLOR)

    for row, spec in enumerate(FIELDS):
        axes = []
        for col, colcfg in enumerate(columns):
            is_zoom = colcfg["zoom"]
            state = lam_state if is_zoom else global_state
            plotter = l_plotter if is_zoom else g_plotter
            field = _preprocess(spec["key"], state["fields"])
            style = _field_style(spec["key"], is_regional=is_zoom)
            subplot = fig.add_map(row=row, column=col)
            # Every column gets its own colorbar, sized to the column width.
            if colcfg.get("tripcolor"):
                lv = _field_levels(spec["key"], is_regional=True)
                _tripcolor_panel(
                    subplot,
                    field,
                    cmap=_field_cmap(spec["key"]),
                    levels=lv,
                    icon_tri=icon_tri,
                    plotter=plotter,
                    extent=colcfg["extent"],
                )
                _add_column_colorbar(mpl_fig, subplot.ax, spec["key"], is_regional=True)
            else:
                plotter.plot_field(subplot, field, colorbar=False, **style)
                _add_column_colorbar(
                    mpl_fig, subplot.ax, spec["key"], is_regional=is_zoom
                )
            if spec["key"] == "SP_10M":
                _add_wind_arrows(
                    subplot.ax,
                    state["longitudes"],
                    state["latitudes"],
                    state["fields"]["U_10M"],
                    state["fields"]["V_10M"],
                    **colcfg["arrows"],
                )
            if is_zoom:
                subplot.ax.set_extent(colcfg["extent"], crs=ccrs.PlateCarree())
                _add_regional_borders(subplot.ax, color=OUTLINE_COLOR, linewidth=0.8)
            if colcfg["box"] is not None:
                _draw_extent_box(
                    subplot.ax, colcfg["box"], color=CALLOUT_COLOR, linewidth=1.6
                )
            if col == 1:
                subplot.ax.text(
                    0.0,
                    1.02,
                    spec["label"],
                    transform=subplot.ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=12,
                    color=CAPTION_COLOR,
                )
            _strip_chrome(subplot.ax)
            axes.append(subplot.ax)

        # Callouts chain each zoom to the next (globe -> CH -> Val Calanca).
        for src in range(len(columns) - 1):
            _add_zoom_callout(
                mpl_fig,
                axes[src],
                axes[src + 1],
                columns[src]["box"],
                color=CALLOUT_COLOR,
                linewidth=1.3,
            )

    valid = lam_state["valid_time"].strftime("%Y-%m-%d %H:%M UTC")
    mpl_fig.text(
        0.015,
        0.985,
        "Varda-Single",
        ha="left",
        va="top",
        fontsize=14,
        color=CAPTION_COLOR,
    )
    mpl_fig.text(
        0.015,
        0.955,
        f"valid {valid}  ·  +{lead_time} h",
        ha="left",
        va="top",
        fontsize=10,
        color=CAPTION_COLOR,
    )

    # earthkit uses a constrained layout engine; tighten via its padding, and
    # reserve a top band (rect) so the caption sits above the globes, not on them.
    engine = mpl_fig.get_layout_engine()
    if engine is not None:
        try:
            engine.set(
                w_pad=0.005,
                h_pad=0.01,
                wspace=0.0,
                hspace=0.02,
                rect=(0, 0, 1, 0.93),
            )
        except (AttributeError, TypeError):
            pass

    out_png = output / "publication_teaser.png"
    out_pdf = output / "publication_teaser.pdf"
    fig.save(out_pdf, bbox_inches="tight", dpi=300)
    fig.save(out_png, bbox_inches="tight", dpi=300)
    LOG.info("Saved %s", out_png)

    (output / "publication_teaser.html").write_text(
        "<!doctype html><html><body>"
        '<img src="publication_teaser.png" style="max-width:100%"></body></html>'
    )
    return out_png


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="Inference-sandbox GRIB dir for one init_time."
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Reference datetime (init_time), e.g. 202505111200.",
    )
    parser.add_argument(
        "--leadtime", type=int, required=True, help="Lead time in hours."
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--grid",
        type=Path,
        default=Path("/store_new/mch/msopr/ml/grids/icon_grid_0001_R19B08_mch.nc"),
        help="ICON grid file for the true-mesh (tripcolor) zoom. Falls back to a "
        "Delaunay mesh if the file is missing.",
    )
    parser.add_argument(
        "--dark",
        action="store_true",
        help="Dark variant: grey background, white text/ticks, white locator "
        "boxes and connector lines.",
    )
    args = parser.parse_args()

    if args.dark:
        global BG_COLOR, CAPTION_COLOR, CALLOUT_COLOR, GLOW
        BG_COLOR = DARK_BG
        CAPTION_COLOR = "white"
        CALLOUT_COLOR = "white"
        GLOW = True
        plt.rcParams.update(
            {
                "text.color": "white",
                "axes.labelcolor": "white",
                "xtick.color": "white",
                "ytick.color": "white",
                "figure.facecolor": DARK_BG,
                "axes.facecolor": DARK_BG,
                "savefig.facecolor": DARK_BG,
            }
        )

    LOG.info(
        "Teaser: input=%s date=%s leadtime=%s", args.input, args.date, args.leadtime
    )
    icon_tri = None
    if args.grid and args.grid.exists():
        LOG.info("Loading ICON grid %s", args.grid)
        icon_tri = load_icon_triangulation(args.grid)
    else:
        LOG.warning(
            "ICON grid %s not found; Val Calanca uses a Delaunay mesh.", args.grid
        )
    global_state, lam_state = load_states(Path(args.input), args.leadtime)
    make_teaser(global_state, lam_state, args.leadtime, args.output, icon_tri=icon_tri)


if __name__ == "__main__":
    main()
