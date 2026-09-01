"""Map of Switzerland: the three forecast regions + the SwissMetNet stations
that were actually used in the paper's verification.

Shades the three main forecast regions (Jura, Swiss Plateau/Mittelland, Alps)
from their LV95 (EPSG:2056) shapefiles, and overlays the stations that
contributed to the paper's station verification — read straight from the evalml
result files ``<PARAM>_<lt>_caa0.nc`` (caa0 = the SwissMetNet truth hash), which
store the exact per-parameter station coordinates. Stations are coloured by
observation completeness:

  * All parameters    — temperature and pressure (the full automatic sites)
  * Without pressure  — temperature/wind but no pressure
  * Precipitation only

For the paper run this is ~440 distinct stations (T_2M 187, SP_10M 151, PMSL 59,
TOT_PREC6 277) — NOT the ~2000-station raw DWH candidate pool. The four
case-study meteogram sites are ringed. Region fills use the shared colourblind-
safe palette (``style.REGION_COLORS``).

Single-column publication figure, shared print style, no on-figure title. Writes
a PNG (250 dpi) + a vector PDF. The station dots come from the result files;
``.env`` (jretrieve) is only needed for the four case-study coordinates.

    set -a; source .env; set +a
    python workflow/scripts/plot_region_map.py
"""

import logging
from argparse import ArgumentParser
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely
import xarray as xr
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from data_input import jretrieve as jr
from evalml.publication.style import (
    COLOR_VARDA,
    figure_width,
    mplstyle_path,
    region_color,
    region_label,
)

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

REGION_DIR = Path("/scratch/mch/bhendj/regions/Prognoseregionen_LV95_20220517")

# Regions to shade, in draw / legend order. Colours (colourblind-safe) come from
# the shared publication palette (style.REGION_COLORS).
REGIONS = ("jura", "mittelland", "alpen")

# Stations to highlight (nat_abbr) — the meteogram case-study sites. Each maps to
# a text-label offset in points, tuned to keep the code clear of the ring marker.
HIGHLIGHT_STATIONS = {
    "SIO": (8, 0),    # Sion
    "KLO": (8, 0),    # Zürich / Kloten
    "ALT": (8, 0),    # Altdorf
    "LUG": (8, 0),    # Lugano
}

# Map projection + crop. Orthographic tangent near the country centre keeps the
# shape faithful at this scale; extent is padded lon/lat around Switzerland.
PROJ = ccrs.Orthographic(central_longitude=8.2, central_latitude=46.8)
# Extra room on the left (min lon 4.5) so the station legend clears the country.
EXTENT = [4.5, 10.6, 45.7, 47.9]  # minlon, maxlon, minlat, maxlat

# Copernicus DEM tile covering Switzerland (N40-N50, E00-E20), a 30 m lat/lon
# grid. READ-ONLY external data (not ours) — never write under this path.
COPERNICUS_DEM_TILE = Path(
    "/scratch/mch/csteger/projects/topo_comparison/Copernicus_DEM"
    "/COPERNICUS_N50-N40_E000-E020.nc"
)


# evalml station-verification result files. `<PARAM>_<lt>_caa0.nc` (caa0 = the
# SwissMetNet truth hash) hold the exact stations used per parameter, with
# latitude/longitude coords on a `values` dim. Each parameter tags a network the
# station belongs to; the sibling `_2b83`/`_ffde` files (~1.1M values) are the
# gridded scoremaps, not stations, so the `caa0` suffix is essential.
VERIF_STORE = Path("/store_new/mch/msopr/ml/evaluation/varda-single_paper/data")
VERIF_PARAM_TAGS = {
    "T_2M": "temp",
    "SP_10M": "wind",
    "PMSL": "pressure",
    "TOT_PREC6": "precip",
}
HIGHLIGHT_META_PARAM = "tre200s0"  # any current param; only used to fetch coords

# Station categories encoded by marker SHAPE (all black, thin outlines), in draw
# + legend order (least → most complete drawn last, so the sparse "all
# parameters" sites sit on top): cross → open triangle → open circle.
STATION_COLOR = "black"
STATION_CATEGORIES = [
    ("precip", "Precipitation only",
     dict(marker="+", s=7, c=STATION_COLOR, linewidths=0.35), 3.1),
    ("no_pressure", "Without pressure",
     dict(marker="^", s=8, facecolors="none", edgecolors=STATION_COLOR,
          linewidths=0.35), 3.3),
    ("all", "All parameters",
     dict(marker="o", s=6, facecolors="none", edgecolors=STATION_COLOR,
          linewidths=0.35), 3.5),
]


def _classify(tags: set) -> str:
    """Category from the parameter networks a station belongs to.

    Partitions every station: full sites (temperature + pressure) → ``all``;
    pure rain gauges → ``precip``; anything else (temperature/wind, or the odd
    pressure-only site, but not the full suite) → ``no_pressure``.
    """
    if "temp" in tags and "pressure" in tags:
        return "all"
    if "precip" in tags and not (tags & {"temp", "wind", "pressure"}):
        return "precip"
    return "no_pressure"


def load_verified_stations() -> gpd.GeoDataFrame:
    """Stations used in the paper's verification, from the evalml result files.

    Each ``*_caa0.nc`` file carries the station code (``nat_abbr``, e.g. ``ALT``)
    on its ``values`` coordinate. Union the stations by that code — robust to a
    site's sensors being recorded at slightly different coordinates across the
    parameter files — tagging each with the parameter networks it reports and
    keeping one representative location (the first file it appears in). The
    category then encodes observation completeness.
    """
    tags_by_id: dict = {}
    coord_by_id: dict = {}
    per_param: dict = {}
    for param, tag in VERIF_PARAM_TAGS.items():
        matches = sorted(VERIF_STORE.glob(f"**/{param}_*_caa0.nc"))
        if not matches:
            LOG.warning("no station-verification file found for %s", param)
            continue
        ds = xr.open_dataset(matches[0])
        codes = np.asarray(ds["values"].values).astype(str)
        lat = np.asarray(ds["latitude"].values, dtype=float)
        lon = np.asarray(ds["longitude"].values, dtype=float)
        per_param[param] = len(codes)
        for code, la, lo in zip(codes, lat, lon):
            tags_by_id.setdefault(code, set()).add(tag)
            coord_by_id.setdefault(code, (la, lo))  # keep first-seen location

    lons, lats, cats = [], [], []
    for code, tags in tags_by_id.items():
        la, lo = coord_by_id[code]
        lons.append(lo)
        lats.append(la)
        cats.append(_classify(tags))
    gdf = gpd.GeoDataFrame(
        {"category": cats},
        geometry=gpd.points_from_xy(lons, lats),
        crs="EPSG:4326",
    )
    # Station-count info for the figure caption (NOT drawn on the plot).
    LOG.info("=== station counts for caption ===")
    LOG.info("  stations with 2m temperature : %d", per_param.get("T_2M", 0))
    LOG.info("  stations with pressure       : %d", per_param.get("PMSL", 0))
    LOG.info("  stations with total precip   : %d", per_param.get("TOT_PREC6", 0))
    LOG.info("  stations with wind speed     : %d", per_param.get("SP_10M", 0))
    counts = gdf["category"].value_counts().to_dict()
    LOG.info("  --- by symbol (union by station code) ---")
    LOG.info("  all parameters (filled dot)  : %d", counts.get("all", 0))
    LOG.info("  without pressure (open dot)  : %d", counts.get("no_pressure", 0))
    LOG.info("  precipitation only (cross)   : %d", counts.get("precip", 0))
    LOG.info("  distinct stations (total)    : %d", len(gdf))
    return gdf


def fetch_highlight_coords() -> gpd.GeoDataFrame:
    """Coordinates of the case-study sites (by nat_abbr) via jretrieve."""
    jr.check_prerequisites("prod")
    selector = "jretrievedwh:locations=" + ",".join(HIGHLIGHT_STATIONS)
    stations, stage, seq_type = jr.parse_selection(selector)
    meta = jr.fetch_meta(
        stations=stations, params=[HIGHLIGHT_META_PARAM], seq_type=seq_type, stage=stage
    )
    cat = jr.StationCatalog.from_meta(meta)
    return gpd.GeoDataFrame(
        {"nat_abbr": cat.nat_abbr},
        geometry=gpd.points_from_xy(cat.longitude, cat.latitude),
        crs="EPSG:4326",
    )


def _outer_boundary(geom):
    """Geometry with interior rings dropped.

    The three region shapefiles don't align perfectly, so their union has tiny
    sliver holes; outlining them would draw specks inside the country. Keep only
    the exterior ring(s) for a clean national outline.
    """
    from shapely.geometry import MultiPolygon, Polygon

    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    return MultiPolygon([Polygon(g.exterior) for g in geom.geoms])


def load_hillshade(bounds, region_union, target_px: int = 1400, vert_exag: float = 2.0):
    """Shaded-relief intensity of the terrain inside Switzerland.

    Reads a strided Switzerland subset of the (READ-ONLY) Copernicus DEM tile,
    computes a hillshade, and masks it to the region union so only the country is
    shaded. Returns ``(masked_intensity, imshow_extent)`` for ``ax.imshow``.
    """
    lon0, lon1, lat0, lat1 = bounds
    da = xr.open_dataset(COPERNICUS_DEM_TILE)["elevation"]  # lat descending
    da = da.sel(lat=slice(lat1 + 0.3, lat0 - 0.3), lon=slice(lon0 - 0.2, lon1 + 0.2))
    step = max(1, int(da.sizes["lon"] / target_px))
    da = da.isel(lat=slice(None, None, step), lon=slice(None, None, step))
    elev = da.values.astype("float32")
    lons = da["lon"].values
    lats = da["lat"].values

    lon2d, lat2d = np.meshgrid(lons, lats)
    inside = shapely.contains_xy(region_union, lon2d, lat2d)

    lat_mid = float(np.mean(lats))
    dx = abs(lons[1] - lons[0]) * 111320.0 * np.cos(np.radians(lat_mid))
    dy = abs(lats[1] - lats[0]) * 110540.0
    intensity = LightSource(azdeg=315, altdeg=45).hillshade(
        elev, vert_exag=vert_exag, dx=dx, dy=dy
    )
    extent = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]
    return np.ma.masked_where(~inside, intensity), extent


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outfn",
        default="output/figures/regions/switzerland_regions_stations.png",
        help="Output PNG path (a .pdf sibling is also written).",
    )
    args = parser.parse_args()

    # Stations actually used in the verification (from the evalml result files).
    stations = load_verified_stations()

    # Region polygons, reprojected LV95 -> WGS84 for plotting.
    regions = {
        name: gpd.read_file(REGION_DIR / f"{name}.shp").to_crs("EPSG:4326")
        for name in REGIONS
    }
    region_union = gpd.GeoSeries(
        [g.union_all() for g in regions.values()], crs="EPSG:4326"
    ).union_all()

    # Case-study meteogram sites (coordinates via jretrieve).
    highlights = fetch_highlight_coords()
    missing = set(HIGHLIGHT_STATIONS) - set(highlights["nat_abbr"])
    if missing:
        LOG.warning("highlight stations not found: %s", ", ".join(sorted(missing)))

    plt.style.use(mplstyle_path())
    width = figure_width(1)  # single column, 3.35 in
    fig = plt.figure(figsize=(width, width * 0.72))
    ax = fig.add_subplot(111, projection=PROJ)
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())

    # Shaded-relief base (Copernicus DEM), masked to Switzerland. Optional: skip
    # gracefully if the external DEM isn't reachable.
    try:
        hillshade, hs_extent = load_hillshade(EXTENT, region_union)
        ax.imshow(
            hillshade, extent=hs_extent, transform=ccrs.PlateCarree(),
            origin="upper", cmap="gray", vmin=0, vmax=1,
            interpolation="bilinear", zorder=0,
        )
    except (FileNotFoundError, OSError) as exc:
        LOG.warning("hillshade unavailable (%s); drawing without relief", exc)

    # Shaded regions — translucent fill over the relief (station markers on top).
    for name in REGIONS:
        color = region_color(name)
        ax.add_geometries(
            regions[name].geometry,
            crs=ccrs.PlateCarree(),
            facecolor=color,
            edgecolor=color,
            alpha=0.35,
            linewidth=0.4,
            zorder=1,
        )

    # Neighbouring-country borders + lakes, kept light so they recede.
    ax.add_feature(
        cfeature.BORDERS.with_scale("10m"), edgecolor="0.6", linewidth=0.3, zorder=2
    )
    ax.add_feature(
        cfeature.LAKES.with_scale("10m"),
        facecolor="white",
        edgecolor="0.7",
        linewidth=0.25,
        zorder=2,
    )
    # A single crisp national outline (the region union traces the Swiss border),
    # exterior only so the tiny inter-region sliver holes don't draw as specks.
    ax.add_geometries(
        [_outer_boundary(region_union)],
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="black",
        linewidth=0.7,
        zorder=2.5,
    )

    # SwissMetNet stations, drawn per parameter category (least → most complete).
    for key, _label, style, zorder in STATION_CATEGORIES:
        sub = stations[stations["category"] == key]
        if sub.empty:
            continue
        ax.scatter(
            sub.geometry.x, sub.geometry.y,
            transform=ccrs.PlateCarree(), zorder=zorder, **style,
        )

    # Highlighted case-study stations: a ring marker (white fill, dark-red edge +
    # centre dot) reads cleanly over the dots without shouting.
    ax.scatter(
        highlights.geometry.x, highlights.geometry.y,
        transform=ccrs.PlateCarree(),
        s=20, marker="o", facecolors="white", edgecolors=COLOR_VARDA,
        linewidths=0.6, zorder=5,
    )
    ax.scatter(
        highlights.geometry.x, highlights.geometry.y,
        transform=ccrs.PlateCarree(),
        s=1.8, marker="o", c=COLOR_VARDA, linewidths=0.0, zorder=6,
    )
    to_display = ccrs.PlateCarree()._as_mpl_transform(ax)
    for _, row in highlights.iterrows():
        dx, dy = HIGHLIGHT_STATIONS[row["nat_abbr"]]
        ax.annotate(
            row["nat_abbr"],
            xy=(row.geometry.x, row.geometry.y),
            xycoords=to_display,
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6,
            color="black",
            va="center",
            zorder=7,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.7", lw=0.3, alpha=0.9),
        )

    # Thin black frame (no ticks/labels), matching the shared axes style.
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor("black")
        sp.set_linewidth(0.4)

    # Station-type legend in the (widened) empty NW corner, marker styles matching
    # the scatters. No counts on the figure (they go in the caption); no frame.
    station_handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=3.2,
               markerfacecolor="none", markeredgecolor=STATION_COLOR, markeredgewidth=0.45,
               label="All parameters"),
        Line2D([], [], linestyle="none", marker="^", markersize=3.4,
               markerfacecolor="none", markeredgecolor=STATION_COLOR, markeredgewidth=0.45,
               label="Without pressure"),
        Line2D([], [], linestyle="none", marker="+", markersize=3.6,
               markeredgecolor=STATION_COLOR, markeredgewidth=0.6,
               label="Precipitation only"),
        Line2D([], [], linestyle="none", marker="o", markersize=4,
               markerfacecolor="white", markeredgecolor=COLOR_VARDA, markeredgewidth=0.7,
               label="Case-study sites"),
    ]
    ax.legend(
        handles=station_handles, loc="upper left", frameon=False,
        labelspacing=0.6, handletextpad=0.6, borderpad=0.4,
    )

    # Region legend below the map (shared style, frameon=False).
    region_handles = [
        Patch(facecolor=region_color(name), edgecolor=region_color(name),
              alpha=0.4, label=region_label(name))
        for name in REGIONS
    ]
    fig.legend(
        handles=region_handles, loc="lower center", ncol=len(region_handles),
        bbox_to_anchor=(0.5, 0.0), columnspacing=1.2, handletextpad=0.4,
    )

    outfn = Path(args.outfn)
    outfn.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.07, 1, 1], pad=0.2)
    fig.savefig(outfn, dpi=250)
    fig.savefig(outfn.with_suffix(".pdf"))  # vector for the paper
    plt.close(fig)
    LOG.info("saved: %s (+ .pdf)", outfn)


if __name__ == "__main__":
    main()
