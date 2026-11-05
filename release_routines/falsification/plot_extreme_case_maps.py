# 2x3 map RC | KENDA | RC-KENDA of one field (bottom row: zoom on the native ICON triangles) for one init and lead time; marks the RC extreme. Run from src/ with the evalml venv
# and ECCODES_DEFINITION_PATH set to the cosmo definitions. Difference panel is in raw units (to match the evalML BIAS colour scales).
import io
import sys
from argparse import ArgumentParser
from datetime import datetime, timedelta
from pathlib import Path

import cartopy.crs as ccrs
import earthkit.plots as ekp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import xarray as xr
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workflow/scripts"))
from plot_forecast_frame import get_style

from data_input import load_forecast_data, load_truth_data
from plotting import DOMAINS, StatePlotter
from plotting.colormap_defaults import CMAP_DEFAULTS

ICON_GRID = Path("/store_new/mch/msopr/ml/grids/icon_grid_0001_R19B08_mch.nc")
plt.style.use(Path(ekp.__file__).parent / "data/styles/earthkit.mplstyle")

p = ArgumentParser()
p.add_argument("--init")
p.add_argument("--step", type=int)
p.add_argument("--rc")
p.add_argument("--truth")
p.add_argument("--out")
p.add_argument("--param")
p.add_argument("--rc_param", help="name in the RC grib if different (e.g. T_G for SKT)")
p.add_argument("--stat", choices=["min", "max"])
p.add_argument("--limit", type=float, help="in display units")
p.add_argument("--scale", type=float, default=1)
p.add_argument("--offset", type=float, default=0)
p.add_argument("--units", default="")
p.add_argument("--domain", default="icon-ch")
a = p.parse_args()
ZOOM_KM = 15
BEYOND = "#00FF00"  # R's "green": colour for values beyond the limit
init, rcp = datetime.strptime(a.init, "%Y%m%d%H%M"), a.rc_param or a.param

rc = load_forecast_data(Path(a.rc), init, [a.step], [rcp]).rename({rcp: a.param})
tr = load_truth_data(Path(a.truth), init, [a.step], [a.param])
lon, lat = rc.longitude.values.ravel(), rc.latitude.values.ravel()
assert np.allclose(lon, tr.longitude.values.ravel(), atol=3e-4) and np.allclose(
    lat, tr.latitude.values.ravel(), atol=3e-4
), "grids differ"
raw = {k: np.asarray(d[a.param].values).ravel() for k, d in [("rc", rc), ("tr", tr)]}
f = {k: v * a.scale + a.offset for k, v in raw.items()}
ext, arg = (np.nanmin, np.nanargmin) if a.stat == "min" else (np.nanmax, np.nanargmax)
k = int(arg(f["rc"]))
beyond = (f["rc"] < a.limit) if a.stat == "min" else (f["rc"] > a.limit)
print(
    f"RC {a.stat} {f['rc'][k]:.1f} {a.units} at lon {lon[k]:.2f} lat {lat[k]:.2f}; KENDA there {f['tr'][k]:.1f}; "
    f"RC points beyond {a.limit}: {int(beyond.sum())}; KENDA {a.stat} {ext(f['tr']):.1f}"
)


def abs_scale():
    """One absolute scale for RC and KENDA in both rows: the evalML one (or 5-unit plasma bins over both fields if there is
    none), plus an extra bin at the limit if the limit lies outside it, so that cells beyond the limit show in BEYOND."""
    cfg = CMAP_DEFAULTS.get(a.param)
    if cfg is None:
        both = np.concatenate([f["rc"], f["tr"]])
        bounds = list(
            np.arange(
                np.floor(np.nanmin(both) / 5) * 5,
                np.ceil(np.nanmax(both) / 5) * 5 + 5,
                5,
            )
        )
        cmap = mcolors.ListedColormap(
            plt.get_cmap("plasma")(np.linspace(0, 1, len(bounds) - 1))
        )
        cmap.set_under(cmap.colors[0])
        cmap.set_over(cmap.colors[-1])
    else:
        bounds, cmap = list(cfg["bounds"]), cfg["cmap"]
    colors, under, over = list(cmap.colors), cmap.get_under(), cmap.get_over()
    if a.stat == "min" and a.limit < bounds[0]:
        bounds, colors, under = [a.limit] + bounds, [under] + colors, BEYOND
    if a.stat == "max" and a.limit > bounds[-1]:
        bounds, colors, over = bounds + [a.limit], colors + [over], BEYOND
    cmap = mcolors.ListedColormap(colors)
    cmap.set_under(under)
    cmap.set_over(over)
    return cmap, bounds, mcolors.BoundaryNorm(bounds, cmap.N)


abs_cmap, abs_bounds, abs_norm = abs_scale()
LIMIT_BIN = a.limit in (
    abs_bounds[0],
    abs_bounds[-1],
)  # extra bin added: show the limit as a tick
NEXT_EDGE = (
    abs_bounds[1] if a.stat == "min" else abs_bounds[-2]
)  # its other edge: label dropped, it would overlap the limit
limit_ticks = lambda ticks: sorted((set(ticks) - {NEXT_EDGE}) | {a.limit})
abs_style = {
    "style": ekp.styles.Style(
        colors=abs_cmap,
        vmin=abs_bounds[0],
        vmax=abs_bounds[-1],
        extend="both",
        units=a.units,
    ),
    "norm": abs_norm,
}
abs_style["style"]._kwargs["levels"] = abs_bounds  # as in plot_forecast_frame.get_style
panels = [
    ("Release-Candidate", f["rc"], abs_style, a.units),
    ("KENDA-CH1 (truth)", f["tr"], abs_style, a.units),
    (
        "RC - KENDA",
        raw["rc"] - raw["tr"],
        get_style(f"{a.param}.BIAS.map"),
        CMAP_DEFAULTS[f"{a.param}.BIAS.map"]["units"],
    ),
]
plotter = StatePlotter(lon, lat, Path(a.out).parent)
fig = ekp.Figure(
    crs=DOMAINS[a.domain]["projection"],
    domain=DOMAINS[a.domain]["extent"],
    rows=1,
    columns=3,
    figsize=(15, 5.6),
)
for c, (label, vals, style, _) in enumerate(panels):
    plotter.plot_field(
        fig.add_map(row=0, column=c),
        vals,
        title=label,
        colorbar=True,
        gridline_labels=False,
        **style,
    )
fig._release_queue()
m = fig.fig
for c, ax in enumerate(ax for ax in m.axes if hasattr(ax, "projection")):
    ax.plot(
        lon[k],
        lat[k],
        "o",
        mfc="none",
        mec="magenta",
        ms=22,
        mew=2.5,
        transform=ccrs.PlateCarree(),
    )
    ax.text(
        0.985,
        0.02,
        f"{a.stat} {ext(panels[c][1]):.1f} {panels[c][3]}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"fc": "white", "ec": "#aaaaaa", "pad": 2.5},
    )
for cax in (
    [ax for ax in m.axes if not hasattr(ax, "projection")][:2] if LIMIT_BIN else []
):
    cax.set_xticks(limit_ticks(cax.get_xticks()))
v = init + timedelta(hours=a.step)
m.suptitle(
    f"{a.param} valid {v:%Y-%m-%d %H:%M} UTC   (init {init:%Y-%m-%d %H:%M} UTC, +{a.step:03d} h)   circle = RC {a.stat}imum, "
    f"{int(beyond.sum())} RC grid points {'<' if a.stat == 'min' else '>'} {a.limit:g} {a.units}",
    fontsize=12,
)
top = io.BytesIO()
m.savefig(top, dpi=100, bbox_inches="tight")

# zoom row: +-ZOOM_KM around the extreme on the native ICON triangles (after projects/2026_09_01_plot_scoremaps),
# cells beyond the limit outlined in black
with xr.open_dataset(ICON_GRID) as g:
    vlon, vlat = np.degrees(g["vlon"].values), np.degrees(g["vlat"].values)
    triangles = g["vertex_of_cell"].values.T - 1
    clon, clat = np.degrees(g["clon"].values), np.degrees(g["clat"].values)
assert (
    len(triangles) == lon.size
    and np.abs(clon - lon).max() < 3e-4
    and np.abs(clat - lat).max() < 3e-4
), "ICON cell order does not match the data"
la0, lo0 = lat[k], lon[k]
dla = ZOOM_KM / 111.2
dlo = dla / np.cos(np.radians(la0))
zoom = [lo0 - dlo, lo0 + dlo, la0 - dla, la0 + dla]
z = (np.abs(lat - la0) < 1.3 * dla) & (np.abs(lon - lo0) < 1.3 * dlo)
tri, tri_beyond = (
    mtri.Triangulation(vlon, vlat, triangles[m]) for m in (z, z & beyond)
)


def cmap_norm(name):
    cfg = CMAP_DEFAULTS.get(name)
    return {
        "cmap": cfg["cmap"],
        "norm": mcolors.BoundaryNorm(cfg["levels"], cfg["cmap"].N),
    }


abs_cn = {"cmap": abs_cmap, "norm": abs_norm}
zf, zax = plt.subplots(
    1, 3, figsize=(15, 5.6), subplot_kw={"projection": DOMAINS[a.domain]["projection"]}
)
for ax, (label, vals, _, u), cn in zip(
    zax, panels, [abs_cn, abs_cn, cmap_norm(f"{a.param}.BIAS.map")]
):
    ax.set_extent(zoom, crs=ccrs.PlateCarree())
    mp = ax.tripcolor(
        tri, facecolors=vals[z], shading="flat", transform=ccrs.PlateCarree(), **cn
    )
    ax.triplot(
        tri, color="black", linewidth=0.1, alpha=0.35, transform=ccrs.PlateCarree()
    )
    ax.triplot(tri_beyond, color="black", linewidth=1.5, transform=ccrs.PlateCarree())
    ax.set_title(
        f"{label}, zoom ±{ZOOM_KM:g} km   ({a.stat} {ext(vals[z]):.1f} {u})",
        fontsize=11,
    )
    cb = zf.colorbar(
        mp,
        ax=ax,
        orientation="horizontal",
        fraction=0.046,
        pad=0.03,
        label=u,
        extend="both",
    )
    if cn is abs_cn and LIMIT_BIN:
        cb.set_ticks(limit_ticks(cb.get_ticks()))
zf.suptitle(
    f"zoom around {lo0:.2f}°E {la0:.2f}°N; black outline = RC cells {'<' if a.stat == 'min' else '>'} {a.limit:g} {a.units}",
    fontsize=12,
    y=1.02,
)
zf.tight_layout()
bot = io.BytesIO()
zf.savefig(bot, dpi=100, bbox_inches="tight")

imgs = [Image.open(b) for b in (top, bot)]
w = max(i.width for i in imgs)
out = Image.new("RGB", (w, sum(i.height for i in imgs)), "white")
out.paste(imgs[0], ((w - imgs[0].width) // 2, 0))
out.paste(imgs[1], ((w - imgs[1].width) // 2, imgs[0].height))
out.save(a.out)
print(a.out)
