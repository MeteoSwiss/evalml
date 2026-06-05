# `plotting`

The `plotting` package wraps `earthkit-plots` and `cartopy` for fast,
consistent geographic figures. It is imported by the Marimo notebooks
under `workflow/scripts/plot_*.mo.py` and by the dashboard pipeline.

## Top-level API

```{eval-rst}
.. automodule:: plotting
   :members:
```

`StatePlotter` precomputes a Delaunay triangulation in `__init__`, so
repeated calls to `plot_field` on the same grid are inexpensive. For
orthographic projections, only the visible hemisphere is triangulated —
this is handled by the cached `_orthographic_tri` property.

`DOMAINS` is a small registry mapping domain names (`globe`, `europe`,
`centraleurope`, `switzerland`) to bounding boxes and projections, so
plotting code can stay free of magic numbers:

```python
from plotting import DOMAINS, StatePlotter

plotter = StatePlotter(lon, lat, out_dir)
fig = plotter.init_geoaxes(**DOMAINS["switzerland"])
plotter.plot_field(fig.subplots[0, 0], field, title="T2m")
```

## Colormap loader

```{eval-rst}
.. automodule:: plotting.colormap_loader
   :members:
```

NCL-style `.ct` files live under `resources/report/plotting/`. The loader
returns a dict containing a `ListedColormap`, a `BoundaryNorm`, and the
list of bounds — suitable for direct use with Matplotlib's contour
functions.

## Default colormaps per parameter

```{eval-rst}
.. automodule:: plotting.colormap_defaults
   :members:
```

`CMAP_DEFAULTS` is the lookup table consumed by the Marimo plotting
notebooks. Keys are upper-case parameter names (`T_2M`, `V_10M`,
`TOT_PREC_1H`, `TOT_PREC_6H`, `SP`, `FI_850`, …) and values bundle
`cmap`, `norm`, `units`, and (where appropriate) `vmin`/`vmax`.
Precipitation is keyed per accumulation window because the colour
levels need to scale with the integration period — the `plot_forecast_frame`
rule passes an `--accu` value (in hours) so the notebook can pick the
matching entry. Unknown keys fall back to a viridis colormap with a
warning, so an experiment that tries to plot a brand-new parameter will
produce something readable while you fix the mapping.

## State helpers (`plotting.compat`)

```{eval-rst}
.. automodule:: plotting.compat
   :members:
```

`load_state_from_grib` and `load_state_from_raw` return a dict shaped
like:

```python
{
    "forecast_reference_time": ...,
    "valid_time": ...,
    "longitudes": ...,
    "latitudes": ...,
    "lam_envelope": GeoSeries,
    "fields": {"T_2M": np.ndarray, ...},
}
```

This shape mirrors the in-memory state that anemoi-inference produces,
which is why these helpers are grouped under `compat` — they exist to
let plotting code consume the same shape regardless of whether a forecast
came back as a GRIB directory on disk or a `.npy` snapshot.
