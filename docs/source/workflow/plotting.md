# Plotting

`workflow/rules/plot.smk` produces three kinds of artefact:

- **Forecast frames** — single-time PNGs of a parameter over a region.
- **Forecast animations** — frames stitched into an animated GIF.
- **Meteograms** — per-station time series PNGs with forecast, truth, and
  baselines overlaid.

Both `plot_forecast_frame` and `plot_meteogram` shell out to **Marimo**
notebooks (`*.mo.py`) that can also be edited interactively, which is the
key reason the scripts use the `.mo.py` extension.

## `plot_forecast_frame`

| Property | Value |
| --- | --- |
| Output | `data/runs/{run_id}/{init_time}/frames/frame_{leadtime}_{param}_{region}.png` |
| Wildcard constraint | `leadtime=r"\d+"` |
| Resources | `slurm_partition=postproc`, 1 CPU, 10 min |

The rule sets `ECCODES_DEFINITION_PATH` to the project venv's bundled COSMO
definitions, then runs:

```bash
python scripts/plot_forecast_frame.mo.py \
    --input {grib_out_dir} \
    --date {init_time} \
    --outfn {output} \
    --param {param} \
    --leadtime {leadtime} \
    --region {region}
```

Marimo notebooks accept the same CLI args, which means you can switch from a
batch render to an interactive edit by toggling the rule to `localrule: True`
and replacing `python` with `marimo edit`. The intended interactive command
is left as a comment in the rule body.

## `make_forecast_animation`

A localrule that takes the expanded list of frame outputs and runs:

```bash
convert -delay {delay} -loop 0 {input} {output}
```

`{delay}` is computed as `10 * step` — i.e. forecasts with a 6-hourly cadence
get a 60 ms inter-frame delay. `convert` is part of ImageMagick.

## `plot_meteogram`

| Property | Value |
| --- | --- |
| Output | `results/{showcase}/{run_id}/{init_time}/{init_time}_{param}_{sta}.png` |
| Resources | `postproc`, 1 CPU, 10 min |

Inputs:

- The inference `.ok` for the forecast.
- `truth.root` for analysis.
- `data_download_obs_from_peakweather` output for station observations.

The shell block builds a CLI argument list including any number of
baselines (collected by `_get_available_baselines(wc)`), then invokes the
Marimo notebook. As with frame plots, the rule has commented-out variants
that show how to launch `marimo edit` for live development.

## Style and colormap choices

The plotting helpers used by both notebooks live in `src/plotting/`:

- `StatePlotter` — wraps an earthkit-plots GeoAxes; pre-computes a Delaunay
  triangulation so successive `plot_field` calls on the same grid are cheap.
  Includes a fast-path for orthographic projections (only the visible
  hemisphere is triangulated).
- `DOMAINS` — named extents for `globe`, `europe`, `centraleurope`,
  `switzerland`, with sensible default projections.
- `colormap_loader.load_ncl_colormap` — parses NCL `.ct` files from
  `resources/report/plotting/`.
- `colormap_defaults.CMAP_DEFAULTS` — a `defaultdict` keyed by parameter
  name (`T_2M`, `V_10M`, `TOT_PREC`, …) that returns sensible
  `cmap`/`norm`/`units` for plotting.

If you add a new variable, prefer extending `CMAP_DEFAULTS` over hard-coding
colours in the notebook — the fallback returns viridis with a warning,
which is correct but ugly.
