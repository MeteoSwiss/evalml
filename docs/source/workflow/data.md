# Data

The `workflow/rules/data.smk` rule module owns one rule and a tiny piece of conditional logic at parse time:

```python
if config["truth"]["root"].endswith("peakweather"):
    output_peakweather_root = config["truth"]["root"]
else:
    output_peakweather_root = OUT_ROOT / "data/observations/peakweather"
```

This means: if the user pointed `truth.root` at an existing PeakWeather
location, write directly there; otherwise cache observations under
`OUT_ROOT/data/observations/peakweather` so multiple experiments can share
the same download.

## `data_download_obs_from_peakweather`

```{eval-rst}
.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **Local rule**
     - yes — runs on the workflow head node; never submitted to SLURM.
   * - **Output**
     - ``output_peakweather_root`` (a directory)
   * - **Run body**
     - Instantiates ``peakweather.dataset.PeakWeatherDataset(root=output.root)``,
       which downloads the dataset from Hugging Face on first use.
```

The rule has no `shell:` block; the entire download is implemented inside a
Snakemake `run:` block, which means PeakWeather is imported in the Snakemake
process itself.

## How baseline data is consumed

Baselines are not produced by a `data_*` rule; they are read directly by
`verification_metrics_baseline` from `{root}/FCST{YY}.zarr` (where `YY` is
the two-digit year extracted from `init_time`). The shape is determined by
the upstream archive and EvalML does not currently transform it before
verification.

A standalone helper script `workflow/scripts/data_extract_baseline.py` exists
for ad-hoc baseline extraction, it is not invoked by any rule in the default pipeline and needs to be run manually.

## Ground-truth dispatch

`src/data_input/__init__.py` exposes a `load_truth_data(root, ...)` function
that dispatches based on `root`:

- A path ending in `.zarr` is loaded via `load_analysis_data_from_zarr`.
- A path ending in `peakweather` (or pointing inside the PeakWeather cache)
  is loaded via `load_obs_data_from_peakweather`.

`load_forecast_data` does the analogous dispatch between GRIB directories
(EvalML inference output) and Zarr baselines. See the
[data_input module reference](../modules/data_input.md) for signatures.
