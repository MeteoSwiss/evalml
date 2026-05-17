# Resources

The `resources/` directory ships static assets used by the workflow at
runtime. Nothing under `resources/` is generated; everything is checked
into git and is part of the repository's reproducibility contract.

## `resources/inference/`

Layout:

```text
resources/inference/
├── configs/        # Anemoi inference YAML templates
├── metadata/       # ICON-CH1 patch metadata
├── templates/      # GRIB output templates
└── sandbox/        # Jinja2 README for sandbox zips
```

### `configs/`

Inference templates for forecasters and temporal downscaler ('interpolator'), parameterised by model family (COSMO-2 / ICON-CH1 / IFS) and grid (regional / global). Files include:

- `forecaster.yaml` and `interpolator.yaml` — the defaults referenced by
  `ForecasterConfig` and `InterpolatorConfig` when no `config` is given.
- `sgm-forecaster-global.yaml`, `sgm-forecaster-global_trimedge.yaml` —
  global forecaster variants.
- ICON-CH1- and COSMO-2-specific templates for regional inference.

A run picks one of these via `config:` in the YAML; the workflow then
renders run-specific values (lead time, paths) on top.

### `metadata/`

ICON-CH1 patch metadata for operational and multi-dataset configurations.
These files carry information that the inference pipeline expects but
which is not present in every checkpoint's MLflow metadata.

### `templates/`

GRIB output templates used to give inference output the right edition,
table version, and indexing for COSMO-1E, COSMO-2, ICON-CH1, and IFS. An
`index.yaml` per family points at the matching template files.

### `sandbox/`

A Jinja2 README template (`readme.md.jinja2`) that's rendered into the
sandbox zip created by `inference_create_sandbox`. The rendered README
explains how to extract and use the sandbox.

## `resources/report/`

Layout:

```text
resources/report/
├── dashboard/      # HTML template + script.js for the dashboard
└── plotting/       # NCL .ct colormaps for plotting
```

### `dashboard/`

The dashboard template (`template.html.jinja2`) and its front-end
JavaScript (`script.js`) are read by `report_experiment_dashboard.py`
and embedded into a self-contained HTML file in
`results/{experiment}/dashboard/`. The script tag is inlined; no
external CDN dependencies.

### `plotting/`

NCL-style `.ct` colormap files for weather fields (T2M, UV winds, RH at
various levels). Loaded by `plotting.colormap_loader.load_ncl_colormap`
and exposed through `plotting.colormap_defaults.CMAP_DEFAULTS`.

To add a new colormap:

1. Drop a `.ct` file under `resources/report/plotting/`.
2. Register it in `CMAP_DEFAULTS` in
   `src/plotting/colormap_defaults.py` against the parameter name.
3. Verify with `pytest tests/unit/test_colormaps.py`.
