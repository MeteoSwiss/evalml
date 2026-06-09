# REAL-CH1 multi-output architecture (VMAX_10M) support

This documents how evalml supports the new **multi-output "realv2" anemoi
architecture** and, in particular, the `VMAX_10M` (maximum 10 m wind gust)
diagnostic — including the fixes/workarounds required to make the **showcase**
run end-to-end.

## The architecture

The checkpoint emits **two named output streams** (`metadata_inference.dataset_names
== ["data", "realv2"]`):

- **`data`** — the cutout state: ICON-CH1 1km LAM + AIFS N320 global
  (1,688,650 points, IFS variable names `2t`, `10u`, `tp`, …).
- **`realv2`** — a **diagnostic-only** stream with a single variable **`VMAX_10M`**
  on the REAL-CH1 / ICON-CH1 1km grid (1,147,980 points). It has **no input
  variables** (`data_indices.input == {}`).

## How to run the showcase

```bash
evalml showcase config/forecasters-realv2.yaml
```

This builds the inference env, runs inference for each reference time, normalises the
GRIB output, and renders per-leadtime `VMAX_10M` frames over Switzerland which are
assembled into a GIF (`make_forecast_animation`).

Key config files:

- `config/forecasters-realv2.yaml` — example experiment config.
- `resources/inference/configs/sgm-multidataset-forecaster-realv2-ich1.yaml` — the
  inference template. Routes both streams to GRIB: `data` → ICON LAM + IFS global
  (with cutout masks), `realv2` → `grib/realv2-*.grib` with `write_initial_state: false`
  (the diagnostic has no analysis/step-0 state).
- `resources/inference/metadata/sgm-realv2-ich1-patch.yaml` — metadata patch (see below).
- `resources/inference/templates/templates_index_realch1.yaml` +
  `icon-ch1-shortName=VMAX_10M.grib` — GRIB sample/template for VMAX_10M.

## Fixes / workarounds (why each exists)

Running this architecture surfaced four issues. Each is fixed in-repo; all four were
verified by running inference on GPU and rendering a `VMAX_10M` frame/GIF.

Two of them are genuine anemoi-inference bugs patched at env-build time by
`workflow/scripts/patch_anemoi_inference.py` (called from `inference_create_venv`
after `pip install`, inside the freshly-built venv before it is squashed). Both
patches are idempotent and no-ops once upstream ships the fixes — **TODO:** submit
upstream and delete the script + its call.

### 1. anemoi-inference `EmptyInput` drops the date (upstream bug)

`EmptyInput.create_input_state` returns a state **without** a `date`. For a
diagnostic-only output dataset (`realv2`), *every* input provider is the
`EmptyInput`, so the combined input state has no date and the forecast loop dies in
`add_initial_forcings_to_input_state` with `KeyError: 'date'`. This affects
anemoi-inference 0.10.2 **and** 0.11.1 (latest main).

- **Fix:** `return dict(date=date, fields=dict(), _input=self)`.

### 2. eccodes / eccodes-cosmo-resources version mismatch (segfault)

The checkpoint requirements pin `eccodes==2.39.1` but leave
`eccodes-cosmo-resources-python` unpinned, so the build pulls the latest (2.44.x),
whose definitions are incompatible with eccodes 2.39 and **segfault** when writing
GRIB.

- **Fix:** pin `eccodes-cosmo-resources-python==2.38.3.1` in the run's
  `extra_requirements` (see `config/forecasters-realv2.yaml`).

### 3. VMAX_10M GRIB time-processing assertion + wrong units

Two related problems with the VMAX_10M time encoding:

- The native metadata period is `['650m', '12h']` (a sub-hour start). The GRIB
  time-processing encoder asserts whole-hour steps (`_step_in_hours`) → `AssertionError`.
  **Fix:** `sgm-realv2-ich1-patch.yaml` overrides the realv2 `VMAX_10M` period to
  `['6h', '12h']` (a whole-hour 6 h max window matching the model step). The same patch
  also remaps the `data` stream's IFS names (`2t`, `10u`, `tp`, …) to the COSMO
  shortNames (`T_2M`, `U_10M`, `TOT_PREC`, …) expected by the ICON GRIB templates.
- The `VMAX_10M` GRIB **sample template** must be in **hours**. Extracted straight from
  an ICON source field it carries `stepUnits = minutes`, so the 6 h max window is
  mislabelled as 6 minutes (`stepRange '0m-6m'`) and the step leaks into filenames as
  `_6m`. **Fix:** generate the template by retargeting the (hours) heightAboveGround
  template to `VMAX_10M` @ 10 m (see `icon-ch1_generate_templates.sh`); the result is
  `stepType=max`, `stepUnits=hours`, `stepRange '0-6'`/`'6-12'`.

### 4. anemoi-inference strips path-template format specifiers (upstream bug)

The `@format_dataset_name("path")` decorator substitutes `{dataset}` via
`str.format_map(DefaultFormat(...))`. That call also consumes the `:04` / `:03`
specifiers on the still-unresolved `date`/`time`/`step` placeholders, so
`grib/{date}{time:04}_{step:03}.grib` collapses to `grib/{date}{time}_{step}.grib` and
files land unpadded (`202402010_6.grib`). This affects **every** GRIB output and config.

- **Fix:** substitute only the dataset placeholders, leaving the rest for
  `render_template` to format with the (integer) GRIB key values:
  `kwargs[self.arg] = kwargs[self.arg].replace("{dataset_name}", name).replace("{dataset}", name)`.
  Applied by `patch_anemoi_inference.py`.

With #3 (hours template) and #4 (spec preservation) in place, anemoi writes the
canonical `{prefix}{YYYYMMDDHHMM}_{NNN}.grib` names natively — no post-hoc filename
normalisation is needed.

## Showcase plotting wiring

- `src/plotting/colormap_defaults.py` — `VMAX_10M` colormap (m/s, wind palette).
- `src/plotting/compat.py` — `load_state_from_grib` redirects `VMAX_10M` (a realv2
  param, in `REALV2_PARAMS`) to the sibling `realv2-*.grib` file.
- `workflow/Snakefile` — `showcase_all` renders `VMAX_10M` animations over the
  `switzerland` domain (the realv2 stream is the Alpine LAM only).
- `workflow/rules/plot.smk` — `VMAX_10M` is treated like other period diagnostics
  (lead time 0 skipped).

## Status

Verified end-to-end on GPU: inference produces a valid `realv2` `VMAX_10M` GRIB
(correct COSMO `shortName`, `stepType=max`, realistic gust values), and
`plot_forecast_frame` renders the Switzerland-domain frame. See
`figures/showcase_vmax10m_switzerland_006.png`.
