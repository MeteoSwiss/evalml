# Decouple publication plotting from Snakemake

**Date:** 2026-08-04
**Status:** Approved design, ready for implementation planning
**Branch:** paper-figures

## Problem

The publication-figures subsystem is tightly coupled to Snakemake. The three
figure rules (`publication_figures`, `publication_meteogram`,
`publication_scoremaps` in `workflow/rules/publication.smk`) are `localrule`
wrappers that shell out to `python -m evalml.publication ...`, yet they carry
full DAG input declarations. This creates three concrete pain points:

1. **Duplicated path logic.** `_pub_scoremap_inputs` rebuilds scoremap file
   paths from in-memory globals using the *same* template `manifest.py` already
   captures — two sources of truth that can drift.
2. **Rerun cascades.** Because the figure rules are part of the DAG, Snakemake's
   `code`/`params` rerun-triggers make tweaking a plot want to recompute
   upstream work. Getting Snakemake to "just re-render" is cumbersome.
3. **Config proliferation.** Multiple near-identical paper configs exist largely
   to feed different figure variants through the DAG.

The publication figures are, by nature, hand-tuned artifacts for a manuscript.
They should be produced interactively, not through a build graph.

## Goal

Decouple result generation from publication plotting. Snakemake produces the
**results + a manifest** and stops. All publication plotting happens in
**standalone Jupyter notebooks** driven only by the manifest. The two worlds
share exactly one contract: `manifest.json`.

## Design

### The boundary

```
evalml publication <config>          # inference → verification → manifest.json  (STOP)
        │
        │  results on disk + output/publication/<truth>/manifest.json
        ▼
notebooks/publication/leadtime.ipynb   ← standalone; imports Manifest; plots
notebooks/publication/meteogram.ipynb
notebooks/publication/scoremaps.ipynb
```

Snakemake never invokes a notebook; a notebook never invokes Snakemake.

`publication_all` is redefined so one command still reproducibly produces
*everything a figure needs*, then stops before plotting:

```
publication_all  →  manifest.json                       (publication_manifest)
                 +  all verif_aggregated_*.nc            (EXPERIMENT_PARTICIPANTS)
                 +  scoremap *.nc  (only when publication.scoremaps.enabled)
```

These are the exact result-dependencies the deleted figure rules used to
declare (the candidate-verif dependency that the meteogram rule needed is
subsumed by `EXPERIMENT_PARTICIPANTS`; the scoremap NC list is still computed
from in-memory globals via the surviving helpers). It keeps the reproducible
chain intact minus the plotting step.

### Components

**Kept, unchanged (the value of the prior refactor):**

- `src/evalml/publication/manifest.py` — `build_manifest`, `load_manifest`,
  `write_manifest`.
- `src/evalml/publication/resolver.py` — `Manifest` / `Participant` /
  `validate_request` / `ResolutionError`. The `Manifest` class already exposes
  everything a notebook needs: `verif_paths()`, `grib_dir()`, `scoremap_path()`,
  `meteogram_baseline_specs()`, `get_candidate()`, `resolve_baseline()`,
  `validate_request()`.
- `src/evalml/config.py` publication validation — coherence checks still guard
  the manifest at config-load time.
- `workflow/rules/publication.smk` → **only** the `publication_manifest` rule.

**Deleted (the coupling + the redundant render path):**

- `workflow/rules/publication.smk`: the three figure rules
  (`publication_figures`, `publication_meteogram`, `publication_scoremaps`) and
  the meteogram data-dep helper `_meteogram_data_dep` (subsumed — see below).
- `src/evalml/publication/cli.py` and `src/evalml/publication/__main__.py` —
  the standalone renderer. The notebooks replace it.

**Kept but rewired (result-dependency listing, not plotting):**

In the paper configs `experiment.scoremaps.enabled` is `false` while
`publication.scoremaps.enabled` is `true`, so the publication target is the
*only* thing that pulls scoremap NC production into the DAG. The scoremap
input-listing helpers therefore **survive** and are re-wired to feed
`publication_all` (they compute file paths from in-memory globals — required,
because the manifest file a sibling rule produces cannot be read at
DAG-build time):

- `_pub_scoremap_inputs`, `_pub_scoremap_cfg`, `_pub_scoremap_leadtimes`,
  `_pub_candidate_run_id` — kept; now feed `publication_all` instead of the
  deleted plot rule. Scoremap NC files are `…/scoremaps/{param}_{leadtime}_
  {TRUTH_HASH}.nc`.
- `_meteogram_data_dep` is deleted: the candidate's `verif_aggregated` (already
  in `EXPERIMENT_PARTICIPANTS`) guarantees inference ran for every reftime, so
  the meteogram's GRIB is present without a separate dep.
- `workflow/scripts/publication_figures.py`, `publication_meteogram.py`,
  `publication_scoremaps.py` — the marimo apps. Their plotting logic is *ported*
  into the notebooks, then the files are removed.

**Moved / promoted into the package:**

- `workflow/scripts/publication_style.py` → `src/evalml/publication/style.py`
- `workflow/scripts/publication.mplstyle` →
  `src/evalml/publication/publication.mplstyle`, loaded via `importlib.resources`
  through a new `style.mplstyle_path()` helper.

The promotion is done additively first (the package module is the canonical
copy; the originals stay until the marimo scripts that import them are deleted),
so every intermediate commit stays green. One *other* consumer must migrate:
`workflow/scripts/plot_meteogram_region.py` (a main-workflow paper script, kept)
imports `publication_style` **and** applies `publication.mplstyle`. It is
repointed to `evalml.publication.style` + `style.mplstyle_path()` as part of the
final cleanup, after which the two `workflow/scripts/` originals are deleted.

**New:**

- `notebooks/publication/{leadtime,meteogram,scoremaps}.ipynb`

### Two shared, importable surfaces

1. **`evalml.publication.style`** — colors, `line_style()`, `param_label()`,
   `SKILL_CMAP`, `SKILL_LEVELS`, `SCORE_LABELS`, `PARAM_LABELS`, plus a new
   `mplstyle_path()` returning the packaged `.mplstyle`. *Common styling, one
   place.* Carried over unchanged except for the `mplstyle_path()` addition and
   the import path.
2. **`evalml.publication.manifest` / `.resolver`** — data resolution, one place.

Everything else — the matplotlib layout, panels, annotations — lives **in the
notebook cells**, visible and directly editable. This is the "plotting in the
notebook, styling shared" arrangement.

### Imports the notebooks need (beyond style)

The ported plotting logic reuses existing helpers. Most are importable as-is
(the editable install exposes `src/`): `data_input`, `verification.spatial`,
`plotting` (`DOMAINS`, `StatePlotter`), and `evalml.publication.*`. Two helper
modules live in `workflow/scripts/` and are **shared with the main workflow**,
so they stay put and the notebooks reach them via a one-line bootstrap that
prepends `workflow/scripts` (and, defensively, `src`) to `sys.path`:

- `verification_plot_metrics` → `_ensure_unique_lead_time`,
  `_select_best_sources`, `decode_metric` (leadtime notebook)
- `meteogram_derivations` → `add_derived`, `expand_to_base_params`,
  `station_timeseries_to_long` (meteogram notebook)

This bootstrap is distinct from the removed style `sys.path` hack: style is now
a proper package import; this only reaches genuinely-shared workflow helpers we
deliberately do not move.

### Tooling dependency

The env currently declares only `marimo`. Authoring and executing Jupyter
notebooks needs `ipykernel` + `nbconvert` (execution / port-fidelity check) and
optionally `jupyterlab` (interactive editing). These go in a new
`[dependency-groups]` group (e.g. `notebooks`), not the core `dependencies`.

### Notebook anatomy

Every notebook follows the same three-block skeleton:

```python
# ── Cell 1: load ──────────────────────────────────────────
from evalml.publication.manifest import load_manifest
from evalml.publication import style
import matplotlib.pyplot as plt

m = load_manifest()                 # auto-finds output/publication/<truth>/manifest.json
                                    #   (or EVALML_MANIFEST / explicit path)
m.validate_request("figures")       # loud, clear error if the manifest is incoherent
plt.style.use(style.mplstyle_path())

# ── Cell 2: resolve data (paths straight from the manifest — no hashes) ──
pairs = m.verif_paths()             # [(path, label), ...]
data  = {label: xr.open_dataset(p) for p, label in pairs}

# ── Cell 3+: plot (the part you tweak) ────────────────────
fig, ax = plt.subplots(...)
for label, ds in data.items():
    ax.plot(..., **style.line_style(label))
fig.savefig("figures/leadtime/rmse_bias.pdf")
```

- Each notebook calls `m.validate_request(<figure>, ...)` up front, preserving
  the "clear error, not a cryptic plot failure" property that previously lived
  in the retired CLI. The resolver's figure keys are `"figures"` (lead-time),
  `"meteogram"`, and `"scoremaps"` — note the lead-time notebook validates via
  the `"figures"` key even though its config block is `leadtimes` and its output
  dir is `leadtime` (a pre-existing naming quirk to carry over, not fix here).
- Each notebook reads its **configured case** from `m.publication[...]` (e.g.
  `meteogram.station`, `scoremaps.params/steps`), so the paper's default figure
  reproduces with no edits; overrides are just a changed variable in a cell.

### Data flow

```
config/*.yaml (publication: block)
   │  validated at load (config.py)
   ▼
Snakemake: inference → verification → results (*.nc)  +  publication_manifest
   │
   ▼
output/publication/<truth>/manifest.json   ← single contract
   │
   ▼
notebook: load_manifest() → Manifest.{verif_paths,scoremap_path,grib_dir,…}
          → open datasets → plot with evalml.publication.style → savefig
```

### Error handling

- **Config-load time:** `ConfigModel` publication validators unchanged — an
  incoherent `publication:` block still fails the launch early.
- **Notebook time:** `m.validate_request(<figure>, ...)` re-runs the same
  coherence checks against the manifest, so a manifest-only consumer still gets
  a clear `ResolutionError` rather than a cryptic plotting failure.

### Testing

- **`tests/unit/test_publication_style.py`** (new/adapted): `param_label`
  fallback, `line_style` selects CH1/CH2/Varda/EPS-mean correctly,
  `mplstyle_path()` returns an existing packaged file. Cheap, no data.
- **`manifest.py` / `resolver.py` tests** — untouched, still green.
- **`test_publication_config.py`** — untouched (config validation unchanged).
- **Notebooks are not unit-tested.** They need real result data; their
  correctness is verified by the manual side-by-side below. This is stated
  explicitly rather than implied.

### Port-fidelity verification (the main risk)

The plotting code moves from marimo cells into notebook cells. Because
`style.py` carries over unchanged and data comes from the same manifest, output
should be near-identical. To verify rather than assume: **before deleting each
marimo script, render its figure once the old way and once from the new
notebook, and compare side by side.** Delete the old script only once its
notebook reproduces it.

### Docs

`docs/publication_figures.md` is rewritten to match:

- The "Standalone CLI (Section B)" and "notebooks (Section C)" sections collapse
  into a single "Open the notebook" story.
- The architecture diagram loses the CLI/scripts boxes and ends at
  `manifest → notebook`.
- The Snakemake section shrinks to "produces results + manifest."

## Summary of change

| | Before | After |
|---|---|---|
| Snakemake publication | manifest + 3 figure rules (duplicate path logic) | **manifest only** (+ result deps) |
| Render path | `python -m evalml.publication` CLI → marimo scripts | **3 Jupyter notebooks**, standalone |
| Data resolution | manifest + resolver | **manifest + resolver (unchanged)** |
| Styling | `workflow/scripts/publication_style.py` | **`evalml.publication.style` (promoted)** |

## Out of scope

- Consolidating the run/baseline/truth hashing + path conventions into one
  module (the manifest is already the first step; a larger follow-up).
- The `eckit`/`eckitlib`/`eccodeslib` native-stack pinning for meteogram global
  grid decoding — a pre-existing environment concern, unchanged here.
- Reducing the number of paper config files — may become possible once plotting
  is decoupled, but not addressed by this change.
```