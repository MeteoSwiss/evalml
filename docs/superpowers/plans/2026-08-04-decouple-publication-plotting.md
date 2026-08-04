# Decouple Publication Plotting from Snakemake — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Snakemake produce results + a manifest and stop; move all publication plotting into three standalone Jupyter notebooks driven only by the manifest.

**Architecture:** Keep the manifest/resolver data layer untouched. Promote the shared style module into the `evalml.publication` package. Rewrite the three marimo figure scripts as standalone `.ipynb` notebooks that load the manifest, resolve paths through `Manifest`, and plot with the shared style. Slim the Snakemake `publication.smk` down to the `publication_manifest` rule (plus the surviving scoremap input-listing helpers that feed `publication_all`), and delete the standalone renderer CLI.

**Tech Stack:** Python ≥3.11, Snakemake (<9.10), pydantic, xarray/pandas/numpy, matplotlib, earthkit-plots + cartopy (scoremaps), Jupyter (`ipykernel`/`nbconvert`), `nbformat` (already present).

## Global Constraints

- **Core untouched:** the inference pipeline, verification metric/score computation, and the hashing identity model in `common.smk` are not modified. `TRUTH_HASH`/`VERIF_HASH`/`run_id` and all on-disk paths stay identical.
- **Manifest is the single source of truth for paths.** Never split a `run_id` (it contains `/`); only `str.format`-join templates. Notebooks resolve every path through `Manifest`, never by hand-assembling hashes.
- **`evalml.resolution` stays import-light** (no Snakemake globals) so the workflow process, the CLI, tests, and notebooks can all import it.
- **Every intermediate commit stays green.** Style promotion is additive first; originals are deleted only after their importers are gone.
- **Notebooks are `.ipynb`**, one per figure type, standalone (never invoke Snakemake; Snakemake never invokes them).
- **Python ≥3.11.** Notebook tooling goes in a `[dependency-groups]` group, never in core `dependencies`.
- **Commits:** concise, self-contained messages. This branch already carries `Co-Authored-By: Francesco Zanetta <62377868+frazane@users.noreply.github.com>` — keep that trailer on each commit. Do not push; do not open PRs.

## File Structure

**Created:**
- `src/evalml/publication/style.py` — shared style (colors, `line_style`, `param_label`, skill colormap artifacts) + new `mplstyle_path()`. Canonical copy of the old `workflow/scripts/publication_style.py`.
- `src/evalml/publication/publication.mplstyle` — packaged copy of the mplstyle.
- `tests/unit/test_publication_style.py` — unit tests for the promoted module.
- `notebooks/publication/leadtime.ipynb`
- `notebooks/publication/meteogram.ipynb`
- `notebooks/publication/scoremaps.ipynb`

**Modified:**
- `pyproject.toml` — add a `notebooks` dependency group.
- `workflow/rules/publication.smk` — keep `publication_manifest` + scoremap input helpers; delete the three figure rules + `_meteogram_data_dep`.
- `workflow/Snakefile` — redefine `publication_all` to depend on manifest + results.
- `workflow/scripts/plot_meteogram_region.py` — repoint style import to the package.
- `docs/publication_figures.md` — rewrite for the manifest→notebook flow.

**Deleted (in the cleanup tasks, after replacements exist):**
- `src/evalml/publication/cli.py`, `src/evalml/publication/__main__.py`
- `workflow/scripts/publication_figures.py`, `publication_meteogram.py`, `publication_scoremaps.py`
- `workflow/scripts/publication_style.py`, `workflow/scripts/publication.mplstyle`

---

## Task 1: Promote the style module into the package

**Files:**
- Create: `src/evalml/publication/style.py`
- Create: `src/evalml/publication/publication.mplstyle`
- Test: `tests/unit/test_publication_style.py`

**Interfaces:**
- Produces: module `evalml.publication.style` exposing (unchanged from the old `publication_style.py`) `OBS_LABEL: str`, `COLOR_OBS/COLOR_CH1/COLOR_CH2/COLOR_VARDA: str`, `COLOR_SKILL_MODEL_BETTER/COLOR_SKILL_BASELINE_BETTER: str`, `SKILL_CMAP: LinearSegmentedColormap`, `SKILL_GREY: str`, `SKILL_LEVELS: list[float]`, `SCORE_LABELS: dict`, `PARAM_LABELS: dict`, `param_label(param: str) -> str`, `line_style(src: str) -> dict`; plus new `mplstyle_path() -> pathlib.Path` returning the packaged `.mplstyle`.

This is additive: the old `workflow/scripts/publication_style.py` and `.mplstyle` stay in place (they still serve the not-yet-deleted marimo scripts and `plot_meteogram_region.py`). Both copies are byte-identical in content except for the new helper.

- [ ] **Step 1: Copy the mplstyle into the package**

Copy `workflow/scripts/publication.mplstyle` to `src/evalml/publication/publication.mplstyle` verbatim (no content change).

- [ ] **Step 2: Create the package style module**

Create `src/evalml/publication/style.py` with the full contents of `workflow/scripts/publication_style.py` (copy verbatim), then append the `mplstyle_path()` helper and its import. The head of the file gains:

```python
from importlib import resources
from pathlib import Path
```

and at the end of the file add:

```python
def mplstyle_path() -> Path:
    """Filesystem path to the packaged publication matplotlib style.

    Apply with ``plt.style.use(mplstyle_path())``. Kept as a function (not a
    module constant) so ``importlib.resources`` resolves it lazily and works
    both from the editable checkout and an installed wheel.
    """
    return Path(resources.files("evalml.publication") / "publication.mplstyle")
```

Update the module docstring's first paragraph to read (replacing the reference to the three marimo scripts):

```python
"""Shared visual style for the publication figures.

Source of truth for colors, markers, line styles, and human-readable parameter
labels used by the publication notebooks (``notebooks/publication/*.ipynb``) and
by ``plot_meteogram_region.py``.  Font sizes and layout defaults live in the
packaged ``publication.mplstyle``; apply it with::

    import matplotlib.pyplot as plt
    from evalml.publication import style
    plt.style.use(style.mplstyle_path())

Tweak the look of the paper figures here.
"""
```

- [ ] **Step 3: Write the failing test**

Create `tests/unit/test_publication_style.py`:

```python
"""Tests for the promoted publication style module."""

from evalml.publication import style


def test_param_label_known_and_fallback():
    assert style.param_label("T_2M") == "2m Temperature"
    # Unknown codes fall back to the code itself.
    assert style.param_label("ZZZ") == "ZZZ"


def test_line_style_source_selection():
    assert style.line_style("ICON-CH1-CTRL")["color"] == style.COLOR_CH1
    assert style.line_style("ICON-CH2-CTRL")["color"] == style.COLOR_CH2
    assert style.line_style("Varda-Single")["color"] == style.COLOR_VARDA
    # EPS mean sources are dashed.
    assert style.line_style("ICON-CH1-EPS mean")["linestyle"] == "--"
    # The observations source is markers-only (no line).
    assert style.line_style(style.OBS_LABEL)["linestyle"] == "none"


def test_mplstyle_path_exists():
    p = style.mplstyle_path()
    assert p.name == "publication.mplstyle"
    assert p.is_file()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_publication_style.py -v`
Expected: PASS (all three tests). If `mplstyle_path()` fails with a resources error, confirm the `.mplstyle` sits directly under `src/evalml/publication/`.

- [ ] **Step 5: Verify the packaged style is discoverable and the old copy still works**

Run: `.venv/bin/python -c "from evalml.publication import style; print(style.mplstyle_path()); import matplotlib.pyplot as plt; plt.style.use(style.mplstyle_path()); print('style applies OK')"`
Expected: prints the packaged path and `style applies OK`.

- [ ] **Step 6: Commit**

```bash
git add src/evalml/publication/style.py src/evalml/publication/publication.mplstyle tests/unit/test_publication_style.py
git commit -m "Promote publication style into evalml.publication.style

Co-Authored-By: Francesco Zanetta <62377868+frazane@users.noreply.github.com>"
```

---

## Task 2: Add the notebooks tooling dependency group

**Files:**
- Modify: `pyproject.toml:43-47` (the `[dependency-groups]` table)

**Interfaces:**
- Produces: a `notebooks` dependency group providing `ipykernel`, `nbconvert`, `jupyterlab` in the environment; `.venv` gains a `python -m nbconvert` entry point and an `evalml` Jupyter kernel.

- [ ] **Step 1: Add the group to pyproject**

Edit the `[dependency-groups]` table in `pyproject.toml` (currently only `dev`) to add:

```toml
[dependency-groups]
dev = [
    "pre-commit>=4.2.0",
    "snakefmt>=2.0",
]
notebooks = [
    "jupyterlab>=4.0",
    "nbconvert>=7.0",
    "ipykernel>=6.0",
]
```

- [ ] **Step 2: Sync the environment**

Run: `uv sync --group notebooks`
Expected: resolves and installs `jupyterlab`, `nbconvert`, `ipykernel` (and deps) into `.venv`; updates `uv.lock`.

- [ ] **Step 3: Verify the tools import and a kernel can run**

Run: `.venv/bin/python -c "import nbconvert, ipykernel, nbformat; print('notebook tooling OK')"`
Expected: `notebook tooling OK`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add notebooks dependency group (jupyterlab, nbconvert, ipykernel)

Co-Authored-By: Francesco Zanetta <62377868+frazane@users.noreply.github.com>"
```

---

## Task 3: Leadtime notebook (`leadtime.ipynb`)

Port `workflow/scripts/publication_figures.py` into a standalone Jupyter notebook. The plotting bodies are copied verbatim from the marimo cells; the manifest-driven load replaces the marimo/argparse plumbing, and the style import changes to the package.

**Files:**
- Create: `notebooks/publication/leadtime.ipynb`
- Reference (source of the ported plotting code): `workflow/scripts/publication_figures.py`
- Reference (helpers, unchanged, stay in workflow/scripts): `workflow/scripts/verification_plot_metrics.py` (`_ensure_unique_lead_time`, `_select_best_sources`, `decode_metric`)

**Interfaces:**
- Consumes: `evalml.publication.manifest.load_manifest`, `evalml.publication.manifest.figures_dir`, `Manifest.verif_paths() -> list[tuple[str,str]]`, `Manifest.validate_request("figures")`, `Manifest.output_root`, `Manifest.truth`; `evalml.publication.style.{line_style, param_label, mplstyle_path}`.
- Produces: `output/figures/<truth>/leadtime/{publication_figures_rmse_bias,publication_figures_ets,publication_figures_rmse_bias_skill,publication_figures_ets_skill}.{pdf,png}` + `publication_figures.html`.

- [ ] **Step 1: Author the notebook with an nbformat builder**

Create a throwaway builder script `build_leadtime_nb.py` at the repo root (deleted in Step 4) that writes `notebooks/publication/leadtime.ipynb` from an ordered list of cell sources. Use this exact scaffold; fill the `CELLS` list with the cell sources given in Step 2:

```python
import nbformat as nbf
from pathlib import Path

CELLS = [
    # (filled in Step 2, in order)
]

nb = nbf.v4.new_notebook()
nb["cells"] = [nbf.v4.new_code_cell(src) for src in CELLS]
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = Path("notebooks/publication/leadtime.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(out))
print("wrote", out)
```

- [ ] **Step 2: Fill CELLS with the leadtime notebook cells**

The `CELLS` list is these sources, in order. **Cell A (bootstrap)** and **Cell B (load)** are new; **Cells C–H** are copied verbatim from `publication_figures.py` with only the two mechanical edits noted.

Cell A — bootstrap (reach the workflow-script helper, apply style):
```python
import sys
from pathlib import Path

# Repo root = two levels up from notebooks/publication/. Notebooks run with cwd
# at the repo root; fall back to the file's location if needed.
PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "workflow").is_dir():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
# verification_plot_metrics lives in workflow/scripts (shared with the main
# workflow, deliberately not moved); plotting/data_input/verification come from
# the editable src/ install. Style is a proper package import (no path hack).
sys.path.insert(0, str(PROJECT_ROOT / "workflow" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib.pyplot as plt
from evalml.publication import style
plt.style.use(style.mplstyle_path())
```

Cell B — load from the manifest (replaces the marimo/argparse cells of the original):
```python
import xarray as xr
from evalml.publication.manifest import load_manifest, figures_dir
from verification_plot_metrics import (
    _ensure_unique_lead_time as ensure_unique_lead_time,
    _select_best_sources as select_best_sources,
    decode_metric,
)

# Auto-discovers output/publication/<truth>/manifest.json (or $EVALML_MANIFEST,
# or set truth=... below when several truths exist).
m = load_manifest()
m.validate_request("figures")

pairs = m.verif_paths()                       # [(path, label), ...]
sources = [label for _, label in pairs]
output_dir = str(figures_dir(m.output_root, m.truth["label"]) / "leadtime")

def _abs(f):
    p = Path(f)
    return p if p.is_absolute() else PROJECT_ROOT / p

_dfs = [xr.open_dataset(_abs(p)) for p, _ in pairs]
_dfs = [ensure_unique_lead_time(d) for d in _dfs]
_dfs = select_best_sources(_dfs)
ds = xr.concat(_dfs, dim="source", join="outer")
```

Cell C — `ds_to_df` + `df`: copy the body of the `ds_to_df` cell from `publication_figures.py:107-129` **verbatim**, but delete the line `from verification_plot_metrics import decode_metric as _decode_metric` (Cell B already imports `decode_metric`) and rename `_decode_metric` → `decode_metric` in the one call inside `ds_to_df`. Keep `df = ds_to_df(ds)`.

Cell D — `df_all`: copy `publication_figures.py:133-137` verbatim:
```python
df_all = df[
    (df["region"] == "all") & (df["season"] == "all") & (df["init_hour"] == "all")
].copy()
```

Cell E — skill computation: copy the `compute_skill_ds` cell body from `publication_figures.py:142-177` **verbatim** (it defines `compute_skill_ds`, `skill_sources`, `ds_skill`, `df_skill_all`). No edits.

Cell F — `plot_panels`: copy the `plot_panels` cell from `publication_figures.py:181-270` **verbatim**, with two edits:
- replace `from publication_style import line_style as _line_style` with `from evalml.publication.style import line_style as _line_style`
- delete the line `_plt.style.use(Path(__file__).resolve().parent / "publication.mplstyle")` (style is already applied in Cell A; `__file__` is undefined in a notebook).
Keep everything else (the `import matplotlib.pyplot as _plt`, `matplotlib.ticker`, `numpy` imports, `_XSCALE_KW`, `_XTICKS`, and the full `plot_panels` function).

Cell G — RMSE/BIAS + ETS + skill figures: copy the four figure-drawing cells from `publication_figures.py:273-546` **verbatim into one cell**, with these edits applied to each:
- replace every `from publication_style import param_label as _param_label` with `from evalml.publication.style import param_label as _param_label`
- remove the trailing `mo.image(...)` line from each (no marimo in a notebook); optionally end the cell with `plt.show()` for inline preview.
- the `output_dir`, `df_all`, `df_skill_all`, `plot_panels`, `sources`, `skill_sources` names all resolve from earlier cells.
Keep the `Path(output_dir).mkdir(...)`, the `savefig` calls, and the final `(_out / "publication_figures.html").write_text(...)` block that writes the HTML index.

- [ ] **Step 3: Build and execute the notebook (smoke + fidelity)**

Run the builder, then execute headless:
```bash
.venv/bin/python build_leadtime_nb.py
EVALML_MANIFEST=$(ls output/publication/*/manifest.json | head -1) \
  .venv/bin/python -m nbconvert --to notebook --execute --inplace \
  notebooks/publication/leadtime.ipynb
```
Expected: notebook executes without error and writes the four PDFs/PNGs + `publication_figures.html` under `output/figures/<truth>/leadtime/`.

If no results exist yet under `output/`, this is a fidelity check you cannot run here — record that and defer to a run where `evalml publication <config>` has produced results. Do not delete the old script until the notebook has reproduced its figures at least once.

- [ ] **Step 4: Fidelity compare, then remove the builder**

If results exist: render the old way for comparison and eyeball side-by-side:
```bash
EVALML_MANIFEST=$(ls output/publication/*/manifest.json | head -1) \
  .venv/bin/python -m marimo run workflow/scripts/publication_figures.py \
  -- --output /tmp/leadtime_old || true
```
Compare `/tmp/leadtime_old/*.png` against `output/figures/<truth>/leadtime/*.png`. They should match. Then remove the throwaway builder:
```bash
rm build_leadtime_nb.py
```

- [ ] **Step 5: Commit**

```bash
git add notebooks/publication/leadtime.ipynb
git commit -m "Add standalone leadtime publication notebook

Co-Authored-By: Francesco Zanetta <62377868+frazane@users.noreply.github.com>"
```

---

## Task 4: Meteogram notebook (`meteogram.ipynb`)

Port `workflow/scripts/publication_meteogram.py`. This notebook is heavy: it fetches station observations live from jretrieve and decodes GRIB, so headless execution needs credentials + data and the matched eckit/eccodes stack. Author it faithfully; execution/fidelity is a manual step where the environment allows.

**Files:**
- Create: `notebooks/publication/meteogram.ipynb`
- Reference (ported plotting): `workflow/scripts/publication_meteogram.py`
- Reference (helpers): `data_input` (src), `meteogram_derivations` (workflow/scripts), `verification.spatial` (src)

**Interfaces:**
- Consumes: `load_manifest`, `figures_dir`, `Manifest.publication` (`["meteogram"]` block), `Manifest.get_candidate()`, `Manifest.grib_dir(participant, init_time)`, `Manifest.meteogram_baseline_specs() -> str`, `Manifest.validate_request("meteogram", init_time=...)`; `evalml.publication.style.{OBS_LABEL, line_style, param_label, mplstyle_path}`; helpers `data_input.{load_forecast_data, load_obs_data_from_jretrieve, parse_steps}`, `meteogram_derivations.{add_derived, expand_to_base_params, station_timeseries_to_long}`, `verification.spatial.map_forecast_to_truth`.
- Produces: `output/figures/<truth>/meteogram/publication_meteogram.{pdf,png}` + `publication_meteogram.html`.

- [ ] **Step 1: Author with an nbformat builder**

Create throwaway `build_meteogram_nb.py` at repo root using the same scaffold as Task 3 Step 1 (change the output path to `notebooks/publication/meteogram.ipynb`), with the `CELLS` from Step 2.

- [ ] **Step 2: Fill CELLS with the meteogram notebook cells**

Cell A — bootstrap + logging (new):
```python
import sys, time, logging
from pathlib import Path

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "workflow").is_dir():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "workflow" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LOG = logging.getLogger("meteogram")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

import matplotlib.pyplot as plt
from evalml.publication import style
plt.style.use(style.mplstyle_path())
```

Cell B — resolve the configured case from the manifest (replaces the marimo/argparse defaults cell `publication_meteogram.py:33-127`):
```python
from evalml.publication.manifest import load_manifest, figures_dir

m = load_manifest()
_mg = m.publication.get("meteogram") or {}

# Configured case; override any of these variables in-cell to retarget.
date = _mg.get("init_time", "202504010000")
station = _mg.get("station", "KLO")
display_params = _mg.get("params", ["T_2M", "TOT_PREC", "SP_10M", "DD_10M"])

m.validate_request("meteogram", init_time=date)

_cand = m.get_candidate()
forecast = m.grib_dir(_cand, date)
forecast_steps = _cand.steps
forecast_label = _cand.label
output_dir = str(figures_dir(m.output_root, m.truth["label"]) / "meteogram")

# Parse the structured baseline spec "root|steps|member|label;..." from the manifest.
def _parse_baselines(raw):
    out = []
    for spec in [s for s in raw.split(";") if s.strip()]:
        root, steps, member, label = spec.split("|")
        out.append({"root": root, "steps": steps, "member": member, "label": label})
    return out

baselines = _parse_baselines(m.meteogram_baseline_specs())
```

Cell C — load obs + forecasts into the long dataframe: copy the data-loading cell body from `publication_meteogram.py:144-221` **verbatim**, with these edits:
- replace `from publication_style import OBS_LABEL` with `from evalml.publication.style import OBS_LABEL`
- delete the marimo-injected `LOG`, `Path`, `time`, `project_root` cell-arg dependencies — `LOG`, `time` (via `import time`), and `Path` are provided by Cell A; replace `project_root` with `PROJECT_ROOT` in the `_abs` helper.
Keep the imports of `datetime`, `data_input.*`, `meteogram_derivations.*`, `verification.spatial.map_forecast_to_truth`, `pandas as pd`, and the full loading logic producing `df`, `source_order`, `init_time`, `OBS_LABEL`.

Cell D — render the meteogram figure: copy the plotting cell body from `publication_meteogram.py:238-328` **verbatim**, with these edits:
- replace `from publication_style import line_style, param_label` with `from evalml.publication.style import line_style, param_label`
- delete `plt.style.use(Path(__file__).resolve().parent / "publication.mplstyle")` (applied in Cell A; `__file__` undefined)
- remove the trailing `mo.image(...)` line; optionally end with `plt.show()`.
Keep the `_UNITS` map, the per-param subplot loop, the DD_10M circular handling, the legend/suptitle/`tight_layout`, the `savefig` calls, and the `publication_meteogram.html` write.

- [ ] **Step 3: Build the notebook**

Run: `.venv/bin/python build_meteogram_nb.py`
Expected: writes `notebooks/publication/meteogram.ipynb`.

- [ ] **Step 4: Execute if the environment allows; otherwise defer**

The meteogram needs jretrieve credentials (`JRETRIEVE_CLIENT_ID`/`_SECRET` in a repo-root `.env`), the candidate GRIB for `date` on disk, and the matched eckit/eccodes native stack. If all are present:
```bash
EVALML_MANIFEST=$(ls output/publication/*/manifest.json | head -1) \
  .venv/bin/python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 notebooks/publication/meteogram.ipynb
```
Expected: writes `publication_meteogram.{pdf,png}` + `.html` under `output/figures/<truth>/meteogram/`. If credentials/data/eckit are unavailable here, record that execution is deferred (this is the documented not-unit-tested figure) and do not delete the old script until it has been reproduced once.

- [ ] **Step 5: Remove the builder and commit**

```bash
rm build_meteogram_nb.py
git add notebooks/publication/meteogram.ipynb
git commit -m "Add standalone meteogram publication notebook

Co-Authored-By: Francesco Zanetta <62377868+frazane@users.noreply.github.com>"
```

---

## Task 5: Scoremaps notebook (`scoremaps.ipynb`)

Port `workflow/scripts/publication_scoremaps.py` (a plain script, not marimo). Its module-level helpers become notebook cells; the `main()` argument parsing is replaced by manifest resolution.

**Files:**
- Create: `notebooks/publication/scoremaps.ipynb`
- Reference (ported logic): `workflow/scripts/publication_scoremaps.py`
- Reference (helpers): `plotting` (src: `DOMAINS`, `StatePlotter`)

**Interfaces:**
- Consumes: `load_manifest`, `figures_dir`, `Manifest.publication` (`["scoremaps"]` block), `Manifest.get_candidate()`, `Manifest.resolve_baseline(label)`, `Manifest.scoremap_path(participant, param, leadtime) -> str`, `Manifest.validate_request("scoremaps", baseline=..., leadtime=...)`; `evalml.publication.style.{COLOR_SKILL_BASELINE_BETTER, COLOR_SKILL_MODEL_BETTER, PARAM_LABELS, SCORE_LABELS, SKILL_CMAP, SKILL_GREY, SKILL_LEVELS, mplstyle_path}`; `plotting.{DOMAINS, StatePlotter}`; `earthkit.plots`, `cartopy`.
- Produces: `output/figures/<truth>/scoremaps/publication_scoremaps_<lt>h.{pdf,png}` (one per lead time) + `publication_scoremaps.html`.

- [ ] **Step 1: Author with an nbformat builder**

Create throwaway `build_scoremaps_nb.py` (same scaffold, output `notebooks/publication/scoremaps.ipynb`).

- [ ] **Step 2: Fill CELLS with the scoremaps notebook cells**

Cell A — bootstrap + earthkit schema setup: copy the module-level setup of `publication_scoremaps.py:16-65` **verbatim** with edits:
- keep the imports (`logging`, `earthkit.plots as ekp`, `matplotlib.colors as mcolors`, `numpy as np`, `xarray as xr`, `cartopy...Gridliner`, `matplotlib.pyplot as plt`, `to_hex`)
- replace the `sys.path` block (`publication_scoremaps.py:29-31`) with the notebook bootstrap:
  ```python
  import sys
  from pathlib import Path
  PROJECT_ROOT = Path.cwd()
  if not (PROJECT_ROOT / "workflow").is_dir():
      PROJECT_ROOT = Path(__file__).resolve().parents[2]
  sys.path.insert(0, str(PROJECT_ROOT / "src"))
  ```
- replace `plt.style.use(_script_dir / "publication.mplstyle")` with:
  ```python
  from evalml.publication import style as _style
  plt.style.use(_style.mplstyle_path())
  ```
- replace the `from publication_style import (...)` block with `from evalml.publication.style import (COLOR_SKILL_BASELINE_BETTER, COLOR_SKILL_MODEL_BETTER, PARAM_LABELS, SCORE_LABELS, SKILL_CMAP, SKILL_GREY, SKILL_LEVELS)`
- keep `from plotting import DOMAINS, StatePlotter`
- keep the `LOG` setup, the `ekp.schema.*` border/coastline overrides, `_PUB_EXTENTS`, and `_SENTINEL`.

Cell B — pure helper functions: copy `_build_skill_artifacts`, `_load_raw`, `_compute_panel`, `_remove_latlon_labels`, `_make_figure` from `publication_scoremaps.py:68-287` **verbatim** (no edits — they reference the names imported in Cell A).

Cell C — resolve the configured case from the manifest (replaces `main()`'s argparse + resolution, `publication_scoremaps.py:290-412`):
```python
from evalml.publication.manifest import load_manifest, figures_dir

m = load_manifest()
_sm = m.publication.get("scoremaps") or {}

# Configured case; override in-cell to retarget.
params = _sm.get("params", ["T_2M", "SP_10M"])
scores = _sm.get("scores", ["MSE_SKILL", "BIAS_CONTRIB"])
leadtimes = [int(s) for s in (_sm.get("steps") or [24])]
baseline_label = _sm.get("baseline_label", "ICON-CH1-CTRL")
region = _sm.get("region", "switzerland")
season = _sm.get("season", "all")
candidate_label = m.get_candidate().label
output = figures_dir(m.output_root, m.truth["label"]) / "scoremaps"

cand = m.get_candidate()
base = m.resolve_baseline(baseline_label)
for _lt in leadtimes:
    m.validate_request("scoremaps", baseline=baseline_label, leadtime=_lt)

# Leadtime-major ordering (all params for leadtimes[0], then leadtimes[1], ...).
candidate_files = [Path(m.scoremap_path(cand, p, lt)) for lt in leadtimes for p in params]
baseline_files = [Path(m.scoremap_path(base, p, lt)) for lt in leadtimes for p in params]
n_params = len(params)
```

Cell D — build the plotter and render one figure per lead time: copy `publication_scoremaps.py:393-449` **verbatim** with edits:
- drop the `parser.error(...)` count-mismatch guards (`publication_scoremaps.py:382-391`) — the manifest resolution guarantees the counts; keep an `assert len(candidate_files) == n_params * len(leadtimes)` in their place.
- replace `args.output` with `output`, `args.region` with `region`, `args.season` with `season`, `args.candidate_label` with `candidate_label`, `args.baseline_label` with `baseline_label`, `args.leadtimes` with `leadtimes`.
Keep: reading `lons`/`lats` from `candidate_files[0]`, `output.mkdir(...)`, `plotter = StatePlotter(lons, lats, output)`, `domain` selection with `_PUB_EXTENTS`, `style, skill_cmap, skill_norm = _build_skill_artifacts()`, the per-leadtime `_make_figure(...)` loop with `fig.save(...)` to `publication_scoremaps_<lt>h.{pdf,png}`, and the `publication_scoremaps.html` index write. Optionally end with `plt.show()`.

- [ ] **Step 3: Build the notebook**

Run: `.venv/bin/python build_scoremaps_nb.py`
Expected: writes `notebooks/publication/scoremaps.ipynb`.

- [ ] **Step 4: Execute against a gridded (zarr) manifest, else defer**

Scoremaps require a gridded truth and the scoremap NC files on disk. Pick a zarr-truth manifest (e.g. KENDA-CH1). If available:
```bash
EVALML_MANIFEST=output/publication/KENDA-CH1/manifest.json \
  .venv/bin/python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 notebooks/publication/scoremaps.ipynb
```
Expected: writes `publication_scoremaps_<lt>h.{pdf,png}` + `.html` under `output/figures/<truth>/scoremaps/`. If the gridded data/earthkit stack is unavailable here, record execution as deferred; do not delete the old script until reproduced once.

- [ ] **Step 5: Remove the builder and commit**

```bash
rm build_scoremaps_nb.py
git add notebooks/publication/scoremaps.ipynb
git commit -m "Add standalone scoremaps publication notebook

Co-Authored-By: Francesco Zanetta <62377868+frazane@users.noreply.github.com>"
```

---

## Task 6: Slim the Snakemake publication rules and delete the renderer CLI

Reduce `publication.smk` to the manifest rule plus the surviving scoremap input helpers; redefine `publication_all`; delete the marimo scripts and the standalone CLI.

**Files:**
- Modify: `workflow/rules/publication.smk`
- Modify: `workflow/Snakefile:274-297` (`rule publication_all`)
- Delete: `src/evalml/publication/cli.py`, `src/evalml/publication/__main__.py`
- Delete: `workflow/scripts/publication_figures.py`, `publication_meteogram.py`, `publication_scoremaps.py`

**Interfaces:**
- Consumes: existing `common.smk` globals — `RUN_CONFIGS`, `BASELINE_CONFIGS`, `EXPERIMENT_PARTICIPANTS`, `TRUTH_HASH`, `OUT_ROOT`, `resolve_baseline_id`, and `config`; `evalml.publication.manifest.{truth_slug, build_manifest, write_manifest}`.
- Produces: `rule publication_manifest` (unchanged output `OUT_ROOT/publication/<slug>/manifest.json`) and `rule publication_all` now depending on the manifest + `EXPERIMENT_PARTICIPANTS.values()` + (scoremap NC files when `publication.scoremaps.enabled`).

- [ ] **Step 1: Rewrite `publication.smk` to keep only the manifest rule + scoremap input helpers**

Replace the entire contents of `workflow/rules/publication.smk` with the following (this keeps `publication_manifest` verbatim, keeps the scoremap input-listing helpers, and drops the three figure rules and `_meteogram_data_dep`):

```python
# ----------------------------------------------------- #
# Publication-grade figures workflow                    #
# ----------------------------------------------------- #
# Snakemake produces the *results* and a *manifest* (run/baseline -> hash ->
# data-path mapping) and stops. All publication-ready plotting happens in the
# standalone notebooks under `notebooks/publication/`, driven only by the
# manifest — see docs/publication_figures.md.

from evalml.publication.manifest import truth_slug

# Publication outputs are namespaced by the truth label so station-based and
# analysis-based runs don't overwrite each other's manifest.
TRUTH_SLUG = truth_slug((config.get("truth") or {}).get("label", ""))


rule publication_manifest:
    """Persist the run/baseline -> hash -> data-path mapping for the notebooks.

Cheap localrule: dumps the in-memory workflow globals to JSON, so paths can be
resolved (interactively or by the notebooks) without recomputing any hash.
The master hash (a digest of the whole config) is a rule param, so Snakemake's
`params` rerun-trigger regenerates the manifest whenever the config content
changes — without re-running on a no-op file touch.
"""
    output:
        OUT_ROOT / f"publication/{TRUTH_SLUG}/manifest.json",
    input:
        script="src/evalml/publication/manifest.py",
    localrule: True
    params:
        master_hash=master_hash(),
    run:
        from evalml.publication.manifest import build_manifest, write_manifest

        manifest = build_manifest(
            run_configs=RUN_CONFIGS,
            baseline_configs=BASELINE_CONFIGS,
            truth_cfg=config.get("truth"),
            truth_hash=TRUTH_HASH,
            verif_hash=VERIF_HASH,
            reftimes=REFTIMES,
            output_root=str(OUT_ROOT),
            publication_cfg=config.get("publication", {}),
            master_hash=params.master_hash,
        )
        write_manifest(output[0], manifest)


# --- Result dependencies for `publication_all` -------------------------------
# The scoremap NC files must be listed from in-memory globals (the manifest file
# a sibling rule produces cannot be read at DAG-build time). In the paper configs
# `experiment.scoremaps.enabled` is false while `publication.scoremaps.enabled`
# is true, so the publication target is the only thing pulling scoremap
# production — these helpers keep that dependency alive.


def _pub_candidate_run_id():
    """The single publication candidate run_id (raises if not exactly one)."""
    candidates = [rid for rid, cfg in RUN_CONFIGS.items() if cfg.get("_is_candidate")]
    if len(candidates) != 1:
        raise ValueError(
            f"The publication workflow expects exactly one candidate run; "
            f"found {len(candidates)}. Pick a single candidate in the config."
        )
    return candidates[0]


def _pub_scoremap_cfg():
    return (config.get("publication", {}) or {}).get("scoremaps") or {}


def _pub_scoremap_leadtimes(cfg):
    """Lead times (hours) to plot: publication.scoremaps.steps when set,
    otherwise falls back to experiment.scoremaps.leadtimes.
    """
    steps = cfg.get("steps")
    if steps is not None:
        return [int(s) for s in steps]
    return list(
        config.get("experiment", {}).get("scoremaps", {}).get("leadtimes", [6, 24])
    )


def _pub_scoremap_files():
    """Scoremap NC files (candidate + baseline) required by publication_all.

    Ordered leadtime-major (all params for leadtimes[0], then leadtimes[1], …),
    using the same path template the manifest/notebook resolves, so the declared
    dependency always matches what the scoremaps notebook reads.
    """
    cfg = _pub_scoremap_cfg()
    params = cfg.get("params", ["T_2M", "SP_10M"])
    leadtimes = _pub_scoremap_leadtimes(cfg)
    cand_id = _pub_candidate_run_id()
    base_id = resolve_baseline_id(cfg.get("baseline_label", "ICON-CH1-CTRL"))
    cand = [
        str(OUT_ROOT / f"data/runs/{cand_id}/scoremaps/{p}_{lt}_{TRUTH_HASH}.nc")
        for lt in leadtimes
        for p in params
    ]
    base = [
        str(OUT_ROOT / f"data/baselines/{base_id}/scoremaps/{p}_{lt}_{TRUTH_HASH}.nc")
        for lt in leadtimes
        for p in params
    ]
    return cand + base
```

- [ ] **Step 2: Redefine `publication_all` in `workflow/Snakefile`**

Replace `rule publication_all:` (`workflow/Snakefile:274-297`) with:

```python
rule publication_all:
    """Target: produce all results + the manifest the notebooks plot from.

    Stops before plotting. Figures are rendered by the standalone notebooks in
    notebooks/publication/, driven only by the manifest.
    """
    input:
        [rules.publication_manifest.output[0]]
        + list(EXPERIMENT_PARTICIPANTS.values())
        + (
            _pub_scoremap_files()
            if ((config.get("publication") or {}).get("scoremaps") or {}).get(
                "enabled", False
            )
            else []
        ),
```

- [ ] **Step 3: Delete the marimo figure scripts and the renderer CLI**

```bash
git rm workflow/scripts/publication_figures.py \
       workflow/scripts/publication_meteogram.py \
       workflow/scripts/publication_scoremaps.py \
       src/evalml/publication/cli.py \
       src/evalml/publication/__main__.py
```

- [ ] **Step 4: Verify the DAG builds and stops at manifest + results**

Run a dry-run of the publication target against a paper config (choose one that exists, e.g. the scoremaps forecaster config):
```bash
.venv/bin/evalml publication config/varda-single_paper_forecaster_scoremaps.yaml --dry-run
```
Expected: the DAG lists `publication_manifest`, verification/aggregation rules, and `verification_scoremaps*` — and **no** `publication_figures/meteogram/scoremaps` rules. No `MissingRuleException`/`NameError` from the slimmed `publication.smk`.

- [ ] **Step 5: Verify the renderer CLI is gone**

Run: `.venv/bin/python -c "import evalml.publication.cli" 2>&1 | tail -1`
Expected: `ModuleNotFoundError`. Also confirm nothing else imports it:
`grep -rn "evalml.publication.cli\|python -m evalml.publication" --include=*.py --include=*.smk workflow src` → no matches.

- [ ] **Step 6: Run the unit tests (nothing regressed)**

Run: `.venv/bin/python -m pytest tests/unit/test_publication_manifest.py tests/unit/test_publication_config.py tests/unit/test_publication_style.py tests/unit/test_resolution.py -v`
Expected: all PASS (the manifest/config/resolution suites are unaffected by the rule slimming).

- [ ] **Step 7: Commit**

```bash
git add workflow/rules/publication.smk workflow/Snakefile
git commit -m "Snakemake publication target: manifest + results only, drop figure rules and renderer CLI

Co-Authored-By: Francesco Zanetta <62377868+frazane@users.noreply.github.com>"
```

---

## Task 7: Migrate the remaining style consumer and delete the originals

`workflow/scripts/plot_meteogram_region.py` (a main-workflow paper script, kept) still imports the old `publication_style` and applies the old `.mplstyle`. Repoint it to the package, then delete the two `workflow/scripts/` originals.

**Files:**
- Modify: `workflow/scripts/plot_meteogram_region.py:25,101`
- Delete: `workflow/scripts/publication_style.py`, `workflow/scripts/publication.mplstyle`

**Interfaces:**
- Consumes: `evalml.publication.style.{line_style, param_label, mplstyle_path}`.

- [ ] **Step 1: Repoint the style import**

In `workflow/scripts/plot_meteogram_region.py`, replace line 25:
```python
from publication_style import line_style, param_label
```
with:
```python
from evalml.publication.style import line_style, param_label, mplstyle_path
```
(Remove the now-stale sibling-module comment on the line above if present.)

- [ ] **Step 2: Repoint the mplstyle application**

In the same file, replace line 101:
```python
    plt.style.use(Path(__file__).resolve().parent / "publication.mplstyle")
```
with:
```python
    plt.style.use(mplstyle_path())
```

- [ ] **Step 3: Confirm no remaining references to the originals**

Run:
```bash
grep -rn "publication_style\|publication.mplstyle" --include=*.py --include=*.smk workflow src
```
Expected: no matches (the three publication scripts and the three figure rules are already gone; `plot_meteogram_region.py` now uses the package). If any match remains, fix it before deleting.

- [ ] **Step 4: Delete the originals**

```bash
git rm workflow/scripts/publication_style.py workflow/scripts/publication.mplstyle
```

- [ ] **Step 5: Verify `plot_meteogram_region.py` still imports cleanly**

Run: `.venv/bin/python -c "import ast; ast.parse(open('workflow/scripts/plot_meteogram_region.py').read()); from evalml.publication.style import line_style, param_label, mplstyle_path; print('imports OK')"`
Expected: `imports OK`. (A full run needs data/credentials; the import + style resolution is the check here.)

- [ ] **Step 6: Commit**

```bash
git add workflow/scripts/plot_meteogram_region.py
git commit -m "Repoint plot_meteogram_region to packaged style; remove workflow/scripts style originals

Co-Authored-By: Francesco Zanetta <62377868+frazane@users.noreply.github.com>"
```

---

## Task 8: Rewrite the documentation

Update `docs/publication_figures.md` so it describes the manifest→notebook flow: Snakemake produces results + manifest; notebooks render.

**Files:**
- Modify: `docs/publication_figures.md`

- [ ] **Step 1: Rewrite the runbook sections**

Edit `docs/publication_figures.md`:
- **TL;DR:** replace the `python -m evalml.publication ...` block with:
  ```bash
  # 1. Produce results + manifest (reproducible), via Snakemake
  evalml publication config/varda-single_paper_forecaster_scoremaps.yaml

  # 2. Render the paper figures (standalone, no Snakemake)
  jupyter lab notebooks/publication/       # open leadtime / meteogram / scoremaps
  # or headless:
  EVALML_MANIFEST=output/publication/<truth>/manifest.json \
    jupyter nbconvert --to notebook --execute --inplace notebooks/publication/leadtime.ipynb
  ```
- Collapse the old "Standalone CLI (Section B)" and "Interactive notebooks (Section C)" into a single **"Rendering the figures (notebooks)"** section describing: manifest discovery (`load_manifest()` auto-find, `$EVALML_MANIFEST`, or `truth=`), the three notebooks, that the configured case comes from the `publication:` block and is overridable in-cell, and where figures are written (`output/figures/<truth>/<figure>/`).
- **Architecture diagram:** update the mermaid flow to end at `manifest → notebook`; remove the `cli.py` / `python -m evalml.publication` / marimo-scripts boxes. The `pub` subgraph keeps only `manifest.py` and `resolver.py`.
- **"How to run → A" (Snakemake):** change "then renders the figures via thin wrapper rules" to "and stops; figures are rendered from the manifest by the notebooks." Update the second mermaid so `publication_all` depends on `publication_manifest` + the `verif_aggregated_*` / scoremap results (no `F1/F2/F3` figure rules).
- **"Where figures are stored":** keep the `output/figures/<truth>/<figure>/` table but drop the CLI-specific rows; note the notebooks write there by default via `figures_dir(...)`.
- **Troubleshooting:** drop the `--allowed-rules` / `python -m evalml.publication` items; keep the jretrieve-credentials and eckit/eccodes items (still relevant to the meteogram notebook). Add one line: "To re-render without any Snakemake rerun, just run the notebook — it never triggers the inference/verification cascade."
- **"For developers":** update the file tree to drop `cli.py`/`__main__.py` and the `workflow/scripts/publication_*.py` marimo entries; add `src/evalml/publication/style.py` (+ `publication.mplstyle`) and `notebooks/publication/{leadtime,meteogram,scoremaps}.ipynb`. Add `tests/unit/test_publication_style.py` to the test list.

- [ ] **Step 2: Verify no stale references remain in the doc**

Run:
```bash
grep -nE "python -m evalml.publication|thin wrapper|publication_figures.py|publication_meteogram.py|publication_scoremaps.py|marimo" docs/publication_figures.md
```
Expected: no matches (all replaced). Fix any that remain.

- [ ] **Step 3: Commit**

```bash
git add docs/publication_figures.md
git commit -m "Docs: describe manifest -> notebook publication flow

Co-Authored-By: Francesco Zanetta <62377868+frazane@users.noreply.github.com>"
```

---

## Final verification (after all tasks)

- [ ] Run the unit suite: `.venv/bin/python -m pytest tests/unit -q` → all pass (respecting `-m 'not longtest'`).
- [ ] `evalml publication <paper-config> --dry-run` builds a DAG ending at manifest + results, with no figure rules.
- [ ] `notebooks/publication/` contains the three `.ipynb` files; at least the leadtime notebook has been executed against real results and its figures match the pre-refactor output.
- [ ] `grep -rn "publication_style\|python -m evalml.publication" workflow src` → no matches.
- [ ] The three throwaway `build_*_nb.py` files are gone (not committed).
