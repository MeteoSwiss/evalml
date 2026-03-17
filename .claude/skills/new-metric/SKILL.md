---
name: new-metric
description: Add a new verification metric (continuous or categorical) to the evalml pipeline. Use when the user wants to add metrics such as ETS, Hanssen-Kuipers Discriminant, POD, FAR, CSI, or any custom formula.
user-invocable: true
argument-hint: "[metric-name]"
---

# Add a New Verification Metric

Your task is to implement a new verification metric end-to-end in the evalml pipeline.

## Step 1 – Gather information

Ask the user for each piece of information **one at a time**, in order, using the `AskUserQuestion`
tool where possible. Do not ask for multiple things at once. Infer from `$ARGUMENTS` where obvious
(e.g. "ETS" implies a categorical metric).

1. **Metric name** – the short uppercase key used in variable names, e.g. `ETS`, `HKD`, `POD`.

2. **Metric type** – is it a **continuous** metric (operates directly on raw forecast and
   observation values) or a **categorical** metric (requires thresholding both fields to produce
   binary events before scoring)?
   Offer these options via `AskUserQuestion`:
   - Continuous (e.g. MAE, RMSE, CORR — formula on raw values)
   - Categorical (e.g. ETS, HKD, POD — requires a threshold)

3. **Formula** – the mathematical definition.
   For well-known metrics (ETS, HKD, POD, FAR, POFD, CSI, FBI, ACC) do **not** ask — use the
   standard formulas from Step 2 below. For unknown metrics, ask the user to describe the formula.

4. **For categorical metrics only — threshold(s)**:
   Ask for a mapping of parameter names to threshold values, e.g. `TOT_PREC:1.0,T_2M:273.15`.
   Explain that units must match the dataset parameter units.
   Then ask whether thresholds should be:
   - Hard-coded in `verify()` (simpler, recommended for most cases)
   - Exposed as a `--thresholds` CLI argument in `verif_single_init.py` (more flexible)

5. **Post-aggregation transform** – does the metric need a rename or sqrt-transform after
   time-averaging (like MSE→RMSE)? For categorical metrics the answer is almost always no.
   Offer: Yes / No (default No).

After each answer, confirm what was captured before moving to the next question.

---

## Step 2 – Read context files

Before writing any code, read:

- `src/verification/__init__.py` — full file (understand `_compute_scores`, `verify`, and whether
  `_compute_categorical_scores` already exists)
- `workflow/scripts/verif_aggregation.py` — only if a post-aggregation transform is needed
- `workflow/scripts/verif_single_init.py` — only if CLI-configurable thresholds are needed
- `tests/unit/test_verification.py` — if it exists, to match the existing test style;
  otherwise look at `tests/unit/test_spatial_mapping.py` for the project's test conventions

---

## Step 3 – Implement the metric

### For a continuous metric

Add one entry to the dict inside `_compute_scores()` in `src/verification/__init__.py`:

```python
f"{prefix}METRIC_NAME{suffix}": <xarray expression>,
```

In scope: `error = fcst - obs`, `dim` (list of spatial dims to reduce over), `fcst`, `obs`.
Always use `skipna=True` on reductions.

**Standard continuous formulas**

| Metric | Formula |
|--------|---------|
| BIAS   | `error.mean(dim=dim, skipna=True)` |
| MAE    | `abs(error).mean(dim=dim, skipna=True)` |
| MSE    | `(error**2).mean(dim=dim, skipna=True)` |
| CORR   | `xr.corr(fcst, obs, dim=dim)` |
| R2     | `xr.corr(fcst, obs, dim=dim) ** 2` |

### For a categorical metric

**3a.** If `_compute_categorical_scores` does not yet exist, add it to
`src/verification/__init__.py` immediately after `_compute_scores()`:

```python
def _compute_categorical_scores(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    threshold: float,
    dim: list[str],
    prefix="",
    suffix="",
    source="",
) -> xr.Dataset:
    """
    Compute categorical verification metrics for a given threshold.
    Binary events: 1 where value >= threshold, 0 otherwise.
    Returns an xr.Dataset with the computed scores.
    """
    fcst_bin = (fcst >= threshold).astype(float)
    obs_bin = (obs >= threshold).astype(float)

    hits       = (fcst_bin * obs_bin).sum(dim=dim, skipna=True)              # H
    false_alarms = (fcst_bin * (1 - obs_bin)).sum(dim=dim, skipna=True)      # FA
    misses     = ((1 - fcst_bin) * obs_bin).sum(dim=dim, skipna=True)        # M
    correct_neg = ((1 - fcst_bin) * (1 - obs_bin)).sum(dim=dim, skipna=True) # CN
    total = hits + false_alarms + misses + correct_neg                        # N

    scores = xr.Dataset(
        {
            # entries added per metric, see formulas below
        }
    )
    scores = scores.expand_dims({"source": [source]})
    return scores
```

**3b.** Add the new metric entry/entries inside the `scores` dict using the formulas below.
Guard every division with `xr.where(denominator != 0, numerator / denominator, np.nan)`.

**Standard categorical formulas** (H, FA, M, CN, N as defined above)

| Metric | Name | Formula |
|--------|------|---------|
| ETS  | Equitable Threat Score | `H_r = (H+FA)*(H+M)/N` → `(H - H_r) / (H + FA + M - H_r)` |
| HKD  | Hanssen-Kuipers Discriminant | `H/(H+M) - FA/(FA+CN)` |
| POD  | Probability of Detection | `H / (H+M)` |
| FAR  | False Alarm Ratio | `FA / (H+FA)` |
| POFD | Probability of False Detection | `FA / (FA+CN)` |
| CSI  | Critical Success Index | `H / (H+FA+M)` |
| FBI  | Frequency Bias Index | `(H+FA) / (H+M)` |
| ACC  | Accuracy | `(H+CN) / N` |

**3c.** Add a module-level constant near the top of `src/verification/__init__.py` (below
imports) listing the thresholds, then call `_compute_categorical_scores()` inside the
per-region loop in `verify()`, after the existing `_compute_scores()` call:

```python
# Module-level constant — extend to add more parameters/thresholds
CATEGORICAL_THRESHOLDS: dict[str, float] = {
    "TOT_PREC": 1.0,  # mm — adjust as agreed with user
}
```

```python
# Inside the per-region loop in verify(), after the _compute_scores() append:
if param in CATEGORICAL_THRESHOLDS:
    score.append(
        _compute_categorical_scores(
            fcst_param,
            obs_param,
            threshold=CATEGORICAL_THRESHOLDS[param],
            prefix=param + ".",
            source=fcst_label,
            dim=dim,
        ).expand_dims(region=[region])
    )
```

**3d.** If the user requested CLI-configurable thresholds, also update
`verif_single_init.py`:

1. Add a `--thresholds` argument (format `PARAM:value,PARAM:value`) to the `ArgumentParser`.
2. Parse it into a `dict[str, float]` and pass it to `verify()` as a new keyword argument.
3. Update `verify()` and `_compute_categorical_scores()` signatures accordingly, replacing the
   module-level constant with the passed-in dict.

### Post-aggregation transform (if needed)

If the metric needs a rename or sqrt-transform after time-averaging, add it to the
`var_transform` dict in `verif_aggregation.py:aggregate_results()`:

```python
var_transform = {
    d: d.replace("VAR", "STDE").replace("var", "std").replace("MSE", "RMSE")
    # .replace("OLD_NAME", "NEW_NAME")   ← add here
    for d in out.data_vars
    if "VAR" in d or "var" in d or "MSE" in d  # extend condition if needed
}
```

---

## Step 4 – Add unit tests

Create or extend `tests/unit/test_verification.py`.

### For a continuous metric

```python
import numpy as np
import xarray as xr
from verification import _compute_scores


def test_<metric>_perfect_forecast():
    vals = xr.DataArray([1.0, 2.0, 3.0], dims=["values"])
    ds = _compute_scores(vals, vals, dim=["values"], prefix="T_2M.")
    assert float(ds["T_2M.<METRIC>"]) == <expected_perfect_value>


def test_<metric>_known_case():
    fcst = xr.DataArray([...], dims=["values"])
    obs  = xr.DataArray([...], dims=["values"])
    ds = _compute_scores(fcst, obs, dim=["values"], prefix="T_2M.")
    assert np.isclose(float(ds["T_2M.<METRIC>"]), <expected_value>)
```

### For a categorical metric

```python
import numpy as np
import xarray as xr
from verification import _compute_categorical_scores


def test_<metric>_perfect_forecast():
    # All above threshold → all hits
    fcst = xr.DataArray([2.0, 3.0, 2.0], dims=["values"])
    obs  = xr.DataArray([2.0, 3.0, 2.0], dims=["values"])
    ds = _compute_categorical_scores(
        fcst, obs, threshold=1.0, dim=["values"], prefix="TOT_PREC."
    )
    assert float(ds["TOT_PREC.<METRIC>"]) == <perfect_score>


def test_<metric>_known_contingency_table():
    # H=2, FA=1, M=1, CN=0 — verify by hand before writing the assertion
    fcst = xr.DataArray([2.0, 2.0, 0.5, 2.0], dims=["values"])
    obs  = xr.DataArray([2.0, 2.0, 2.0, 0.5], dims=["values"])
    ds = _compute_categorical_scores(
        fcst, obs, threshold=1.0, dim=["values"], prefix="TOT_PREC."
    )
    assert np.isclose(float(ds["TOT_PREC.<METRIC>"]), <expected_value>)
```

Always include at least: a perfect-forecast test, a hand-computable known-values test,
and a no-skill or all-miss test if meaningful for the metric.

---

## Step 5 – Confirm and validate

After writing all code, tell the user:

1. Which files were modified and what changed in each.
2. How to run the new tests:
   ```
   uv run pytest tests/unit/test_verification.py -v
   ```
3. That the metric will appear automatically in plots and the dashboard once the pipeline
   runs — no changes to `verif_plot_metrics.py`, `report_experiment_dashboard.py`, or
   `script.js` are needed, because those discover metrics dynamically from dataset variable names.
