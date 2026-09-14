# Experiment plan: assessing the added value of the multistep approach

## Question

Does a single model predicting multiple hourly steps jointly per autoregressive
pass (multistep) represent sub-6-hourly weather better than Varda-single's
two-stage forecaster plus temporal downscaler?

## Background

**Varda-single (two-stage).** A 6-hourly autoregressive forecaster, plus a
temporal downscaler that reconstructs the five intermediate hourly states from
the two bounding 6h states. Both are stretched-grid Graph Transformers trained
on MSE. The downscaler is not a numerical interpolator: it sees the full 3D
state (t, q, u, v, z on 13 pressure levels) at both endpoints plus forcings
(insolation, local solar time, Julian day, orography, land-sea mask), and it was
trained on hourly REA-L-CH1 samples. It can therefore diagnose advection and
diurnal forcing, and it is not bounded by its endpoints.

**Key structural point.** The downscaler output is a diagnostic side-branch. At
t+6 everything except precipitation is discarded and replaced by the
forecaster's field. No sub-6h state ever feeds back into the forecast
trajectory. The 6-hourly forecaster must therefore learn an effective 6h map
that implicitly integrates sub-6h processes, in the way a parameterisation
stands in for unresolved physics.

The multistep model (`in7_out6`) takes 7 hourly input steps and produces 6
hourly outputs per pass, so its hourly states *are* carried forward. Both
systems take the same number of autoregressive passes to reach +120h (20), so
rollout count is roughly controlled.

**Motivating hypothesis, from the Varda paper's Limitation 2.** Processes that
need km-scale information live at timescales much shorter than 6h. A forecaster
that steps over those timescales has little incentive to encode km-scale detail
in its latent space. If true, the multistep model should be better *at the 6h
anchor steps as well*, not only in between.

**Constraint on what can discriminate.** Both models are MSE-trained and both
will hedge toward the conditional mean. The paper shows (Sect. 4.3.3) that
structure error in convective precipitation is fully developed after a single 6h
step and barely grows afterwards, i.e. small-scale convective organisation is
already unpredictable at 6h range. Chaotic high-frequency phenomena (convective
cells, gust fronts) will therefore come out flat for both systems and cannot
discriminate. Discriminating phenomena must be high-frequency **and predictable**,
meaning forced by the diurnal cycle and terrain rather than by convective
instability.

## Plan

### Step 1 (primary): temporal variability at sub-6h scales

Compute the temporal power spectral density of hourly T_2M and 10m wind at
station points / grid points over long forecast segments, for Varda stage C,
the multistep model, and REA-L as truth.

Hypothesis: Varda shows a deficit in spectral power at periods below roughly 6h;
the multistep model recovers part of it.

#### What hourly data can and cannot show

Hourly sampling puts the Nyquist limit at a **2h period**. Nothing shorter is
present in the data at all, so convective cells, gust fronts and anything at 10
to 30 minutes are invisible. The accessible band is 2 to 6h, not "everything
below 6h", and the plan's wording should be read that way throughout.

Resolution inside that band is adequate. A 48h window gives bins at 48/k hours,
so k = 8, 10, 12, 16, 20, 24 correspond to 6.0, 4.8, 4.0, 3.0, 2.4 and 2.0h:
17 bins between 6h and 2h.

The limitation is **aliasing**, not resolution. Hourly output of T_2M is an
instantaneous sample and the atmosphere is not band-limited at 2h, so genuine
sub-2h variance folds back into the resolved band and lands on exactly the
frequencies of interest. Worse, it does so unequally: a temporally smooth model
has little sub-2h variance to fold in, a 1km dataset has more. Aliasing
therefore inflates the truth's apparent 2-6h power relative to the models' and
makes any deficit look larger than it is. That bias points the same way as the
hypothesis, which is the direction that can manufacture a false confirmation.

Consequences:

- Do not interpret the bins closest to Nyquist, roughly 2 to 3h. Claims should
  be restricted to about **3 to 6h**.
- Treat any measured deficit as an upper bound.
- State results as relative comparisons between identically sampled series, not
  as measurements of the atmosphere's real spectrum.

#### Two diagnostics that do not depend on near-Nyquist behaviour

The spectrum should not carry Step 1 alone.

**Hourly increment variance (robust primary).** The variance of hour-to-hour
differences at each point: one number per source, capturing how much each system
actually changes from hour to hour. It is the spectrum weighted toward high
frequencies, so it tests the same thing with far fewer assumptions and no
near-Nyquist fragility.

**A spectral line at exactly 6h.** Varda switches field source every 6h, and
precipitation already shows "blinking" at block boundaries. If that seam leaves
a signature it appears as a spike at exactly the 6h bin (k = 8 in a 48h window)
in Varda, absent from the multistep model and from the truth. Well away from
Nyquist and hard to explain away. But see the REA-L join caveat below, which
puts a harmonic on that same bin.

#### REA-L is not an analysis, and that cuts both ways

REA-L-CH1 is assembled from 24-hourly forecast runs with latent heat nudging at
00 UTC only. It is closer to a sequence of free forecasts than to an analysis,
hence "reanalysis light".

Good: it is not a smoothed product, so it carries full model-generated
variability and the earlier worry that there might be no 2-6h signal at all is
much reduced. It is also what both models were trained on, so the comparison is
internally consistent.

Bad: consecutive days come from different model runs, so there may be a
**discontinuity every 24h** where segments meet. A repeating jump is broadband,
placing lines at 24h and every harmonic, including **6h and 4h**, both inside
the band of interest, and 6h is exactly the bin proposed above as the seam test.
The ML models roll forward autoregressively with no daily joins, so they cannot
reproduce this artefact and should not be expected to: it would read as a model
deficit that is really a property of how the truth was built. Same dangerous
direction as aliasing.

#### Result: there is a 00 UTC artefact, but it is small and surface-only

Measured by `workflow/scripts/join_detect.py` (RMS hourly increment against hour
of day; a one-hour spike on top of the smooth diurnal shape indicates an
artefact). 21 contiguous days per season, 200 grid points, all seven Varda
surface fields plus three upper-air levels. The "spike index" divides each hour
by the mean of its neighbours, so smooth diurnal structure gives ~1.

Spike index at 00 UTC:

| variable | winter | summer |
|---|---|---|
| T_2M | **1.57** | 1.09 |
| TD_2M | **1.33** | 1.12 |
| V_10M | **1.28** | 1.04 |
| U_10M | 1.08 | 1.14 |
| PMSL | 0.91 | 0.92 |
| PS | 0.90 | 0.92 |
| TOT_PREC_1H | 0.92 | 0.75 |
| T_850 | 1.03 | 1.01 |
| QV_850 | 1.03 | 1.00 |
| FI_500 | 0.94 | 0.91 |

Nothing physical happens at 00 UTC (01:00 local in winter), so a one-hour
feature at that clock time is an artefact of dataset construction.

**Latent heat nudging is ruled out as the cause.** It acts on latent heating in
the column and would appear in QV_850, T_850 and precipitation. All three sit at
or below 1.0, precipitation included.

**A surface-side daily update fits.** The artefact is confined to near-surface
fields, winter only, with nothing in pressure, precipitation or any upper-air
level. That is the signature of a soil or snow analysis at 00 UTC: it perturbs
the surface, the stable decoupled winter boundary layer passes that straight
into the 2m and 10m fields, and in summer it is mixed away and buried under
larger variance. Not a wholesale segment restart, which would move the mass
field too. The mechanism is inferred from this pattern, not confirmed from the
production chain, and someone who knows that chain could settle it in a
sentence. V_10M being flagged while U_10M is not is more likely sampling noise
than physics.

**Consequences, measured not assumed.** Repairing the 00 UTC value (replacing it
by the mean of its neighbours) and recomputing the spectrum changes the winter
T_2M 3-6h band by only **+6.5%**. T_2M has the largest spike index of all ten
variables, so that is an upper bound for the rest.

- **48h windows stand.** No need to drop to 24h segment-aligned windows, so we
  keep 17 bins between 6h and 2h rather than 8.
- **The 6h seam test stands.** The perturbation varies in size and sign between
  days, so it is not a coherent comb of lines at 24h harmonics sitting on the 6h
  bin; it behaves as a small broadband lift.
- Record as a caveat: roughly 6% inflation of REA-L's 3-6h variability in
  near-surface winter fields, which the models cannot reproduce. Applies to
  T_2M, TD_2M and the wind components in winter only.
- Precipitation is clean with respect to this artefact, so the "blinking" noted
  below is purely a Varda-side 6h issue, not a property of the truth.

#### No reference is clean

- **REA-L**: daily join artefacts at 24h harmonics, as above.
- **KENDA-CH1**: assimilates hourly, so every hour carries an analysis
  increment, part genuine correction and part assimilation noise, appearing at
  the highest resolvable frequencies. It would tend to show too much 2-6h power,
  the opposite error to REA-L.
- **SwissMetNet**: averaging intervals differ per parameter, and the code
  retrieves temperature as `tre200s0`, 10m wind as `fkl010z0` (a 10-minute
  mean) and precipitation as `rre150h0` (an hourly sum). Block averaging is a
  low-pass filter that bites hardest exactly across our band: for a 1h mean
  sampled hourly the power attenuation is about 9% at 6h but about 60% at 2h.
  Correctable by dividing by the filter transfer function, but only if the
  interval is known, and it must be done per parameter. **To check before any
  quantitative use: whether `tre200s0` is an hourly mean or an instantaneous
  value.** Comparing hourly-mean observations against instantaneous model output
  without correction would make the observations look artificially smooth.

Each reference's artefacts must be named when its curve is shown.

This is the temporal counterpart of the spatial smoothing argument the paper
already makes, and it covers all high-frequency phenomena in one diagnostic
instead of one case at a time.

Stratify by: lead time, position within the 6h block, and convectively active
versus quiet days.

Possible null result worth anticipating: both spectra are equally flat, because
MSE hedging suppresses variance regardless of temporal resolution. That would
itself be a finding, pointing at the training objective (paper's Limitation 4)
rather than at the multistep design.

Beyond Varda and the multistep model, include a third curve: the Varda
6-hourly forecaster field linearly interpolated to hourly. This separates how
much sub-6h power the learned downscaler actually recovers from what plain
interpolation would give, which is a much sharper statement than "Varda has a
deficit". It is free, because the forecaster fields are written to
`{init}/forecaster/` by the two-stage run anyway.

Two technical points that decide whether this diagnostic means anything:

- **Window, not whole trajectory.** A 0-120h series is not stationary: it holds
  the synoptic evolution plus a growing error. Transform fixed windows instead,
  which also gives the lead-time stratification: three overlapping 48h windows
  (0-48, 36-84, 72-120). 24h windows would be conceptually cleaner but leave too
  few frequency bins.
- **Detrend and taper.** Diurnal and synoptic power sits orders of magnitude
  above the 2-6h band. Without removing the mean and a linear trend and applying
  a Hann taper, leakage from the low frequencies swamps exactly the band of
  interest and both models look identical for a purely numerical reason. Worth a
  unit test on a synthetic series.

**Pilot order, before committing the compute.** Two checks on the truth, both
cheap, both to be done before any forecast GRIB is read. They are ordered
because the first decides the design of the second.

1. **Join detection** (`workflow/scripts/join_detect.py`). Is there a daily
   discontinuity from REA-L's concatenated forecast segments? Decides the window
   length: 48h windows if not, 24h segment-aligned windows if so.
2. **Spectral pilot** (`workflow/scripts/spectra_pilot.py`). Does REA-L carry
   usable 2-6h power in T_2M at the sample points and seasons? If not, neither
   model can be shown to be missing anything and Step 1 is unfalsifiable. Now
   less likely to fail than originally feared, see the "reanalysis light" note
   above, but still worth confirming.

The spectral pilot also plots REA-L subsampled to 6h and linearly interpolated
back to hourly. That is the floor Varda's learned downscaler must beat to be
doing anything beyond interpolation, and the gap between it and REA-L is the
power in principle available for the multistep design to win.

Both scripts carry a synthetic self-test: the estimator must recover a known
weak 3h signal sitting under a 25x stronger diurnal cycle and a trend. Given how
much leakage and aliasing matter here, no figure should be produced from an
unvalidated estimator.

#### Pilot result: Step 1 is viable

T_2M, 15 windows of 48h per season (`output/spectra_pilot/`). Station points are
the 145 SwissMetNet locations mapped to REA-L grid points by
`workflow/scripts/station_grid_map.py`; grid points are a uniform random sample
of 200, kept as a contrast.

| | winter station | summer station | winter grid | summer grid |
|---|---|---|---|---|
| 3-6h share of total variance | **1.71%** | **0.89%** | 1.24% | 0.80% |
| REA-L / (6h + linear interp) in 3-6h | **7.8x** | **5.0x** | 5.7x | 4.4x |
| 00 UTC artefact inflation of 3-6h | +3.8% | +1.9% | +6.5% | +1.7% |
| PSD at 6h | 0.690 | 1.058 | 0.538 | 1.156 |
| PSD at 4h | 0.278 | 0.370 | 0.185 | 0.456 |
| PSD at 3h | 0.167 | 0.218 | 0.095 | 0.246 |

**Station points are the better sample for this experiment.** In winter they
carry about 40% more power at 6h and 75% more at 3h than a uniform grid sample,
because stations sit in valleys and settled terrain where local forcing is
strong while a uniform sample includes much smooth high terrain. Two
consequences both favour stations: the headroom over pure interpolation is
larger (7.8x vs 5.7x in winter), so the diagnostic is more sensitive where we
intend to use it, and the 00 UTC artefact is smaller (+3.8% vs +6.5%), so the
contamination caveat shrinks. Use station points as the headline and grid points
as a check that conclusions do not depend on where one looks.

**These are model grid points, not station observations.** Every source (REA-L,
Varda, the multistep model) is read at the same 145 grid points, chosen as the
nearest to the SwissMetNet locations so that station data can be brought in
later without changing the sample. No observations are involved yet.

Because all sources share the same grid points, representativeness cancels out
of the model-vs-model comparison. In particular the three summit stations whose
model cell is far below the real peak (JUN +311m, SAE +319m, TIT +463m, where
the 1km orography cannot resolve the summit) need no special treatment here.
They matter only once real station data enters, where a model column is compared
against an instrument.

The mapping is clean: 145 stations to 145 distinct grid points, no collisions,
median distance 360m, elevation differences median -2m (IQR -31 to +34m).

- REA-L carries real power at 3-6h in both seasons and the spectrum falls off
  smoothly, with no numerical floor. **Step 1 is falsifiable.**
- The headroom over pure interpolation is large in both seasons. At exactly 6h
  period the gap is about 100x (winter) and 157x (summer), and at 3h about 175x
  and 360x. Those deep notches are expected, since linear interpolation from
  6-hourly samples has exact zeros at periods of 6h and 3h, and they confirm the
  degradation behaves correctly.
- Varda's learned downscaler sits somewhere inside that headroom: near the
  interpolation curve means it adds little beyond interpolation, near REA-L
  means it does real work. Either way the diagnostic separates them.

**Summer has more absolute sub-6h power, winter more relative.** Summer PSD in
the band is about 2.4x winter's, but its share of total variance is smaller
because the summer diurnal cycle is far stronger (24h PSD 4.7x winter's). The
headroom is also slightly smaller in summer (4.4x vs 5.7x), i.e. interpolation
captures relatively more of the summer sub-6h signal, consistent with more of it
being tied to the smooth strong diurnal cycle.

This does not overturn the choice of winter as the primary test. The pilot
measures how much sub-6h power exists, not how much of it is *predictable*, and
the argument for winter was always that its sub-6h variance is dominated by
forced, predictable processes while summer's is largely convective and
unpredictable, so both models flatten it regardless. Summer remains the stress
test, with the convective stratification applied rather than pooling all days.

**Expectation to set before interpreting Step 2.** The 3-6h band holds only
about 1% of total T_2M variance in either season, since diurnal and synoptic
scales dominate. So even a complete win for the multistep model in this band
will barely move overall RMSE. This is why the spectral diagnostic is the right
tool, and why small differences in the Step 2 anchor-versus-in-between scores
should not be read as "no effect".

### Step 2: anchor versus in-between decomposition

Split scores at multiples of 6h (pure forecaster versus pure forecaster, no
downscaler involved) from all other lead times.

- Better only in between: the benefit is in the reconstruction step, and
  Limitation 2 is not supported.
- Better at the anchors too: Limitation 2 is supported.

Cheap, comes out of the existing verification, and it determines which mechanism
the case studies should target.

**Confound, and the control for it.** Varda's forecaster only accepts inits on
the 6-hourly grid, so init hours are always multiples of 6. The anchor lead
times therefore *always* fall at 00/06/12/18 UTC and the in-between steps always
at other hours, whatever init stride is chosen. The anchor-versus-in-between
contrast is thus structurally entangled with time of day, and no sampling design
removes it.

The control is the multistep model itself. It has no 6h seam, so any 6-hourly
periodicity in *its* error-versus-lead-time curve is diurnal or otherwise
spurious. Subtract that from Varda's curve; what remains is attributable to the
seam. Read Varda's sawtooth on its own and the result is not interpretable.

Sampling all four init hours does not fix the confound but does damp it, since
each lead time then averages over four widely spaced times of day. This is why
the init stride must not be a multiple of 24h (see Setup).

### Step 3: feedback test

The distinctive prediction of the multistep design is that resolving sub-6h
states improves the *subsequent* evolution, because those states are carried
forward. Test by comparing skill at day 2 and day 3 following convectively
active versus quiet days.

### Step 4: case studies, as illustration

Run only after steps 1 to 3, to illustrate an effect already established.

Already covered by a collaborator, so avoid unless extending deliberately:
valley-scale cold-air pooling, Alpine Foehn, diurnal valley winds.

Preferred new candidate: **low stratus / fog burn-off on the Swiss Plateau.**
Sharp transition (1 to 2h), diurnally forced and therefore predictable,
operationally relevant, distinct from the existing three, and it exercises TD_2M
and the pressure-level humidity that are otherwise unused.

Secondary candidate: **peak attenuation during a windstorm.** Reframed as "does
MSE hedging in the downscaler attenuate peaks more than in a direct 1h model",
not as an interpolation bound. May well come out neutral.

Deprioritised: frontal passage (the downscaler has the upper-air state at both
endpoints and can infer propagation, so there is no structural reason for it to
fail); thunderstorm gust fronts and convective initiation (predictability limit
dominates, see above).

## Setup and caveats

Comparison configured in `config/multistep-stage-C-winter2024.yaml` and
`config/multistep-stage-C-summer2024.yaml`. Both models are stage C: no rollout
training, REA-L-CH1 only.

### Sampling

Two seasons, 41 inits each, 82 total, at a 54h stride.

| season | inits | init hours 00/06/12/18 | range |
|---|---|---|---|
| winter 2024 (Jan + Feb + Dec) | 41 | 10 / 11 / 11 / 9 | 2024-01-01T00 → 2024-12-30T12 |
| summer 2024 (JJA) | 41 | 11 / 10 / 10 / 10 | 2024-06-02T00 → 2024-08-31T00 |

The stride must be a multiple of 6h (Varda's forecaster training stride) but
**not** a multiple of 24h, or the init hour never changes and lead time becomes
identical to time of day. `stride mod 24` gives the init-hour cycle: 30h, 42h and
54h all give 6 and so cycle through 00/06/12/18; 36h gives 12 and samples only
two init hours; 24h and 48h sample one. Note that 48h over two seasons (90
inits) is *more* expensive than 54h over two seasons (82) and gives up the
coverage for nothing.

Inits are given as explicit lists, not start/end ranges, taken from a single 54h
grid anchored at 2024-01-01T00 and then filtered by month. Anchoring the grid
once rather than restarting the stride per month block is what keeps the init
hours balanced.

### Why the winter sample is Jan + Feb + Dec 2024

Two hard constraints, from a coverage audit of the actual datasets:

1. Inits must be after Varda's training cutoff, i.e. 2024-01-01 or later.
2. Inits must lie inside the global initial-condition datasets, which end
   **2024-12-31**: `aifs-ea-...-n320-1979-2024-6h-v1-for-single-v2` (Varda) at
   2024-12-31T18 and `aifs-ea-...-n320-1979-2024-1h-v2-with-era51` (multistep)
   at 2024-12-31T23. There is no 2025 global dataset in
   `/store_new/mch/msopr/ml/datasets/` at all, so this cannot be worked around
   by swapping datasets; it would need operational IFS analyses, which is a
   stage E setup and no longer this comparison.

So DJF 2024/25 is impossible (no ICs for Jan/Feb 2025) and DJF 2023/24 is
contaminated (Dec 2023 is Varda training data). The winter months satisfying
both constraints are January, February and December 2024. That is 91 days, a
full DJF's worth, but spanning two different winters. Label it "winter 2024",
not DJF 2024.

Only the initial condition needs global data, not every forecast step: the
models' forcing inputs are `insolation`, `cos/sin_julian_day`,
`cos/sin_local_time` plus static fields (`lsm`, `sdor`, `slor`, lat/lon), all
either constant or computable from the valid time. So late-December inits may
run into January 2025, which the REA-L truth covers (it ends 2025-03-31). This
is inferred from the checkpoint metadata's forcing list and should be confirmed
by one test run at a late-December init before the full campaign.

### Training and validation splits differ between the two models

Extracted from the embedded `anemoi.json` in each checkpoint:

| | training | validation | test |
|---|---|---|---|
| Varda stage C | up to 2023 | 2024 onward | 2024 onward |
| Multistep stage C | 1985-2020 | 2021 | 2022 onward |

For the 2024 sample this means: Varda's *validation* set, the multistep model's
*test* set. **Neither model trained on it**, so the earlier caveat that 2024 is
the validation period of both models overstated the problem.

Two residual asymmetries, to report rather than fix:

- 2024 influenced Varda's checkpoint selection and early stopping but was pure
  test data for the multistep model. That tilts slightly in Varda's favour.
  Because Varda's validation range is open-ended from 2024 and REA-L ends March
  2025, there is no period that is both outside Varda's training set and outside
  its validation set. This is the best available.
- Varda trained on REA-L 2005-2023, the multistep model on 2005-2020 (its
  configured 1985 start predates REA-L). Different training volume, unrelated to
  the multistep idea.

### Data coverage, verified

| dataset | role | coverage |
|---|---|---|
| `mch-realch1-...-pl13-v1.0.zarr` | truth | 2005-01-01T12 → 2025-03-31T23, 1h, 1km |
| `mch-realch1-...-ifsnames-v1.0` | Varda LAM IC | 2005 → 2025-03-31 |
| `mch-realch1-...-ifsnames-1h-precip-v1.0` | multistep LAM IC | 2005 → 2025-03-31 |
| `aifs-ea-...-6h-v1-for-single-v2` | Varda global IC | 1979 → **2024-12-31T18** |
| `aifs-ea-...-1h-v2-with-era51` | multistep global IC | 1979 → **2024-12-31T23** |
| ICON-CH1-EPS | baseline | 3-hourly inits, complete 2024-2025 |
| ICON-CH2-EPS | baseline | 6-hourly inits, complete 2024-2025 |

The truth dataset also carries `CAPE_ML`, `CAPE_MU`, `CIN_ML`, `LPI` and
`CLCL`/`CLCT`, so the convective-day classification for Step 3 and the low
stratus case selection for Step 4 can both be built from the truth dataset
itself. No extra data needed.

ICON-CH1-CTRL is scored against REA-L-CH1, which shares its model climate, so it
is flattered here. Keep it as a reference line and do not read a ranking into
it.

- Varda forecaster: `/scratch/mch/apennino/output/checkpoint/a69e95db4b494679ab20440f8540a835/inference-last.ckpt`
- Varda downscaler: <https://service.meteoswiss.ch/mlstore#/models/sruc-m-2-interpolator/versions/3>
- Multistep: `/scratch/mch/miccatta/anemoi-outputs/checkpoint_stage_C_multistep_in7_out6_BALFRIN/0eb1b8759c3c4ad188bc1e7cf8d0f1ab/inference-last.ckpt`

Pairing the stage C forecaster with the released downscaler is not a mismatch:
per the paper's Table 3 the downscaler only ever had the pre-training and
stretched-grid stages, no rollout and no operational fine-tuning, so it is at its
final state either way.

Caveats to keep in view:

- Evaluation period is 2024. Checked: it is out of training for both models,
  Varda's validation and the multistep model's test set. See the splits table
  above for the residual asymmetry.
- Different history lengths: multistep sees 7h (1h timestep × 7 input steps),
  Varda's forecaster sees 12h (6h × 2). Confirmed from checkpoint metadata.
  Unrelated to the multistep idea but affects scores.
- Different architectures trained separately, so "multistep wins" does not by
  itself validate the multistep *approach*. Step 2 is what sharpens this.
- Precipitation is diagnostic, diagnosed per time step, and shows temporal
  autocorrelation artefacts ("blinking") across 6h block boundaries. Consistent
  with the seam at t+6 where the field switches source. Treat precipitation
  separately and with care.
