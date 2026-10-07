# Multistep vs Varda-single: temporal-variability findings

Status 2026-09-25. Companion to `exp_plan_01.md`, which holds the experiment
design. This file states what is currently known. Superseded intermediate
results have been removed rather than annotated.

All numbers are 3-6 h band variance at 145 SwissMetNet station points, estimated
with a zero-phase Butterworth bandpass over the full 121 h forecast, medians over
stations, 2024 winter (JFD) and summer. Section 8 says why that estimator and how
much the choice matters.

---

## 0. Bottom line

1. **Both models lose most of the sub-6 h variability.** They retain 3-16 % of
   what the stations see and 6-26 % of what REA-L holds. REA-L itself holds only
   about half the observed signal, so the gap to reality is roughly twice the gap
   measured against the training target. This dwarfs any difference between the
   two model designs and points at MSE training, not at temporal resolution.
2. **Multistep does not retain more variability than Varda-single; if anything
   Varda retains slightly more.** Pooled over stations, multistep never retains
   significantly more in any variable, season, band, taper, window length or
   estimator tested. Varda retains significantly more in six of the eight cases,
   and the two T_2M exceptions are inconclusive rather than reversed. The
   near-surface differences are small, 0.01-0.025 in retained fraction, against
   the ~90 % both models lose, so this is no improvement from multistep with a
   small, consistent lean toward Varda, not a meaningful Varda advantage. Within
   station classes most intervals include zero, and the one class result in
   multistep's favour (winter T_2M, `local`) may well be an artefact (section 6).
3. **Neither model knows when the sub-6 h variability happens.** The 3-6 h
   timing correlation against REA-L is 0.02-0.29, so at best about 8 % of the
   band variance is explained, and usually far less. The variance that is
   retained is largely not the right variance. This is a bigger fact than any
   difference between the two designs (section 3).
4. **Multistep is never more skilful either.** Against REA-L it is significantly
   worse in all four winter variables and in summer T_2M, and tied in the rest.
   So there is no variance-skill tradeoff: Varda's extra variance costs it
   nothing. But it does not buy anything either, because it is not better timed
   (section 3).
5. **Surface pressure is the clearest case by variance, and the most misleading.**
   Varda retains more at 145 of 145 stations in winter and 144 of 145 in summer,
   in every band tested,
   yet in winter its band timing correlation is 0.023, indistinguishable from
   zero, against multistep's 0.161. There the variance diagnostic is rewarding
   noise (section 3).
6. **What this still does not tell you.** Skill here is measured against REA-L,
   which section 4 shows is itself heavily smoothed. The SwissMetNet analogue
   has not been run.
7. **NEW 2026-09-25, not yet reviewed by Louis: the ICON-CH2 baseline is no
   better timed than the ML models, and mostly worse, despite realistic variance.** It retains
   61-109 % of REA-L's 3-6 h variance (the ML models 7-26 %), but its timing
   correlation is 0.01-0.17. So the low timing correlation is not an ML failure. It
   points to limited predictability of this band at 1-4 days lead (section 14).
8. **NEW 2026-09-28, not yet reviewed: at longer time scales the picture changes,
   but not in multistep's favour.** The variance deficit is a sub-daily problem:
   at the daily cycle and synoptic scales both ML models hold roughly REA-L's
   variance. Timing becomes much better with scale (r 0.03-0.15 at 3-6 h,
   0.22-0.44 at 6-12 h, 0.55-0.88 for the daily cycle, 0.69-0.98 synoptic, with
   the sun-driven part removed).
   At no scale does multistep retain more or time better than Varda-single, and
   on synoptic scales it is less accurate. That is where its overall MSE deficit
   sits (section 15).

---

## 1. The dominant signal is a deficit shared by both models

The difference between the two designs is small next to what they have in
common. 3-6 h variance as a fraction of the station observations:

| case | REA-L | Varda-single | Multistep |
|---|---|---|---|
| winter T_2M | 0.43 | 0.060 | 0.065 |
| winter TD_2M | 0.80 | 0.085 | 0.071 |
| winter SP_10M | 0.43 | 0.035 | 0.034 |
| summer T_2M | 0.55 | 0.149 | 0.138 |
| summer TD_2M | 1.02 | 0.155 | 0.140 |
| summer SP_10M | 0.66 | 0.087 | 0.082 |

Both models sit at a few per cent of reality for the near-surface fields. The
model-to-model differences below are differences between 0.06 and 0.07, inside a
shared shortfall of 0.94.

This points at the training objective rather than at temporal resolution, and is
close to the null result anticipated in the plan (Limitation 4): MSE-trained
models hedge toward the conditional mean and suppress variance regardless of how
finely they step in time. Nothing about the multistep design changes that.

**The deficit does not grow with lead time.** Checked per 24 h lead window out to
120 h: retention is flat, so this is a fixed property of the models rather than a
spin-up transient or something that develops as the forecast ages.

**The deficit worsens toward shorter periods.** Winter T_2M retention against
REA-L runs about 0.13 in the 3-6 h band against 0.35 in 4-8 h. That is what an
over-smoothing model looks like.

## 2. Multistep does not retain more variability than Varda-single

The comparison is **paired by station**: the statistic is the median over the 145
stations of `FRAC_Varda - FRAC_Multistep`. Comparing the two marginal medians
instead, which is what a side-by-side boxplot invites, is unreliable; for summer
T_2M the two marginal medians coincide to four digits while the paired median
difference is clearly positive.

Confidence intervals are 2000-sample paired bootstraps resampling **inits**, which
are 54 h apart and near independent. Stations are spatially correlated, so they
are held fixed; the intervals are still somewhat optimistic.

| case | Varda | Multi | gap | 95 % CI | P(gap>0) | n stations V>M |
|---|---|---|---|---|---|---|
| winter T_2M | 0.133 | 0.153 | -0.003 | [-0.012, +0.001] | 0.05 | 64/145 |
| winter TD_2M | 0.091 | 0.075 | +0.014 | [+0.009, +0.021] | 1.00 | 114/145 |
| winter SP_10M | 0.071 | 0.062 | +0.008 | [+0.003, +0.012] | 1.00 | 97/145 |
| winter PS | 0.176 | 0.109 | +0.062 | [+0.048, +0.085] | 1.00 | 145/145 |
| summer T_2M | 0.258 | 0.246 | +0.023 | [-0.005, +0.039] | 0.93 | 86/145 |
| summer TD_2M | 0.149 | 0.134 | +0.025 | [+0.012, +0.036] | 1.00 | 104/145 |
| summer SP_10M | 0.134 | 0.123 | +0.012 | [+0.004, +0.018] | 1.00 | 95/145 |
| summer PS | 0.218 | 0.113 | +0.108 | [+0.082, +0.136] | 1.00 | 144/145 |

In six of eight cases Varda retains significantly more. The two that do not are both
T_2M, and both are inconclusive rather than favouring multistep: winter T_2M has
an interval straddling zero, and summer T_2M likewise.

**Surface pressure is the strongest result in the study by this measure, and
section 3 shows why that measure misleads here.** Varda retains roughly 1.6x
(winter) to 1.9x (summer) what multistep does, at 145 of 145 stations in winter
and 144 of 145 in summer, in every station class and every band. Nothing else
here is close to that unanimous. But in winter Varda's 3-6 h pressure variance
is essentially uncorrelated with REA-L's (r = 0.023 against multistep's 0.161,
multistep better timed at 139 of 145 stations), so the unanimity is Varda
producing more noise, not more signal.

**Sensitivity to the band.** Widening or narrowing the 3-6 h band leaves seven of
the eight cases with the same sign:

| case | 3-6 | 4-6 | 3-8 | 4-8 |
|---|---|---|---|---|
| winter T_2M | -0.003 | -0.004 | **+0.013** | **+0.009** |
| winter TD_2M | +0.014 | +0.016 | +0.017 | +0.016 |
| winter SP_10M | +0.008 | +0.006 | +0.010 | +0.012 |
| winter PS | +0.062 | +0.074 | +0.070 | +0.076 |
| summer T_2M | +0.023 | +0.026 | +0.008 | +0.005 |
| summer TD_2M | +0.025 | +0.017 | +0.041 | +0.037 |
| summer SP_10M | +0.012 | +0.014 | +0.016 | +0.019 |
| summer PS | +0.108 | +0.151 | +0.157 | +0.192 |

Winter T_2M is the exception: its slight lean toward multistep exists only when
the band stops at 6 h and reverses once 6-8 h is included. Combined with an
interval that straddles zero and a sign flip under a 24 h analysis window, it
does not support a multistep advantage.

Figure: `output/gap_robustness/paired_gap_overview.png`, which plots the 145
per-station differences with bootstrap intervals for all eight cases. Use it
rather than `output/FRAC_var_bxplt_overview/`, which shows unpaired marginals.

## 3. Skill: multistep is never ahead, and the extra variance is not the right variance

Sections 1 and 2 measure variance, not skill. This section pairs them, against
REA-L on both axes so the two are not measured against different truths. The
SwissMetNet analogue has not been run (section 12). Errors are taken over the
same 121 h record with the same 24 h guard bands as the variance, so both axes
see identical samples; 35 of 41 winter and 37 of 41 summer inits are usable
after the NaN drop of section 9.

**Multistep is never significantly more skilful.** Murphy skill score with
multistep as candidate and Varda-single as baseline, `1 - MSE_Multi/MSE_Varda`,
so **positive means multistep is better**. Note this is the opposite orientation
to the gap in section 2. Intervals are the same paired init bootstrap.

| case | MSSS | 95 % CI | n stations Multi better |
|---|---|---|---|
| winter T_2M | -0.136 | [-0.198, -0.095] | 25/145 |
| winter TD_2M | -0.102 | [-0.173, -0.046] | 32/145 |
| winter SP_10M | -0.086 | [-0.138, -0.014] | 48/145 |
| winter PS | -0.346 | [-0.645, -0.096] | 8/145 |
| summer T_2M | -0.127 | [-0.179, -0.036] | 40/145 |
| summer TD_2M | +0.023 | [-0.058, +0.089] | 81/145 |
| summer SP_10M | +0.011 | [-0.017, +0.032] | 81/145 |
| summer PS | +0.031 | [-0.119, +0.158] | 84/145 |

Multistep is significantly worse in all four winter variables and in summer
T_2M. The three remaining summer cases have intervals straddling zero, so they
are ties, not wins. **There is no variance-skill tradeoff**: multistep both
retains less variance and scores no better, so Varda's extra variance costs it
nothing in accuracy.

That is not the same as saying the extra variance is useful, which is what the
rest of this section tests.

### Neither model knows *when* the 3-6 h variability happens

Section 2 and the table above cannot say whether variance arrives at the right
moment. Splitting the band error answers that. For a
bandpassed series the mean is zero, so

    MSE_band = (sd_f - sd_t)^2  +  2 sd_f sd_t (1 - r)
               amplitude term      phase term

The amplitude term is what having the wrong *amount* of variance costs; the
phase term is what having it at the wrong *time* costs. Note the phase term
scales with `sd_f`: if a model has no timing skill, adding variance makes it
worse. Everything below is against REA-L, on the same 3-6 h bandpassed interior
as section 2, with inits averaged before any ratio.

**The timing correlation is low for both models everywhere.**

| case | r Varda | r Multi | n stations Multi better |
|---|---|---|---|
| winter T_2M | 0.193 | 0.214 | 83/145 |
| winter TD_2M | 0.143 | 0.160 | 76/145 |
| winter SP_10M | 0.033 | 0.072 | 103/145 |
| winter PS | 0.023 | 0.161 | 139/145 |
| summer T_2M | 0.285 | 0.261 | 57/145 |
| summer TD_2M | 0.197 | 0.193 | 65/145 |
| summer SP_10M | 0.065 | 0.063 | 67/145 |
| summer PS | 0.097 | 0.087 | 61/145 |

`r` is pooled per station: covariance and both band variances are averaged
over inits first, and the ratio is formed after, as the error split below
requires. The plain mean of per-init correlations gives the same picture, within
0.035 of the pooled value in every case (`r_pooled_vs_per_init.py`). Single
forecasts scatter widely around that, with a per-init spread of about 0.23 at a
typical station.

**Part of this `r` is the sun (found 2026-09-28, section 15).** The daily cycle
has harmonics at 6, 4.8 and 4 h that fall into this band and are predictable for
free. With each source's mean daily cycle removed first, the weather-driven 3-6 h
timing correlation is 0.03-0.15 rather than 0.02-0.29 (summer T_2M: Varda 0.28
-> 0.15). The conclusions below hold either way, since both models lose alike.

At r = 0.2 a model explains about 4 % of the band variance. This is a larger
fact than any model-to-model difference and it sits underneath all of section 2:
the retained variance is mostly not the *right* variance.

**Varda's extra variance is not better timed.** In summer the two are tied, with
differences of 0.002-0.024 and station counts that are coin flips. In winter
multistep is better timed in TD_2M and PS under both pooled and per-init
correlation. For T_2M and SP_10M the winter difference depends on the method: it
flips for T_2M (per-init mean 0.207 Varda against 0.195 multistep) and nearly
vanishes for SP_10M (0.047 against 0.052). So nowhere is Varda's extra variance
robustly better timed, and the section 2 advantage is amplitude only. In every one of the eight cases Varda has the
smaller amplitude term and multistep the smaller phase term, which is the same
fact as "Varda has more variance" written as an error budget.

**Winter PS reverses the section 2 headline.** Varda's band correlation there is
0.023, indistinguishable from zero, while multistep reaches 0.161 and is better
timed at 139 of 145 stations. Varda produces a large amount of 3-6 h surface
pressure variance that is essentially uncorrelated with REA-L's. On that
variable the variance diagnostic is rewarding noise, so the 145/145 unanimity in
section 2 should not be read as multistep being worse at pressure.

**This explains why the class differences sit on the Plateau.** `r` is highest
on `flat` and lowest on `peaks`: winter T_2M runs 0.28-0.33 on flat against
0.06-0.10 on peaks, summer T_2M 0.33 against 0.20. The Plateau is the one place
where sub-6 h variability is predictable at all. Everywhere else there is
essentially no timing signal, so retaining more or less variance can neither
help nor hurt much, and the model-to-model differences collapse. The terrain
hypothesis (section 6) looked in the wrong place: complex terrain is where this
band is *least* predictable, not most. This also accounts for the decoupling
where multistep loses the most variance on `flat` yet its skill deficit is
smallest there, and for the same pattern on `peaks` in winter PS.

**Caveat, and why the argument does not rest on band MSE.** A model that outputs
a flat line has zero amplitude variance and scores a *better* band MSE than any
model with realistic variance, because the phase term vanishes with `sd_f`. Band
MSE therefore structurally favours the smoother model, and multistep is the
smoother model. The `r` comparison above is scale-free and carries no such bias,
which is why the claims here are made on `r` and not on `msss_band`.

**Where the overall MSE difference lives.** Multistep is clearly worse on
full-series MSE in winter, yet its band MSE is level with or ahead
of Varda's. Its overall disadvantage therefore sits at scales *outside* the
3-6 h band, most plausibly synoptic. Overall accuracy and band-scale variability
are separate findings here, and neither explains the other. **Confirmed
2026-09-28 (section 15):** on the synoptic component multistep's median skill
against Varda-single is -0.07 to -0.28 in six of eight cases (about 0 for summer
wind speed, +0.10 for summer PS), while at 3-6 h, 6-12 h and the daily cycle it
is within about +-0.10.

Script: `joint_variance_skill.py`. Table: `output/joint_variance_skill/`.
Figure: `output/joint_variance_skill/timing_vs_variance.png`, retained variance
against `r` per station, with a perfect model at (1, 1).

## 4. REA-L is itself heavily smoothed

REA-L holds only **43-66 %** of the observed 3-6 h variance at station points for
T_2M and wind speed, so deficits measured against REA-L understate the gap to
reality by roughly a factor of two. Dew point is the exception, with REA-L at
80-100 % of observed; station dew point is derived from temperature and humidity
rather than measured directly, which plausibly smooths the observation side.

Part of this is irreducible representativeness, since a point instrument is not a
1 km grid cell. But not all of it. In the wind-speed spectra the observed and
REA-L curves nearly coincide at 24 h and separate progressively toward shorter
periods, ending about 3x apart at 3 h. Pure footprint mismatch would give a
roughly constant ratio across periods; a ratio that grows toward short periods
indicates REA-L is genuinely losing sub-6 h variability.

Consequence: the target the models were trained on is already missing much of
what happened, so the models inherit a deficit before adding their own.

## 5. Retention is about twice as high in summer

For the near-surface fields, both models, consistently:

| | winter V / M | summer V / M | summer/winter |
|---|---|---|---|
| T_2M | 0.133 / 0.153 | 0.258 / 0.246 | 1.9 / 1.6 |
| TD_2M | 0.091 / 0.075 | 0.149 / 0.134 | 1.6 / 1.8 |
| SP_10M | 0.071 / 0.062 | 0.134 / 0.123 | 1.9 / 2.0 |
| PS | 0.176 / 0.109 | 0.218 / 0.113 | 1.2 / 1.0 |

The Varda advantage is also larger in summer for all four variables.

Two readings, not separated here: summer genuinely has more resolvable sub-6 h
variability (convection, a deeper boundary layer) and the models capture some of
it, or winter is dominated by stable, shallow, terrain-locked processes that a
1 km model cannot represent at all. **PS is the discriminator and it supports the
boundary-layer reading**: it is the one variable that is not a boundary-layer
quantity, and it is the one that shows almost no seasonal change.

## 6. Station class and terrain

Using the `T2m_based_class` classification from `spaziurat.lf` (144 of 145
stations matched). Median paired gap within each class, positive meaning Varda
retains more. `*` marks a 95 % init-bootstrap interval that excludes zero, taken
from the same resamples as the pooled interval of section 2. These intervals are
optimistic: stations are held fixed and are spatially correlated, and the smaller
the class the worse this gets. `strange` (n=11) is too small to support anything
and is left out; its numbers are in `output/gap_robustness/by_class*.csv`.

| class | n | w T_2M | w TD_2M | w SP_10M | w PS | s T_2M | s TD_2M | s SP_10M | s PS |
|---|---|---|---|---|---|---|---|---|---|
| flat | 30 | +0.037* | +0.018* | +0.015* | +0.048* | +0.045 | +0.075* | +0.049* | +0.124* |
| local | 60 | -0.016* | +0.015* | +0.004 | +0.073* | +0.009 | +0.011 | -0.001 | +0.101* |
| peaks | 25 | -0.008 | +0.010* | +0.003 | +0.157* | +0.008 | +0.014* | +0.005 | +0.115* |
| support | 18 | -0.008 | +0.015* | +0.007 | +0.049* | +0.030 | +0.034* | +0.011 | +0.099* |

**Varda retains more in 28 of the 32 class-case combinations, but only 18 have an
interval that excludes zero.** Thirteen include zero, so outside `flat` most
class-level differences are not resolved by this sample, and with the intervals
being optimistic even some of the 18 may not hold.

**One class result favours multistep: winter T_2M on `local`** (gap -0.016,
CI [-0.020, -0.009], Varda retains more at only 16 of 60 stations). `flat` goes
the other way (+0.037, CI [+0.019, +0.055]), so the pooled winter T_2M tie of
section 2 is two opposite class results cancelling, not an absence of
difference. Given the optimistic intervals, and that winter T_2M already flips
sign under band and window changes, this may well be an artefact. It is not
evidence of a general multistep advantage.

**`flat` has the largest gap in all six near-surface cases**, with an interval
excluding zero in five (summer T_2M just touches zero). In summer TD_2M and
SP_10M the `flat` interval does not overlap those of `local` and `peaks`. PS
behaves differently: in winter `peaks` has by far the largest gap (CI
[+0.123, +0.187], no overlap with `flat` or `support`) and `flat` one of the
smallest, while in summer all class intervals overlap, so there is no class
difference to explain there. So the Plateau gap is a near-surface phenomenon,
not a general one.
The terrain hypothesis predicted the interesting differences would sit in complex
terrain; for the boundary-layer variables they sit on the flat instead.
**Section 3 explains this**: the Plateau is the only place where the 3-6 h band
carries any timing skill at all, so it is the only place where retaining more or
less variance can make a difference. In complex terrain both models have
essentially no timing signal, and the model-to-model differences collapse with
it.

There is no Simpson's paradox: the class-weighted average of the winter T_2M gaps
is -0.0025 against a pooled -0.0030, so the pooled number is simply the class
average and station mix is hiding nothing.

Figures: `output/gap_robustness/paired_gap_class_{flat,local,peaks,support}.png`,
one per class, same style as `paired_gap_overview.png`, all eight cases on a
shared y axis.

**Representativeness dominates on summits.** Winter T_2M, fraction of observed
and of REA-L:

| class | n | REA-L vs obs | Varda vs REA-L | Multi vs REA-L |
|---|---|---|---|---|
| flat | 30 | 0.50 | 0.215 | 0.178 |
| local | 60 | 0.53 | 0.073 | 0.100 |
| peaks | 25 | 0.26 | 0.258 | 0.265 |
| strange | 11 | 0.44 | 0.058 | 0.109 |
| support | 18 | 0.41 | 0.149 | 0.158 |

`peaks` reverses between the two normalisations. Against observations it looks
like the worst class (REA-L at 0.26), but against REA-L the models do *better*
there than anywhere else. On summits the models reproduce their training target
well; it is REA-L that is far from the real station, which is what a 1 km cell
that cannot see a summit looks like.

`local` is where the models genuinely lose most, at 7-10 % of REA-L even
though REA-L tracks the observations reasonably there (`strange` looks similar
but is too small to rely on). That is model
deficiency, not a grid-resolution artefact.

**The terrain hypothesis is only partly supported.** The deficit is worse at high
elevation and where the model orography misses the station height, but REA-L
shows the same pattern and more strongly. Relative to REA-L the models' retention
is roughly flat with elevation, so most of the terrain dependence is inherited or
representativeness rather than an additional ML-specific failure in complex
terrain.

## 7. What the diagnostic can and cannot see

**Nyquist.** Hourly data puts the limit at a 2 h period. Convective cells and
gust fronts are invisible by construction. Headline claims are made for **3-6 h**;
the 2-3 h bins are too alias-contaminated to trust. The 3-8 h and 4-8 h variants
in section 2 are sensitivity checks on where the band edges are drawn, not
separate claims.

**Aliasing biases toward the hypothesis.** Sub-2 h variance folds back into the
resolved band, and does so more for a variable field than a smooth one, so it
inflates the truth relative to the models. Measured deficits are therefore upper
bounds.

**6-hourly sampling destroys everything below 12 h, not just at 6 h.** The
interpolated-forecaster reference is a floor, and the floor is lower than it
looks. Power retained after subsampling to 6 h and interpolating back:

| period | 24 h | 12 h | 8 h | 6 h | 4.8 h | 4 h | 3 h |
|---|---|---|---|---|---|---|---|
| retained | 69 % | 0 % | 0.85 % | 0 % | 0.13 % | 0 % | 0 % |

The nulls occur wherever the period divides 12 h evenly: a signal completing a
whole number of cycles between samples is sampled at the same phase every time
and vanishes. Sampling *at* 6 h is precisely what makes a 6 h *period* invisible.

Two consequences. The small humps the interpolation curve shows at 4-5 h are
interpolation artefact and aliased ghosts, not reconstructed signal, so "more
power" there would not mean "better". And comparing Varda-single against the
interpolation line *at exactly 6 h* flatters the downscaler, because the
reference sits at a mathematical null there; the band integral is the fair
measure, and by that measure the learned downscaler adds much less over plain
interpolation than the 6 h bin suggests (12.6 % against 11.5 % for winter T_2M).

**Single cases are uninterpretable.** A 48 h periodogram has about 2 degrees of
freedom per bin, so adjacent bins scatter by one to two orders of magnitude. In
20 randomly drawn station/init panels the aggregate ordering
(observations > REA-L > models) is not visible; several panels show the model
above the observations. All stratification must average before taking any ratio,
and ratios of single-case periodograms would be both wildly scattered and biased
upward.

## 8. How the numbers were estimated, and how much that matters

**Estimator.** A zero-phase Butterworth bandpass (order 4, second-order sections)
over 3-6 h, applied to the full 121 h record, with 24 h discarded at each end for
filter transients, then the variance of the interior. Verified against synthetic
data with known truth: inflation 1.0000, and 1e-13 attenuation at the diurnal
period. It needs no taper and no periodic extension.

**The detrend choice does not matter.** Eight variants were compared under a
fixed Hann taper: none, mean removal, least-squares line, weighted least-squares
lines at two weight exponents, the line through the two endpoints, a
`taper^2`-weighted line, and an unweighted quadratic. The largest spread in any
median was 6.8e-04 and in any model gap 8.2e-04, against gaps of 0.003 to 0.108.
A taper that goes to zero at both ends has already closed the wrap-around seam,
so nothing is left for a detrend to fix.

**The taper does matter, but Hann is converged.** Winter T_2M retention against
REA-L: boxcar 0.395, Tukey 0.25 0.217, Hann 0.154, Blackman 0.151. Leakage does
**not** cancel in the model/REA-L ratio, because the two have different amounts of
low-frequency energy to leak, so an untapered analysis would report 40 % rather
than 15 %. Blackman is a distinctly stronger taper and agrees with Hann to 0.3 %,
so the estimate sits on the flat part of the curve. The residual bias is known to
be upward, so reported retention is an upper bound and the measured deficit is if
anything understated.

**Multitaper was tried and rejected.** DPSS at NW=3 gives the best variance
reduction of anything tested (across-case scatter 0.80 to 0.54) but lands near the
boxcar value, because the record is too short: its smearing bandwidth is
`2*NW/N` = 0.125 cyc/h against a 3-6 h band only 0.167 cyc/h wide.

**Window length.** For the periodogram route, 48 h and 96 h agree; 24 h inflates
retention by about 1.8x and flips the winter T_2M sign. The bandpass is flat
across all three, which localises the effect to the estimator rather than the
forecasts.

**Two independent methods agree.** Hann periodogram and time-domain bandpass share
no assumptions and agree to within 10-25 %, against a factor 2.6 spread across
tapers.

**Net.** Direction is robust to every analysis choice tried. Absolute level is
not portable: quoting a retention figure requires naming the estimator, because a
defensible-sounding alternative turns 15 % into 40 %.

## 9. Data-quality findings

**REA-L has a 00 UTC artefact, but it is small and surface-only.** Detected as an
isolated one-hour spike in the RMS hourly increment (winter T_2M spike index
1.57). It appears only in near-surface fields and only in winter; PMSL, PS,
precipitation, T_850, QV_850 and FI_500 all show nothing. This **rules out latent
heat nudging**, which acts through the column, and fits a once-daily surface-side
update such as a soil or snow analysis. Measured effect on the 3-6 h band: +6.5 %
at grid points, +3.8 % at station points. Small enough that 48 h windows and the
6 h seam test both stand.

**The two models have different training splits**, extracted from the
checkpoints: Varda trains to 2023 with 2024 as validation, multistep trains
1985-2020 with 2021 validation and 2022 onward as test. The 2024 evaluation
period is out of training for both, but was seen for Varda's checkpoint
selection, which tilts slightly in Varda's favour. Varda also saw more REA-L
years (2005-2023 against 2005-2020).

**The global initial-condition datasets end 2024-12-31**, which forced the winter
sample to be January, February and December 2024 rather than a contiguous DJF.
Only the initial condition needs global data, since all forcings are computed
from the valid time, so late-December inits run into January 2025.

**Ten Varda-single runs contain all-NaN blocks** from the known GRIB
9999-geopotential collision documented in the `Data_for_Nadja` correction note. A
real geopotential value equal to the missing-value code is read as missing, and
the downscaler's spatial mixing spreads that single NaN across the whole field.
It is deterministic, so re-running does not help. Affected windows are dropped
from *all* sources so the samples stay matched.

**Precipitation units differ between sources**: REA-L stores metres, the
forecasts millimetres. Uncorrected this put the truth spectrum a factor 1e6 below
the models'.

**The ML models produce physically inconsistent dew points.** About 1.6 % of
forecast values have TD > T, median excess 0.28 K, maximum 2.3 K. Absent at step
0 and absent from REA-L, so it develops during the forecast. Expected for
independent output channels with no consistency constraint.

## 10. Method notes worth keeping

**Two-stage design.** Extracting point series from GRIB once (the expensive part,
~1.7 s per step) and analysing from the cache turned out to matter more than
anticipated: the cached dataset is ~96 MB, so the entire analysis survived a
three-day filesystem outage and ran from a laptop-sized copy in `$HOME`.

**Sampling.** 54 h init stride, 41 inits per season, 82 total. The stride must be
a multiple of 6 h (Varda's forecaster stride) but not of 24 h, or the init hour
never changes and lead time becomes identical to time of day. Note that 48 h over
two seasons (90 inits) costs *more* than 54 h over two seasons (82) and gives up
that coverage for nothing.

**The anchor-vs-in-between confound.** Varda's forecaster only accepts inits on
the 6-hourly grid, so anchor lead times always fall at 00/06/12/18 UTC whatever
stride is chosen. The control is the multistep model, which has no 6 h seam: any
6-hourly periodicity in *its* error curve is diurnal, and can be subtracted.

**`scipy.signal.butter` reads `Wn` as a fraction of Nyquist** unless `fs` is
passed explicitly. With hourly data, Nyquist is 0.5 cyc/h, so asking for
`[1/6, 1/3]` silently builds a 6-12 h bandpass rather than a 3-6 h one. It fails
quietly and plausibly: the models retain 6-12 h variability much better, so the
first run of the bandpass cross-check reported FRAC near 0.69 and flipped two of
the six gaps. Always check the transfer function at known periods before
believing a filter. Use `output="sos"` as well, since the transfer-function form
is numerically unstable for a bandpass at these orders.

**Observations.** The DWH returned 10-minute data despite an hourly increment
request. This removed a caveat rather than adding one: 10-minute averaging
attenuates the 3-6 h band by under 0.5 %, whereas hourly averaging would have
cost 9 % at 6 h and 60 % at 2 h. Coverage is 99.88 % direct values, 0.1-0.3 %
interpolated, nothing missing.

## 11. Scope: what was not analysed

**U_10M and V_10M** were dropped in favour of `SP_10M`. Wind speed is the
physically meaningful and natively observed quantity, speed is a nonlinear
function of the components so it must be formed on the series before any
transform, and the components add only directional information that this
diagnostic does not use.

**TOT_PREC1** was dropped because the diagnostic does not apply. Precipitation
spectra are dominated by intermittency, the on/off process, rather than by smooth
variability, so "3-6 h variance retention" does not measure the same thing there.
An earlier screening suggested precipitation was the one variable favouring
multistep; that claim is withdrawn rather than confirmed, since the measure
behind it is not sound for this field.

**Observations exist only for T_2M, TD_2M and SP_10M.** For `PS` only the
comparison against REA-L is available, so section 4's correction for REA-L's own
smoothing cannot be applied to it. That does not affect the model-to-model
comparison, which is what section 2 reports.

## 12. Open questions

- **Does the same picture hold against SwissMetNet?** Section 3 measures skill
  and timing against REA-L, so that both axes share a truth. But section 4 shows
  REA-L is itself heavily smoothed, so a model could be well timed against REA-L
  and poorly timed against reality. Repeating section 3 with observations as
  truth, for T_2M, TD_2M and SP_10M, is now the most important gap.
- **Why is the 3-6 h band so unpredictable?** r = 0.02-0.29 is the dominant
  finding of section 3 and it has no explanation yet. Whether this is the MSE
  objective, the 1 km resolution, the driving initial conditions, or an
  irreducible predictability limit at these scales is untested, and the four
  would imply very different things. **Partly answered by section 14 (new):**
  ICON-CH2 is not MSE-trained and has realistic variance, yet times the band no
  better. That argues against the MSE objective as the cause of the low `r` and
  for a predictability limit, though it does not rule out resolution or
  initial conditions.
- **Why does PS behave so differently?** It shows the largest and most unanimous
  variance difference, almost no seasonal dependence, a class pattern opposite to
  the near-surface fields, and a near-zero Varda timing correlation in winter. It
  is the only non-boundary-layer variable tested, which is probably the
  explanation, but that is untested.
- **Is REA-L instantaneous or time-averaged?** Unverified; `/store_new` was
  unavailable. The models are confirmed `instant` from GRIB metadata. This cannot
  explain the measured deficit either way, since hourly averaging costs under
  10 % at 6 h while the models are 85 % short.
- The 10-minute observations can measure how much variance lives below 2 h, which
  would turn the aliasing caveat from an assumption into a number.

## 13. Way forward

**Repeat section 3 against SwissMetNet.** Skill and timing are currently measured
against REA-L, chosen so that both axes share a truth with the variance work. But
section 4 shows REA-L holds only 43-66 % of the observed 3-6 h variance, so a
model can be well timed against REA-L and poorly timed against reality, and the
r = 0.02-0.29 result could be either understated or overstated. Running the same
decomposition with observations as truth is the single highest-value next step.
It covers T_2M, TD_2M and SP_10M; PS has no observations.

**Then attack the predictability question.** Why r is so low matters more than
the remaining model-to-model detail. A cheap first cut: compute the same r for
the interpolated-forecaster reference and for REA-L against observations. If
REA-L itself scores badly against stations in this band, the limit is upstream of
the ML models entirely.

**What no longer needs doing.** The variance-versus-skill pairing that earlier
versions of this section called for is done (section 3), as is the station-class
explanation it was meant to produce. The `flat` question is closed. Winter T_2M
does not need reopening: it fails a band change, a window change and a bootstrap,
and section 3 shows the band carries almost no timing signal anyway. Its class
split (section 6, `flat` against `local`) does not change that conclusion.

**Aggregation, for whoever runs the above.** The two sides were on different
footings and were reconciled by storing raw per-(season, param, init, station)
band variances, squared errors and cross-covariances, then forming every ratio
only after averaging over inits (section 7 explains why). Keep that discipline;
`joint_variance_skill.py` implements it.

### Conventions and tooling, for a fresh start

- **Compare paired, never unpaired.** Median over stations of
  `FRAC_Varda - FRAC_Multistep`. See section 2.
- **Prefer the time-domain bandpass** (`bandpass_timedomain.py`). Mind the
  `scipy.signal.butter` `fs=` trap documented in section 10.
- **Bootstrap inits, not stations.** Stations are spatially correlated.
- **Station classes** come from `T2m_based_class` in
  `resources/spaziurat.lf_2.6.1.tar.gz` (copy of `~/Documents/software/R/`), read with `rdata`, which is
  not a project dependency: run `uv run --with rdata ...`.
- **Never form a ratio on a single (init, station) case.** Section 7. Store raw
  variances and errors at full resolution, aggregate, then divide.
- **Report `r`, not band MSE, when judging timing.** Band MSE structurally
  favours the smoother model, because the phase term scales with the forecast's
  own standard deviation. See the caveat in section 3.
- **Scripts**: `joint_variance_skill.py` (the joint variance/skill/timing table,
  `--params`, `--seasons` and `--reuse` flags), `gap_robustness.py` (band edges,
  bootstrap, classes with class-level intervals, per-station gaps; `--params`
  and `--tag` flags, observations optional), `plot_paired_gap_by_class.py`,
  `bandpass_timedomain.py`,
  `window_length_sensitivity.py`, `taper_sensitivity.py`,
  `detrend_sensitivity.py`, `plot_paired_gap.py`, `skill_by_station_class.py`,
  `r_pooled_vs_per_init.py`, `plot_timing_vs_variance.py`,
  `rmse_by_station_init.py`.

## 14. NEW (2026-09-25, not yet reviewed): ICON-CH1 / ICON-CH2 baselines

Added unattended over the weekend of 2026-09-26; see `WEEKEND_NOTES.md` for how
the data were extracted and every decision made. Control run (member 000) of the
operational archives, same 82 inits, same 145 station cells (nearest ICON cell;
for CH1 identical to the REA-L cells), same estimators as sections 2-3. Adding
ICON-CH2 did not change the matched sample: every Multistep and Varda number
above is reproduced exactly.

**Caveat that applies to everything here.** Truth is REA-L, which is an
ICON-CH1-based reanalysis. At short lead ICON-CH1 surface pressure matches it to
31 Pa RMSE. ICON is flattered by this, CH1 far more than CH2.

**ICON-CH2 has realistic 3-6 h variance.** Median FRAC against REA-L:

| case | Varda | Multi | ICON-CH2 | CH2 vs obs | REA-L vs obs |
|---|---|---|---|---|---|
| winter T_2M | 0.133 | 0.153 | 0.805 | 0.355 | 0.426 |
| winter TD_2M | 0.091 | 0.075 | 0.843 | 0.680 | 0.801 |
| winter SP_10M | 0.071 | 0.062 | 0.607 | 0.280 | 0.446 |
| winter PS | 0.176 | 0.109 | 0.771 | - | - |
| summer T_2M | 0.258 | 0.245 | 1.091 | 0.584 | 0.545 |
| summer TD_2M | 0.149 | 0.134 | 0.996 | 1.008 | 1.024 |
| summer SP_10M | 0.134 | 0.123 | 0.750 | 0.500 | 0.667 |
| summer PS | 0.218 | 0.113 | 0.808 | - | - |

Against the stations ICON-CH2 is roughly as variable as REA-L itself: similar
in summer and for winter TD_2M, lower for winter T_2M and SP_10M (0.36 and 0.28
against 0.43 and 0.45). The paired
gap Varda minus CH2 is -0.54 to -0.83 with intervals far from zero, and CH2
retains more at 145 of 145 stations in every case (144 in summer PS). So the
section 1 deficit is specific to the ML models, not a property of km-scale
forecasting in general.

**But ICON-CH2 is no better timed.** Median pooled `r` against REA-L, lead 24-97 h:

| case | Varda | Multi | ICON-CH2 | stations CH2 > Varda |
|---|---|---|---|---|
| winter T_2M | 0.193 | 0.214 | 0.123 | 32/145 |
| winter TD_2M | 0.143 | 0.160 | 0.116 | 54/145 |
| winter SP_10M | 0.033 | 0.072 | 0.057 | 88/145 |
| winter PS | 0.023 | 0.161 | 0.011 | 60/145 |
| summer T_2M | 0.285 | 0.261 | 0.152 | 11/145 |
| summer TD_2M | 0.197 | 0.193 | 0.166 | 46/145 |
| summer SP_10M | 0.065 | 0.063 | 0.043 | 55/145 |
| summer PS | 0.097 | 0.087 | 0.042 | 36/145 |

A physics model that produces the right amount of 3-6 h variability places it
at the right time no better, and mostly worse, than the smoothed ML models. This
is the strongest evidence so far that the low `r` of section 3 is a
predictability limit at 1-4 days lead rather than an artefact of MSE training.
Two things weaken it. CH2 runs on a 2 km grid, and REA-L shares CH1's model
climate but not CH2's, so part of CH2's timing deficit may be grid and model
mismatch. The ML models were also trained on REA-L, so they may reproduce its
systematic, terrain-locked sub-daily patterns better than a different model
does.

**Skill follows from that.** Murphy skill against Varda-single (median over
(init, station) cells, full series, REA-L truth), positive means the model is better:

| case | Multi 0-120 h | CH2 0-120 h | Multi 0-33 h | CH2 0-33 h | CH1 0-33 h |
|---|---|---|---|---|---|
| winter T_2M | -0.20 | -0.61 | -0.10 | -1.47 | -0.55 |
| winter TD_2M | -0.16 | -0.45 | -0.05 | -0.93 | -0.51 |
| winter SP_10M | -0.08 | -0.25 | -0.06 | -0.40 | +0.11 |
| summer T_2M | -0.03 | -0.52 | -0.05 | -1.33 | -0.19 |
| summer TD_2M | -0.01 | -0.48 | +0.00 | -0.93 | -0.50 |
| summer SP_10M | +0.00 | -0.06 | -0.01 | -0.15 | +0.12 |

ICON-CH2 is clearly less accurate than Varda-single by MSE. That is what the
double penalty predicts for a model with realistic but poorly timed variance, and
it is the variance-skill tradeoff that section 3 did not find between the two ML
models. ICON-CH1 beats Varda only for SP_10M and PS at 0-33 h, where it
effectively shares REA-L's model climate. Even so it is worse for T_2M and TD_2M.
PS skill is left out of the table: CH2's surface pressure sits ~5 hPa below
REA-L's because of its coarser orography, which drives the skill score to -29 to
-207 and says nothing about forecast quality. The multistep medians here are per
cell, not the station-level MSSS of section 3, so they differ from that table.

Figures, all new, in `output/with_icon/`:
- `skill/skill_overview.png` and `skill/skill_overview_0-33h.png`
- `gap_robustness/paired_gap_overview.png` (one row per comparison)
- `spectra_bandpass/figs_FRAC/order4/all_seasons_params_FRAC_models.png` (with PS)
- `joint_variance_skill/timing_vs_variance.png` and its 8 single panels.

Scripts: `extract_points_icon.py`, then the usual scripts with `--extra-sources`
(driver: `output/icon_test/make_with_icon.sh`).

## 15. NEW (2026-09-28, not yet reviewed): other time scales

Question (Louis): 3-6 h looks almost like noise; do the two designs compare
differently at time scales that carry a more predictable signal? Four scales,
same stations, inits, truth (REA-L) and matched-sample rules as sections 2-3,
with ICON-CH2 added (section 14). All defined in `workflow/scripts/scales.py`:

| scale | how | lead window |
|---|---|---|
| 3-6 h | the original 4th-order bandpass | 24-96 h |
| 6-12 h | 2nd-order Butterworth bandpass (4th order would leave 31 h) | 26-94 h |
| daily cycle | least-squares fit of 24 h + 12 h harmonics | 24-96 h |
| synoptic | 24 h running mean applied twice (keeps ~2 days and longer) | 24-96 h |

A bandpass cannot isolate the daily cycle or longer in a 121 h record, because
its kernel is as long as the record, hence the other two methods. What each scale
passes by period: `output/with_icon/scales/scales_response.png`.

**The daily cycle contaminates every sub-daily band.** It is not a pure sine: its
harmonics at 12, 8, 6, 4.8, 4 h fall into the 3-6 h and 6-12 h bands, and they are
sun-driven, so predictable for free. Every scale was therefore also run with each
source's own mean daily cycle removed first (per station, season, hour of day):
the "anomaly" variant. It keeps only the weather-dependent part. It is the one
quoted below; the raw variant is in the `<scale>/` folders next to `<scale>_anom/`.

Ranges over the eight variable-season cases, anomaly variant (ML = Varda and
Multistep together):

| scale | FRAC, ML | FRAC, ICON-CH2 | r, ML | r, ICON-CH2 | r Multi - Varda |
|---|---|---|---|---|---|
| 3-6 h | 0.06-0.23 | 0.61-1.08 | 0.03-0.15 | 0.03-0.11 | -0.03..+0.06 |
| 6-12 h | 0.19-0.50 | 0.57-1.06 | 0.22-0.44 | 0.17-0.35 | -0.02..+0.06 |
| daily cycle | 0.51-1.10 | 0.85-1.33 | 0.55-0.88 | 0.53-0.85 | -0.03..+0.02 |
| synoptic | 0.75-1.17 | 0.76-1.17 | 0.69-0.98 | 0.69-0.97 | -0.03..+0.01 |

1. **The variance deficit is sub-daily.** It is severe at 3-6 h, substantial at
   6-12 h, and largely gone at the daily cycle and synoptic scales, where both ML
   models hold about REA-L's variance. ICON-CH2 holds 0.6-1.3 of it at every scale.
2. **Timing improves steeply with scale, for every model.** So Louis's point
   holds: longer scales carry signal the models can and do reproduce. Removing
   the mean daily cycle matters for the sub-daily bands (raw 6-12 h r is up to
   0.78, anomaly 0.44) and not at all for synoptic, as it should.
3. **Multistep and Varda-single time every scale alike** (r within 0.06). ICON-CH2
   is slightly below both in most sub-daily cases and equal at synoptic scales.
   Since ICON-CH2 is not REA-L's model, that small gap may be grid/model mismatch
   rather than worse forecasting.
4. **Retained variance, Varda minus Multistep, paired with 95 % intervals**
   (anomaly variant; V = Varda retains more, M = multistep retains more):

   | scale | V | M | CI includes 0 |
   |---|---|---|---|
   | 3-6 h | 6 | 0 | 2 |
   | 6-12 h | 2 | 0 | 6 |
   | daily cycle | 3 | 0 | 5 |
   | synoptic | 2 | 1 (winter T_2M) | 5 |

   The Varda lean of section 2 fades with scale, but it does not reverse.
   Multistep retains significantly more in one case out of 32 (synoptic winter
   T_2M, -0.10). Wind speed is the most consistent: Varda retains more at every
   scale in winter.
5. **Accuracy on each scale's component** (1 - MSE ratio vs Varda, median over
   (init, station) cells, no interval): multistep is within about +-0.10 of Varda
   at 3-6 h, 6-12 h and the daily cycle, but -0.07 to -0.28 at synoptic scales in
   six of eight cases (summer SP_10M -0.02, summer PS +0.10). This locates multistep's full-series
   MSE deficit of section 3 in the synoptic evolution, as suspected there.
   ICON-CH2 is less accurate than Varda on nearly every component (double penalty,
   REA-L truth).

**Caveats.** The synoptic component spans little more than one cycle per init in
a 73 h window, and detrending removes its straight-line part, so its numbers are
coarse. The mean daily cycle is estimated from the same 41 inits per season it is
removed from. Intervals exist only for the paired variance gap, and they are
optimistic (stations held fixed). Lead 24-96 h throughout; nothing here says
anything about day 1.

**Bottom line.** Looking at longer scales does not rescue the multistep
hypothesis. Where predictable signal exists, both designs capture it about
equally well in amount and timing. Where they differ, Varda-single holds
slightly more variance, and multistep is less accurate on the synoptic evolution.

Figures, per scale, under `output/with_icon/scales/<scale>[_anom]/`, mirroring
`output/with_icon/`: `skill/skill_overview.png` (skill on the component),
`gap_robustness/paired_gap_overview.png`,
`spectra_bandpass/figs_FRAC/all_seasons_params_FRAC_models.png`,
`joint_variance_skill/timing_vs_variance.png` + 8 panels.
Overview across scales: `output/with_icon/scales/scales_summary.png` and
`scales_summary_anom.png` (numbers in the matching CSVs).
