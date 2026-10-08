# Evaluation strategy: Varda-single release candidate with cloud cover and skin temperature

Status: draft for team discussion (MRB-1023, part of MRB-984). Author: Louis Frey. Date: 2026-10-08.

## Question

Is the release candidate (RC) of the Varda-single forecaster, trained with cloud cover and skin temperature as additional prognostic variables, acceptable for operations?

The outcome will be a set of findings presented to the team. The decision is taken by the team in discussion. There is no fixed pass/fail rule.

Guiding principle (suggestion by Carlos): keep it simple. Do not introduce new verification methods just for clouds. Cloud cover is likely not well modelled anyway.

## What is evaluated

| Role | System | Notes |
|---|---|---|
| Candidate | Varda-single RC, stage 4 fine-tuning, run `07f642102d8a4da99680f187422170b6` (ECMWF MLflow, experiment `meteoswiss-varda-single-training-mrb-847`) | Checkpoint: `/scratch/mch/huppd/checkpoint/07f642102d8a4da99680f187422170b6`. Fine-tuned on ICON-CH1 analysis 2024-01 to 2025-03-31. |
| Scientific reference | Varda-single 1.0 | The RC should be comparable to it. A slight degradation is acceptable in exchange for gaining cloud cover and SKT. Not a baseline in the technical EvalML sense, it is run as a second candidate. |
| Baselines | ICON-CH1, ICON-CH2 | Baselines in both the technical and the scientific sense. |

New variables: total, low, medium and high cloud cover (CLCT, CLCL, CLCM, CLCH) and skin temperature (SKT, called T_G in ICON).

## Reference data ("truth")

**ICON-KENDA analysis** for all variables, including cloud cover and SKT.

> **Open point for the team:** KENDA cloud cover and skin temperature are model products, not observations. Scoring against them partly measures how close the RC is to ICON, not to reality. ICON baselines also have an advantage, since KENDA comes from the same model. Possible independent references for later: satellite cloud products (e.g. MSG cloud mask), SYNOP cloud cover (octas), satellite land surface temperature (LSA SAF). Do we accept KENDA for this release, or do we want at least one independent check?

## Test period

- April 2025 to March 2026 (full year). No overlap with the training period, which ends 2025-03-31.
- Initialisations roughly every second day. Exact init times and lead times to be decided.
- Why a full year: autumn and early winter are the main fog and low stratus season on the Swiss Plateau (hardest case for clouds and SKT). Spring brings snowmelt (SKT held near 0 °C over melting snow) and the start of convection. Summer brings convection.
- Known caveat: validation during training used the whole dataset (limited to 10 batches). This was also the case for Varda-single 1.0 and the team considers it unproblematic.

## Stage 1: basic evaluation

Goal: answer "is the RC comparable to Varda-single 1.0" (a slight degradation is acceptable in exchange for the new variables) with the standard EvalML tools.

1. **Degradation check on existing variables.** Headline scores (as for Varda-single 1.0) for T_2M, TD_2M, wind, TOT_PREC, PMSL etc. Adding new variables can degrade the old ones. For the go/no-go question this is probably the most important check.
2. **Headline scores for the new variables.** Bias, MAE and RMSE over lead time, for CLCT, CLCL, CLCM, CLCH and SKT.
3. **Distribution of cloud cover.** Cloud cover is mostly near 0 % or near 100 %. A model that predicts smooth, intermediate values everywhere can still get a reasonable RMSE. Therefore: histograms of forecast vs. KENDA values, and frequency bias at a few thresholds (e.g. clear < 20 %, overcast > 80 %).
4. **Showcase animations** of cloud cover (and SKT) for a few selected cases: a fog / low stratus episode, a frontal passage, a summer convection day.

Prerequisite: the TOT_PREC unit fix (precipitation in metres) must be merged into the branch used for the evaluation, otherwise precipitation scores and the stage 2 checks are wrong.

## Stage 2: physical consistency

Goal: check that clouds relate to other variables in a physically sensible way. In each check, the same quantity is computed for KENDA and for the ICON baselines. They are the yardstick, not a perfect value of zero.

1. **Clouds and precipitation.** How often is there precipitation above a threshold while total cloud cover is low? Compare RC, Varda-single 1.0 (precipitation only), ICON-CH1/CH2 and KENDA.
2. **Clouds and daily temperature cycle.** Daily range of SKT and T_2M under clear vs. cloudy skies. Clouds should damp the daily cycle.
3. **Clear-sky nights.** SKT minus T_2M at night under clear skies. It should be negative (the ground cools by radiation).

## Stage 3: additional metrics (optional)

Only if stages 1 and 2 do not give a clear picture, and only where implementation effort is small.

- **Fractions Skill Score (FSS)** on thresholded cloud masks (e.g. cloud cover > 50 %, > 80 %). Shows at which spatial scale the forecast becomes useful. Cheap if a generic implementation can be added to EvalML.
- **SAL, S and L components** for cloud cover. SAL exists partly in EvalML (for precipitation). Open question: cloud fields often cover most of the domain, so object identification needs care. The A component is just the domain-mean bias and adds little beyond stage 1.

## Open points for discussion

1. KENDA as the only reference (see above).
2. Init times and lead times.
3. Showcase cases: which dates?
4. Further ideas from Pirmin, Andreas and Julien Delbeke; literature search on cloud verification.
5. Whether stage 3 is needed at all.
