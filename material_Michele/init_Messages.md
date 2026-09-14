# Handover notes from Michele

## Goal

Compare **Varda's stage C** (the model trained without rollout, on REA-L-CH1 only)
against a **multistep model** trained in the same way (no rollout, REA-L-CH1 only).

The evalml config for this comparison is `multistep-stage-C-analysis.yaml`
(here: `multistep-stage-C-analysis_yaml.yaml`).

## Configs

### Multistep model

- Inference config: `sgm-multidataset-forecaster-global-ich1-1h(2).yaml`
- Patch metadata config: `sgm-multidataset-ich1-patch-1h(2).yaml`

### Varda forecaster (stage C)

- Inference config: `sgm-multidataset-forecaster-global-ich1.yaml`
- Patch metadata config: `sgm-multidataset-ich1-patch.yaml`

The only difference to what is already in evalml is that it uses **REA-L-CH1**
instead of operational analyses as input data.

### Varda temporal downscaler

Use the config already in evalml. It does not specify an input dataset, it just
reads the data produced by the forecaster.

## Models

| Role | Path / link |
| --- | --- |
| Varda stage C, forecaster | `/scratch/mch/apennino/output/checkpoint/a69e95db4b494679ab20440f8540a835/inference-last.ckpt` |
| Varda stage C, temporal downscaler | <https://service.meteoswiss.ch/mlstore#/models/sruc-m-2-interpolator/versions/3> |
| Multistep model (standalone) | `/scratch/mch/miccatta/anemoi-outputs/checkpoint_stage_C_multistep_in7_out6_BALFRIN/0eb1b8759c3c4ad188bc1e7cf8d0f1ab/inference-last.ckpt` |

## Note on the file names in this folder

The files ending in `-1h(2).yaml` are the multistep ones; the `(2)` is just a
download artefact. Their real names (as referenced by the analysis config) are
`sgm-multidataset-forecaster-global-ich1-1h.yaml` and
`sgm-multidataset-ich1-patch-1h.yaml`.

To install into evalml:

| Handover file | Destination |
| --- | --- |
| `sgm-multidataset-forecaster-global-ich1.yaml` | `resources/inference/configs/` |
| `sgm-multidataset-ich1-patch.yaml` | `resources/inference/metadata/` |
| `sgm-multidataset-forecaster-global-ich1-1h(2).yaml` | `resources/inference/configs/sgm-multidataset-forecaster-global-ich1-1h.yaml` |
| `sgm-multidataset-ich1-patch-1h(2).yaml` | `resources/inference/metadata/sgm-multidataset-ich1-patch-1h.yaml` |

## Open points

- With the above, a comparison can be done at the highlighted point.
- Where the rollout training happens is still to be clarified. Michele is back in
  a week and will follow up. In the meantime the non-rollout models are enough to
  start an analysis.
