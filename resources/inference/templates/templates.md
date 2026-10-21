# GRIB Template Notes

## VARDA templates (`varda-*.grib`)

Created for VARDA-SINGLE (generatingProcessIdentifier=180), a deterministic AI forecast model.
Derived from the `icon-ch1-*.grib` templates with the following changes:

| Key | icon-ch1 value | varda value | Reason |
|-----|---------------|-------------|--------|
| `productDefinitionTemplateNumber` | 1 (ensemble instantaneous) / 11 (ensemble accumulation) | 0 (deterministic instantaneous) / 8 (deterministic accumulation) | VARDA-SINGLE is deterministic, not an ensemble |
| `typeOfGeneratingProcess` | 4 (ensemble forecast) | 2 (forecast) | Same reason |
| `typeOfProcessedData` | 5 (control forecast member) | 1 (forecast products) | Same reason |

### Files

| File | levtype | PDT | Notes |
|------|---------|-----|-------|
| `varda-typeOfLevel=isobaricInhPa.grib` | pl | 0 | Pressure level instantaneous fields |
| `varda-typeOfLevel=surface.grib` | sfc | 0 | Surface instantaneous fields (FR_LAND, PS, FIS, SKT, SSO_SIGMA, SSO_STDH, CLCT) |
| `varda-typeOfLevel=heightAboveGround.grib` | sfc | 0 | Height-above-ground fields (T_2M, TD_2M, U_10M, V_10M) |
| `varda-typeOfLevel=meanSea.grib` | sfc | 0 | Mean sea level fields (PMSL) |
| `varda-shortName=TOT_PREC.grib` | sfc | 8 | Total precipitation (temporal accumulation) |
| `varda-shortName=CLCH.grib` | pl | 0 | High cloud cover |
| `varda-shortName=CLCM.grib` | pl | 0 | Medium cloud cover |
| `varda-shortName=CLCL.grib` | unknown | 0 | Low cloud cover |
