#!/bin/bash
# Script to generate GRIB templates for ICON-CH1 model data

# note: data at these paths might be remove in the future
SFC_SAMPLE=/store_new/mch/msopr/osm/ICON-CH1-EPS/FCST25/25010100_638/grib/i1eff00000000_000
PL_SAMPLE=/store_new/mch/msopr/osm/ICON-CH1-EPS/FCST25/25010100_638/grib/i1eff00000000_000p

# template for precipitation
grib_copy -w shortName=TOT_PREC $SFC_SAMPLE /dev/stdout | grib_set -d 0 - icon-ch1-shortName=TOT_PREC.grib

# template for typeOfLevel=heightAboveGround
grib_copy -w shortName=T_2M $SFC_SAMPLE /dev/stdout | grib_set -d 0 - icon-ch1-typeOfLevel=heightAboveGround.grib

# template for typeOfLevel=surface
grib_copy -w shortName=PS $SFC_SAMPLE /dev/stdout | grib_set -d 0 - icon-ch1-typeOfLevel=surface.grib

# template for typeOfLevel=isobaricInhPa
grib_copy -w shortName=T,level=500 $PL_SAMPLE /dev/stdout | grib_set -d 0 - icon-ch1-typeOfLevel=isobaricInhPa.grib

# template for typeOfLevel=meanSea
grib_copy -w shortName=PMSL $SFC_SAMPLE /dev/stdout | grib_set -d 0 - icon-ch1-typeOfLevel=meanSea.grib

# template for VMAX_10M (max 10m wind speed) on the ICON-CH1 1km grid.
# Used by the realv2 output stream of the multi-output architecture. Derive it from
# the heightAboveGround template (which is in HOURS) and retarget to VMAX_10M @ 10m:
# setting shortName=VMAX_10M makes eccodes pick the max stepType automatically, while
# keeping stepUnits=hours. Extracting straight from the ICON source instead yields a
# minute-unit step (stepUnits=0), which mislabels the 6 h max window as 6 minutes.
grib_set -s shortName=VMAX_10M,level=10 -d 0 icon-ch1-typeOfLevel=heightAboveGround.grib icon-ch1-shortName=VMAX_10M.grib
