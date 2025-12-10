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
