#!/bin/bash
# Script to generate GRIB templates for ICON-CH1 model data

# note: data at these paths might be remove in the future
ANA_SAMPLE=/store_new/mch/msopr/osm/KENDA-CH1/ANA25/det/iaf2025010100
FG_SAMPLE=/store_new/mch/msopr/osm/KENDA-CH1/FG25/det/lff2025010100

# template for precipitation
grib_copy -w shortName=TOT_PREC $FG_SAMPLE /dev/stdout | grib_set -d 0 - icon-ch1-shortName=TOT_PREC.grib

# template for typeOfLevel=heightAboveGround
grib_copy -w shortName=T_2M $ANA_SAMPLE /dev/stdout | grib_set -d 0 - icon-ch1-typeOfLevel=heightAboveGround.grib

# template for typeOfLevel=surface
grib_copy -w shortName=PS $ANA_SAMPLE /dev/stdout | grib_set -d 0 - icon-ch1-typeOfLevel=surface.grib

# template for typeOfLevel=isobaricInhPa
grib_copy -w shortName=T,level=50 $ANA_SAMPLE /dev/stdout | grib_set -d 0 -s typeOfLevel=isobaricInhPa - icon-ch1-typeOfLevel=isobaricInhPa.grib
