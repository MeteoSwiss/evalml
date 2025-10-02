#!/bin/bash
# Script to generate GRIB templates for COSMO-1E analysis (KENDA-1)

# note: data at these paths might be remove in the future
ANA_SAMPLE=/store_new/mch/msopr/ml/KENDA-1/ANA24/laf2024010100
FG_SAMPLE=/store_new/mch/msopr/ml/KENDA-1/FG24/lff2024010100

# template for precipitation
grib_copy -w shortName=TOT_PREC $FG_SAMPLE /dev/stdout | grib_set -d 0 - co1e-shortName=TOT_PREC.grib

# template for typeOfLevel=heightAboveGround
grib_copy -w shortName=T_2M $ANA_SAMPLE /dev/stdout | grib_set -d 0 - co1e-typeOfLevel=heightAboveGround.grib

# template for typeOfLevel=surface
grib_copy -w shortName=PS $ANA_SAMPLE /dev/stdout | grib_set -d 0 - co1e-typeOfLevel=surface.grib

# template for typeOfLevel=isobaricInhPa
grib_copy -w shortName=T,level=500 $ANA_SAMPLE /dev/stdout | grib_set -d 0 - co1e-typeOfLevel=isobaricInhPa.grib


