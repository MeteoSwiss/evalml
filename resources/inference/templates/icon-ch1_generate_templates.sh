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

#template for windgust
grib_set -s shortName=VMAX_10M,level=10 -d 0 icon-ch1-typeOfLevel=heightAboveGround.grib icon-ch1-shortName=VMAX_10M.grib

# template for CAPE_MU (instantaneous, atmMU level type)
# NOTE: system grib_set segfaults with COSMO definitions — generate via Python eccodes:
#   python3 -c "
#   import eccodes, os; from eccodes_cosmo_resources.path import get_definitions_path
#   os.environ['ECCODES_DEFINITION_PATH'] = str(get_definitions_path())
#   with open('icon-ch1-typeOfLevel=surface.grib','rb') as f: h = eccodes.codes_grib_new_from_file(f)
#   eccodes.codes_set(h,'shortName','CAPE_MU'); eccodes.codes_set(h,'centre','lssw')
#   eccodes.codes_set_values(h,[0.0]*eccodes.codes_get_size(h,'values'))
#   with open('icon-ch1-shortName=CAPE_MU.grib','wb') as f: eccodes.codes_write(h,f)
#   eccodes.codes_release(h)"
