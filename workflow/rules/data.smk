from pathlib import Path


include: "common.smk"


# Grid-definition files required by earthkit/eckit to decode the ICON-CH grids.
# eckit downloads the raw ATLAS-IO definitions from ECMWF on first use and
# writes a processed eckit::codec-format cache under ~/.local/share/eckit/geo/.
# We trigger that on-demand processing via a SLURM job (not localrule) so that
# it runs on a compute node with outbound internet access.  Dependent jobs
# declare the cache files as inputs so Snakemake serialises the cache-warming
# step before any parallel GRIB readers run.
#
# IMPORTANT: do NOT curl the raw .ek files directly into the cache directory.
# The downloaded files are ATLAS-IO format; eckit's cache reader expects its
# own eckit::codec format.  Placing the raw file there causes the error:
#   eckit::codec::InvalidRecord: version not found in record <cache-file>
ECKIT_GEO_GRID_DIR = Path.home() / ".local/share/eckit/geo/grid/icon"


rule data_download_eckit_geo_grids:
    output:
        ch1=ECKIT_GEO_GRID_DIR
        / "17643da2574959b644d254a3cd6e2bc0-b0699f374c63d05028c18c12f80a48f4.ek",
        ch2=ECKIT_GEO_GRID_DIR
        / "bbbd5a09855499243c7a4aa4c8762920-67adabf5c0cff041ebaafa61a3bda267.ek",
    log:
        OUT_ROOT / "logs/data_download_eckit_geo_grids/download.log",
    resources:
        slurm_partition="postproc",
        cpus_per_task=1,
        mem_mb_per_cpu=1800,
        runtime="15m",
    shell:
        """
        (
            set -euo pipefail
            # Let eckit download the raw grid definitions itself and write the
            # processed eckit::codec cache.  This is the only way to get the
            # cache in the format eckit::codec::RecordReader expects.
            # Must run on a compute node (not localrule) because login nodes
            # do not have outbound internet access for eckit's grid download.
            uv run python - <<'EOF'
import eckit.geo
eckit.geo.Grid("ICON-CH1")
eckit.geo.Grid("ICON-CH2")
print("eckit grid caches generated")
EOF
        ) >{log} 2>&1
        """
