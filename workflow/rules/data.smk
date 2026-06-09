from pathlib import Path


include: "common.smk"


if config["truth"]["root"].endswith("peakweather"):
    output_peakweather_root = config["truth"]["root"]
else:
    output_peakweather_root = OUT_ROOT / "data/observations/peakweather"


rule data_download_obs_from_peakweather:
    output:
        root=directory(output_peakweather_root),
    localrule: True
    run:
        from peakweather.dataset import PeakWeatherDataset

        # Download the data from Huggingface
        ds = PeakWeatherDataset(root=output.root)


# Grid-definition files required by earthkit/eckit to decode the ICON-CH grids.
# Automatic download/caching of these is currently broken (see README), so we
# fetch them into eckit's default geo grid cache under $HOME, where it finds
# them without any ECKIT_GEO_CACHE wiring. The output file names are the cache
# keys eckit computes for each grid.
ECKIT_GEO_GRID_DIR = Path.home() / ".local/share/eckit/geo/grid/icon"
ECKIT_ICON_CH1_URL = (
    "https://sites.ecmwf.int/repository/eckit/geo/grid/icon-ch/icon-ch1-c.ek"
)
ECKIT_ICON_CH2_URL = (
    "https://sites.ecmwf.int/repository/eckit/geo/grid/icon-ch/icon-ch2-c.ek"
)


rule data_download_eckit_geo_grids:
    output:
        ch1=ECKIT_GEO_GRID_DIR
        / "17643da2574959b644d254a3cd6e2bc0-b0699f374c63d05028c18c12f80a48f4.ek",
        ch2=ECKIT_GEO_GRID_DIR
        / "bbbd5a09855499243c7a4aa4c8762920-67adabf5c0cff041ebaafa61a3bda267.ek",
    log:
        OUT_ROOT / "logs/data_download_eckit_geo_grids/download.log",
    localrule: True
    shell:
        """
        (
            set -euo pipefail
            curl -fL {ECKIT_ICON_CH1_URL:q} -o {output.ch1:q}
            curl -fL {ECKIT_ICON_CH2_URL:q} -o {output.ch2:q}
        ) >{log} 2>&1
        """
