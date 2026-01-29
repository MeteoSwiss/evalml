from pathlib import Path


include: "common.smk"


rule download_obs_from_peakweather:
    localrule: True
    output:
        peakweather=directory(OUT_ROOT / "data/observations/peakweather"),
    run:
        from peakweather.dataset import PeakWeatherDataset

        # Download the data from Huggingface
        ds = PeakWeatherDataset(root=output.peakweather)
