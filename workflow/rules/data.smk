from pathlib import Path


include: "common.smk"


if config["truth"]["root"].endswith("peakweather"):
    output_peakweather_root = config["truth"]["root"]
else:
    output_peakweather_root = OUT_ROOT / "data/observations/peakweather"


rule download_obs_from_peakweather:
    localrule: True
    output:
        root=directory(output_peakweather_root),
    run:
        from peakweather.dataset import PeakWeatherDataset

        # Download the data from Huggingface
        ds = PeakWeatherDataset(root=output.root)
