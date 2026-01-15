from pathlib import Path


include: "common.smk"


if "extract_cosmoe" in config.get("include_optional_rules", []):

    rule extract_cosmoe:
        input:
            archive=Path("/archive/mch/msopr/osm/COSMO-E"),
        output:
            fcts=protected(
                directory(Path("/store_new/mch/msopr/ml/COSMO-E/FCST{year}.zarr"))
            ),
        resources:
            cpus_per_task=4,
            runtime="24h",
        params:
            year_postfix=lambda wc: f"FCST{wc.year}",
            steps="0/120/6",
        log:
            OUT_ROOT / "logs/extract-cosmoe-fcts-{year}.log",
        shell:
            """
            python workflow/scripts/extract_baseline.py \
                --archive_dir {input.archive}/{params.year_postfix} \
                --output_store {output.fcts} \
                --steps {params.steps} \
                    > {log} 2>&1
            """


if "extract_cosmo1e" in config.get("include_optional_rules", []):

    rule extract_cosmo1e:
        input:
            archive=Path("/archive/mch/s83/osm/from_GPFS/COSMO-1E"),
        output:
            fcts=protected(
                directory(Path("/store_new/mch/msopr/ml/COSMO-1E/FCST{year}.zarr"))
            ),
        resources:
            cpus_per_task=4,
            runtime="24h",
        params:
            year_postfix=lambda wc: f"FCST{wc.year}",
            steps="0/33/1",
        log:
            OUT_ROOT / "logs/extract-cosmo1e-fcts-{year}.log",
        shell:
            """
            python workflow/scripts/extract_baseline.py \
                --archive_dir {input.archive}/{params.year_postfix} \
                --output_store {output.fcts} \
                --steps {params.steps} \
                    > {log} 2>&1
            """


if "extract_icon1" in config.get("include_optional_rules", []):

    rule extract_icon1:
        input:
            archive=Path("/store_new/mch/msopr/osm/ICON-CH1-EPS"),
        output:
            fcts=protected(
                directory(Path("/store_new/mch/msopr/ml/ICON-CH1-EPS/FCST{year}.zarr"))
            ),
        resources:
            cpus_per_task=4,
            runtime="24h",
        params:
            year_postfix=lambda wc: f"FCST{wc.year}",
            steps="0/33/1",
        log:
            OUT_ROOT / "logs/extract-icon1-fcts-{year}.log",
        shell:
            """
            python workflow/scripts/extract_baseline.py \
                --archive_dir {input.archive}/{params.year_postfix} \
                --output_store {output.fcts} \
                --steps {params.steps} \
                    > {log} 2>&1
            """
