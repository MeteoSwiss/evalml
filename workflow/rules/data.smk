from pathlib import Path


include: "common.smk"


if "extract_cosmoe_fcts" in config.get("include-optional-rules", []):

    rule extract_cosmoe_fcts:
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
            lead_time="0/126/6",
        log:
            OUT_ROOT / "logs/extract-cosmoe-fcts-{year}.log",
        shell:
            """
            python workflow/scripts/extract_cosmoe_fct.py \
                --archive_dir {input.archive}/{params.year_postfix} \
                --output_store {output.fcts} \
                --lead_time {params.lead_time} \
                    > {log} 2>&1
            """


rule generate_references:
    localrule: True
    input:
        OUT_ROOT / "data/runs/{run_id}/{init_time}/grib",
    output:
        OUT_ROOT / "data/runs/{run_id}/{init_time}/references.json",
    params:
        step="0/126/6",
    log:
        "logs/generate-references-{run_id}-{init_time}.log",
    shell:
        """
        uv run  workflow/scripts/vzarr_references.py generate \
                {input}/{wildcards.init_time}_{{step:03}}.grib step={params.step} \
                    --output {output} > {log} 2>&1
        """


rule combine_references:
    localrule: True
    input:
        expand(
            OUT_ROOT / "data/runs/{{run_id}}/{init_time}/references.json",
            init_time=[t.strftime("%Y%m%d%H%M") for t in REFTIMES],
        ),
    output:
        directory(OUT_ROOT / "data/runs/{run_id}/referenced.vzarr"),
    params:
        run_id_root=lambda wc: OUT_ROOT / f"data/runs/{wc.run_id}/",
        start=config["dates"]["start"],
        end=config["dates"]["end"],
        step=config["dates"]["frequency"][:-1],  # remove the trailing unit (e.g., '6h' -> '6')
    log:
        "logs/data_combine_references/{run_id}.log",
    shell:
        """
        uv run workflow/scripts/vzarr_references.py combine {params.run_id_root} \
            --output {output} --start {params.start} --end {params.end} --step {params.step} > {log} 2>&1
        """
