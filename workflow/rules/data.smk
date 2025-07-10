from pathlib import Path

include: "common.smk"

# rule extract_cosmoe_fcts:
#     input:
#         archive=Path("/archive/mch/msopr/osm/COSMO-E"),
#     output:
#         fcts=protected(
#             directory(Path("/scratch/mch/fzanetta/data/COSMO-E/FCST{year}.zarr"))
#         ),
#     resources:
#         cpus_per_task=4,
#         runtime="24h"
#     params:
#         year_postfix=lambda wc: f"FCST{wc.year}",
#         lead_time="0/126/6",
#     log:
#         OUT_ROOT / "logs/extract-cosmoe-fcts-{year}.log",
#     shell:
#         """
#         uv run --with earthkit-data --with xarray --with zarr --with eccodes-cosmo-resources-python \
#             python workflow/scripts/extract_cosmoe_fct.py \
#                 --archive_dir {input.archive}/{params.year_postfix} \
#                 --output_store {output.fcts} \
#                 --lead_time {params.lead_time} \
#                     > {log} 2>&1
#         """


# rule generate_references:
#     localrule: True
#     input:
#         OUT_ROOT / "{experiment}/{init_time}/grib",
#     output:
#         OUT_ROOT / "{experiment}/{init_time}/references.json",
#     params:
#         step="0/126/6",
#     log:
#         "logs/generate-references-{experiment}-{init_time}.log",
#     conda:
#         "../envs/anemoi_inference.yaml"
#     shell:
#         """
#         uv run --with kerchunk --with xarray --with eccodes --with meteodata-lab \
#             workflow/scripts/vzarr_references.py generate \
#                 {input}/{wildcards.init_time}_{{step:03}}.grib step={params.step} \
#                     --output {output} 2> {log}
#         """


# rule combine_references:
#     localrule: True
#     input:
#         expand(
#             OUT_ROOT / "{{experiment}}/{init_time}/references.json",
#             init_time=[t.strftime("%Y%m%d%H%M") for t in REFTIMES],
#         ),
#     output:
#         directory(OUT_ROOT / "{experiment}/output.vzarr"),
#     params:
#         experiment_root=lambda wc: OUT_ROOT / f"{wc.experiment}/",
#         start=config["init_times"]["start"],
#         end=config["init_times"]["end"],
#         step=config["init_times"]["frequency"][:-1],  # remove the trailing unit (e.g., '6h' -> '6')
#     log:
#         "logs/combine-references-{experiment}.log",
#     shell:
#         """
#         uv run --with kerchunk --with xarray --with eccodes --with meteodata-lab --with fastparquet \
#             workflow/scripts/vzarr_references.py combine {params.experiment_root} \
#             --output {output} --start {params.start} --end {params.end} --step {params.step} 2> {log}
#         """
