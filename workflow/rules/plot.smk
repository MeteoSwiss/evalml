# ----------------------------------------------------- #
# REPORTING WORKFLOW                                    #
# ----------------------------------------------------- #
from datetime import datetime


include: "common.smk"


rule plot_forecast:
#    localrule: False
    input:
#        grib_output=rules.map_init_time_to_inference_group.output[0],
        raw_output=rules.map_init_time_to_inference_group.output[1],
    output:
        directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/plots/"),
    params:
        sources=",".join(list(EXPERIMENT_PARTICIPANTS.keys())),
    log:
        OUT_ROOT / "logs/plot_forecast/{run_id}-{init_time}.log",
    shell:
        """
        python workflow/scripts/plot_map.py \
          --input {input.raw_output} \
        """
