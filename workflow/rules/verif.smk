# ----------------------------------------------------- #
# VERIFICATION WORKFLOW                                 #
# ----------------------------------------------------- #


import pandas as pd


START = pd.to_datetime(config["init_times"]["start"])
END = pd.to_datetime(config["init_times"]["end"])
FREQ = pd.to_timedelta(config["init_times"]["frequency"])
INIT_TIMES = pd.date_range(START, END, freq=FREQ, inclusive="left").strftime(
    "%Y%m%d%H00"
)


rule run_verif:
    input:
        expand(
            "resources/inference/{run_id}/output/{init_time}/raw",
            init_time=INIT_TIMES,
            allow_missing=True,
        ),
    output:
        "results/eval_report_{run_id}.html",
    shell:
        ""
