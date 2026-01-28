

def _get_forecaster_run_id(run_id):
    """Get the forecaster run ID from the RUN_CONFIGS."""
    return RUN_CONFIGS[run_id]["forecaster"]["run_id"]

rule nudge_analysis:
    # localrule: True
    input:
        script="workflow/scripts/inference_nudge_analysis.mo.py",
        peakweather_dir=rules.download_obs_from_peakweather.output.peakweather,
        forecasts=lambda wc: (
            [
                OUT_ROOT
                / f"logs/execute_inference/{_get_forecaster_run_id(wc.run_id)}-{wc.init_time}.ok"
            ]
        ),
    output:
        okfile=touch(OUT_ROOT / "logs/nudge_analysis/{run_id}-{init_time}.ok"),
    params:
        grib_file=lambda wc: (
            Path(OUT_ROOT) / f"data/runs/{_get_forecaster_run_id(wc.run_id)}/{wc.init_time}/grib/{wc.init_time}_000.grib"
        ).resolve(),
    resources:
        slurm_partition="postproc",
        cpus_per_task=6,
        runtime="20m",
    shell:
        """
        export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)
        python {input.script} --forecast {params.grib_file} --peakweather {input.peakweather_dir}
        # marimo edit {input.script} -- --forecast {params.grib_file} --peakweather {input.peakweather_dir}
        """
