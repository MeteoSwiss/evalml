from pathlib import Path

rule generate_mec_namelist:
    input:
        template="resources/mec/namelist.jinja2"
    output:
        namelist=OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/namelist",
    run:
        """
        import jinja2

        context = {"init_time": wildcards.init_time}
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader({Path(input.template).parent})
        )
        template = env.get_template(input.template)
        namelist = template.render(**context)

        namelist_fn = Path(output.namelist)
        with namelist_fn.open("w+") as f:
            f.write(namelist)
        """

rule run_mec:
    input:
        grib_dir=OUT_ROOT / "data/runs/{run_id}/{init_time}/grib",
        ekf="path/to/ekf/file{init_time}",
        namelist=generate_mec_namelist.output.namelist
    output:
        feedback=OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/feedbacks/verSYNOP.nc
    # module: sarus?
    resources:
        cpus_per_task=1,
        runtime="1h",
    shell:
        """
        # some code to prepare the data
        # (or use a separate rule)
        # sarus command from Mary
        sarus pull ...
        """

rule rename_feedback:
    input:
        feedback=run_mec.output.feedback
    output:
        feedback=OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/feedback
    shell:


# rule ...