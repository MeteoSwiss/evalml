from pathlib import Path

rule collect_mec_input:
    input:
        inference_dir=rules.prepare_inference_forecaster.output.grib_out_dir
    output:
        obs=directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/input_obs"),
        mod=directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/input_mod"),
    shell:
        """
        # create the input_obs and input_mod dirs
        mkdir -p {output.obs} {output.mod}

        # extract YYYYMM from init_time (which is YYYYMMDDHHMM) and use it in the paths
        init="{wildcards.init_time}"
        ym="${{init:0:6}}"

        # collect obs and mod files
        cp /store_new/mch/msopr/osm/KENDA-1/EKF/${{ym}}/ekfSYNOP_${{init}}00.nc {output.obs}
        cat {input.inference_dir}/*.grib > {output.mod}/fc_${{init}}
        ls -l {output.mod}  {output.obs}
        """

rule generate_mec_namelist:
    input:
        template="resources/mec/namelist.jinja2"
    output:
        namelist=OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/namelist",
    run:
        import jinja2
        import re

        # Construct the leadtimes list for MEC namelist from config steps
        steps_str = None
        cfg_runs = config.get("runs", []) if config else []
        first = cfg_runs[0] if cfg_runs else {}
        forecaster = first.get("forecaster") if isinstance(first, dict) else None
        steps_str = forecaster.get("steps") if isinstance(forecaster, dict) else None

        # Parse steps: start/stop/step (hours). Example: "0/120/6"
        m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*$", str(steps_str))
        if not m:
            raise ValueError(f"Invalid steps format: {steps_str}. Expected 'start/stop/step' in hours")
        start_h, stop_h, step_h = map(int, m.groups())

        # Include stop_h (inclusive). Produce strings like 0000,0600,1200,...,12000
        lead_hours = range(start_h, stop_h + 1, step_h)
        leadtimes = ",".join(f"{h:02d}00" for h in lead_hours)

        # Render template with init_time and computed leadtimes
        context = {"init_time": wildcards.init_time, "leadtimes": leadtimes}
        template_path = Path(input.template)
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(template_path.parent)))
        template = env.get_template(template_path.name)
        namelist = template.render(**context)
        print(f"MEC namelist created: \n{namelist}")
        
        out_path = Path(str(output.namelist))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            f.write(namelist)

rule run_mec:
    input:
        namelist=rules.generate_mec_namelist.output.namelist,
        run_dir=directory(rules.collect_mec_input.output.mod),
        mod_dir=directory(rules.collect_mec_input.output.mod),

    output:
        folder_to_delete=directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/folder_to_delete")
    resources:
        cpus_per_task=1,
        runtime="1h",
    shell:
        #TODO(mmcglohon): Replace podman with sarus if needed.
        """
        echo 'would run mec on namelist:'
        cat {input.namelist}
        ls -l {input.run_dir}
        mkdir -p {output.folder_to_delete}
        # Note: pull command currently redundant; may not be the case with sarus.
        #podman pull container-registry.meteoswiss.ch/mecctr/mec-container:0.1.0-main
        #srun --pty -N1 -c 11 -p postproc -t 2:00:00 podman run --mount=type=bind,source={{input.run_dir}},destination=/src/bin2 --mount=type=bind,source=/oprusers/osm/opr.emme/data/,destination=/oprusers/osm/opr.emme/data/ container-registry.meteoswiss.ch/mecctr/mec-container:0.1.0-main
        #ls -l {{output}}
        """