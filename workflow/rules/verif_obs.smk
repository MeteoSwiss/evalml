from pathlib import Path


rule collect_mec_input:
    input:
        inference_dir=rules.prepare_inference_forecaster.output.grib_out_dir,
    output:
        obs=directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/input_obs"),
        mod=directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/input_mod"),
    params:
        steps=lambda wc: RUN_CONFIGS[wc.run_id]["steps"],
    shell:
        """
        # create the input_obs and input_mod dirs
        mkdir -p {output.obs} {output.mod}

        # extract YYYYMM from init_time (which is YYYYMMDDHHMM) and use it in the paths
        init="{wildcards.init_time}"
        ym="${{init:0:6}}"
        lt="{params.lead_time}" 120h start, end, step
        # collect obs and mod files
        import config here?  
        cp /store_new/mch/msopr/osm/KENDA-1/EKF/${{ym}}/ekfSYNOP_${{init}}00.nc {output.obs}
        cat {input.inference_dir}/*.grib > {output.mod}/fc_${{init}}
        ls -l {output.mod}  {output.obs}
        """


rule generate_mec_namelist:
    localrule: True
    input:
        script="workflow/scripts/generate_mec_namelist.py",
        template="resources/mec/namelist.jinja2",
    output:
        namelist=OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/namelist",
    params:
        steps=lambda wc: RUN_CONFIGS[wc.run_id]["steps"],
    shell:
        """
        uv run {input.script} \
            --steps {params.steps} \
            --init_time {wildcards.init_time} \
            --template {input.template} \
            --namelist {output.namelist}
        """


rule run_mec:
    input:
        namelist=rules.generate_mec_namelist.output.namelist,
        run_dir=directory(rules.collect_mec_input.output.mod),
        mod_dir=directory(rules.collect_mec_input.output.mod),
    output:
        folder_to_delete=directory(
            OUT_ROOT / "data/runs/{run_id}/{init_time}/folder_to_delete"
        ),
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
