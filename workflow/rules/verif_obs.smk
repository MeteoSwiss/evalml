from pathlib import Path

rule generate_observation_data:
    input:
        testcase_dir="/scratch/mch/mmcgloho/MEC/2020102800",
    output:
        input_obs=directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/input_obs"),
        input_mod=directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/input_mod"),
        parent=directory(OUT_ROOT / "data/runs/{run_id}/{init_time}/mec"),
    shell:
        """
        cp -r {input.testcase_dir}/input_obs {output.parent}/
        cp -r {input.testcase_dir}/input_mod {output.parent}/
        ls {output.parent}
        # TODO: Some data still seems to be missing.
        """

rule generate_mec_namelist:
    input:
        template="resources/mec/namelist.jinja2"
    output:
        #namelist=OUT_ROOT / "data/runs/mec/namelist",
        # TODO: get wildcards working.
        namelist=OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/namelist",
    run:
        import jinja2
        # TODO: get wildcards working.
        context = {"init_time": wildcards.init_time}
        template_path = Path(input.template)
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader({template_path.parent})
        )
        template = env.get_template(template_path.name)
        namelist = template.render(**context)
        namelist_fn = Path(output.namelist)
        with namelist_fn.open("w+") as f:
            f.write(namelist)

rule run_mec:
    input:
        testcase_dir=directory(rules.generate_observation_data.output.parent),
        namelist=rules.generate_mec_namelist.output.namelist
    output:
        OUT_ROOT / "data/runs/{run_id}/{init_time}/mec/output/verSYNOP.nc"
    resources:
        cpus_per_task=1,
        runtime="1h",
    shell:
        #TODO(mmcglohon): Replace podman with sarus if needed.
        """
        echo 'running mec on namelist:'
        cat {input.namelist}
        ls {input.testcase_dir}
        # Note: pull command currently redundant; may not be the case with sarus.
        #podman pull container-registry.meteoswiss.ch/mecctr/mec-container:0.1.0-main
        srun --pty -N1 -c 11 -p postproc -t 2:00:00 podman run --mount=type=bind,source={input.testcase_dir},destination=/src/bin2 --mount=type=bind,source=/oprusers/osm/opr.emme/data/,destination=/oprusers/osm/opr.emme/data/ container-registry.meteoswiss.ch/mecctr/mec-container:0.1.0-main
        ls -l {output}
        """