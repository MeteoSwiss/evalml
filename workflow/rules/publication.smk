# ----------------------------------------------------- #
# Publication-grade figures workflow                    #
# ----------------------------------------------------- #
rule publication_figures:
    input:
        script="workflow/scripts/publication_figures.py",
        verif=EXPERIMENT_PARTICIPANTS.values(),
    output:
        report(
            directory("figures/leadtime"),
            htmlindex="publication_figures.html",
        ),
    log:
        OUT_ROOT / "logs/figures/publication_figures.log",
    localrule: True
    params:
        labels=",".join([
            BASELINE_CONFIGS[k]["label"] if k in BASELINE_CONFIGS else RUN_CONFIGS[k]["label"]
            for k in EXPERIMENT_PARTICIPANTS.keys()
        ]),
    shell:
        """
        python {input.script} \
            --verif_files "{input.verif}" \
            --sources "{params.labels}" \
            --output {output} > {log} 2>&1
        """


def _meteogram_candidate_grib(wc):
    """GRIB dir of the (single) candidate run for the configured init_time."""
    run_id = list(CANDIDATES.keys())[0]
    init_time = config["publication"]["meteogram"]["init_time"]
    return str((Path(OUT_ROOT) / f"data/runs/{run_id}/{init_time}/grib").resolve())


def _meteogram_baselines():
    """Structured 'root|steps|member|label;...' for EPS-mean baselines."""
    specs = []
    for cfg in BASELINE_CONFIGS.values():
        if cfg.get("member") == "mean":
            specs.append(
                f"{cfg['root']}|{cfg['steps']}|{cfg['member']}|{cfg['label']}"
            )
    return ";".join(specs)


rule publication_meteogram:
    input:
        script="workflow/scripts/publication_meteogram.py",
        peakweather_dir=rules.data_download_obs_from_peakweather.output.root,
        eckit_grids=rules.data_download_eckit_geo_grids.output,
    output:
        report(
            directory("figures/meteogram"),
            htmlindex="publication_meteogram.html",
        ),
    log:
        OUT_ROOT / "logs/figures/publication_meteogram.log",
    localrule: True
    params:
        grib=_meteogram_candidate_grib,
        forecast_steps=lambda wc: RUN_CONFIGS[list(CANDIDATES.keys())[0]]["steps"],
        forecast_label=lambda wc: RUN_CONFIGS[list(CANDIDATES.keys())[0]]["label"],
        baselines=_meteogram_baselines(),
        date=config["publication"]["meteogram"]["init_time"],
        station=config["publication"]["meteogram"]["station"],
        params=",".join(config["publication"]["meteogram"]["params"]),
    shell:
        """
        set -euo pipefail
        export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)
        python {input.script} \
            --forecast {params.grib:q} \
            --forecast_steps {params.forecast_steps:q} \
            --forecast_label {params.forecast_label:q} \
            --baseline {params.baselines:q} \
            --peakweather {input.peakweather_dir:q} \
            --date {params.date:q} \
            --station {params.station:q} \
            --params {params.params:q} \
            --output {output:q} > {log} 2>&1
        """


rule publication_all:
    """Target rule for publication figures workflow."""
    input:
        (
            [rules.publication_figures.output, rules.publication_meteogram.output]
            if config.get("publication", {}).get("enabled", False)
            else []
        ),
