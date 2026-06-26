# ----------------------------------------------------- #
# Publication-grade figures workflow                    #
# ----------------------------------------------------- #
rule publication_figures:
    input:
        "workflow/scripts/publication_style.py",
        "workflow/scripts/publication.mplstyle",
        script="workflow/scripts/publication_figures.py",
        verif=EXPERIMENT_PARTICIPANTS.values(),
    output:
        report(
            directory(OUT_ROOT / "figures/leadtime"),
            htmlindex="publication_figures.html",
        ),
    log:
        OUT_ROOT / "logs/figures/publication_figures.log",
    localrule: True
    params:
        labels=",".join(
            [
                (
                    BASELINE_CONFIGS[k]["label"]
                    if k in BASELINE_CONFIGS
                    else RUN_CONFIGS[k]["label"]
                )
                for k in EXPERIMENT_PARTICIPANTS.keys()
            ]
        ),
    shell:
        """
        python {input.script} \
            --verif_files "{input.verif}" \
            --sources "{params.labels}" \
            --output {output} >{log} 2>&1
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
            specs.append(f"{cfg['root']}|{cfg['steps']}|{cfg['member']}|{cfg['label']}")
    return ";".join(specs)


_PUB_SCOREMAP_CFG = config.get("publication", {}).get("scoremaps", {})


def _pub_scoremap_candidate_id():
    # RUN_CONFIGS is available at include time (defined in common.smk).
    # CANDIDATES (defined in Snakefile after all includes) filters to runs
    # with _is_candidate=True; replicate that filter here.
    for run_id, cfg in RUN_CONFIGS.items():
        if cfg.get("_is_candidate", False):
            return run_id
    raise ValueError("No candidate run found in RUN_CONFIGS")


def _pub_scoremap_baseline_id():
    label = _PUB_SCOREMAP_CFG.get("baseline_label", "ICON-CH1-CTRL")
    for bid, cfg in BASELINE_CONFIGS.items():
        if cfg.get("label") == label:
            return bid
    raise ValueError(f"No baseline found with label {label!r}")


def _pub_scoremap_leadtimes():
    """Return the lead times (hours) to plot.

    Uses publication.scoremaps.steps when set, otherwise falls back to
    experiment.scoremaps.leadtimes.
    """
    steps = _PUB_SCOREMAP_CFG.get("steps")
    if steps is not None:
        return [int(s) for s in steps]
    return list(config.get("experiment", {}).get("scoremaps", {}).get("leadtimes", [6, 24]))


def _pub_scoremap_inputs(wc):
    """Return named input files for publication_scoremaps (deferred via lambda).

    Files are ordered: all params for leadtime[0], then all params for leadtime[1], …
    so the script can slice them by n_params.
    """
    params = _PUB_SCOREMAP_CFG.get("params", ["T_2M", "SP_10M"])
    leadtimes = _pub_scoremap_leadtimes()
    cand_id = _pub_scoremap_candidate_id()
    base_id = _pub_scoremap_baseline_id()
    return {
        "cand_files": [
            str(OUT_ROOT / f"data/runs/{cand_id}/scoremaps/{p}_{lt}_{TRUTH_HASH}.nc")
            for lt in leadtimes
            for p in params
        ],
        "base_files": [
            str(
                OUT_ROOT
                / f"data/baselines/{base_id}/scoremaps/{p}_{lt}_{TRUTH_HASH}.nc"
            )
            for lt in leadtimes
            for p in params
        ],
    }


rule publication_scoremaps:
    input:
        unpack(_pub_scoremap_inputs),
        script="workflow/scripts/publication_scoremaps.py",
    output:
        report(
            directory(OUT_ROOT / "figures/scoremaps"),
            htmlindex="publication_scoremaps.html",
        ),
    log:
        OUT_ROOT / "logs/figures/publication_scoremaps.log",
    localrule: True
    params:
        candidate_label=lambda wc: RUN_CONFIGS[_pub_scoremap_candidate_id()].get(
            "label", "Varda-Single"
        ),
        baseline_label=_PUB_SCOREMAP_CFG.get("baseline_label", "ICON-CH1-CTRL"),
        leadtimes_str=" ".join(str(lt) for lt in _pub_scoremap_leadtimes()),
        season=_PUB_SCOREMAP_CFG.get("season", "all"),
        region=_PUB_SCOREMAP_CFG.get("region", "switzerland"),
        params_str=",".join(_PUB_SCOREMAP_CFG.get("params", ["T_2M", "SP_10M"])),
        scores_str=",".join(_PUB_SCOREMAP_CFG.get("scores", ["RMSE", "STDE"])),
    shell:
        """
        python {input.script} \
            --candidate_files {input.cand_files} \
            --baseline_files {input.base_files} \
            --params {params.params_str} \
            --scores {params.scores_str} \
            --candidate_label "{params.candidate_label}" \
            --baseline_label "{params.baseline_label}" \
            --leadtimes {params.leadtimes_str} \
            --season {params.season} \
            --region {params.region} \
            --output {output} >{log} 2>&1
        """


rule publication_meteogram:
    input:
        "workflow/scripts/publication_style.py",
        "workflow/scripts/publication.mplstyle",
        script="workflow/scripts/publication_meteogram.py",
        eckit_grids=rules.data_download_eckit_geo_grids.output,
    output:
        report(
            directory(OUT_ROOT / "figures/meteogram"),
            htmlindex="publication_meteogram.html",
        ),
    log:
        OUT_ROOT / "logs/figures/publication_meteogram.log",
    resources:
        slurm_partition="postproc",
        cpus_per_task=8,
        mem_mb=32000,
        runtime="60m",
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
            --date {params.date:q} \
            --station {params.station:q} \
            --params {params.params:q} \
            --output {output:q} >{log} 2>&1
        """
