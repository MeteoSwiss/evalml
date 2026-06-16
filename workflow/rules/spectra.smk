# ----------------------------------------------------- #
# POWER SPECTRA WORKFLOW                                 #
# ----------------------------------------------------- #


include: "common.smk"


SPECTRA = config["experiment"].get("spectra", {})
SPECTRA_LEAD_TIMES = ",".join(str(s) for s in SPECTRA.get("lead_times", []))
SPECTRA_VARIABLES = ",".join(SPECTRA.get("variables", ["T_2M", "WIND_KE", "TOT_PREC"]))
SPECTRA_METHOD = SPECTRA.get("method", "dct")


def _spectra_reftimes():
    hours = SPECTRA.get("init_hours")
    if hours is None:
        return [t.strftime("%Y%m%d%H%M") for t in REFTIMES]
    return [t.strftime("%Y%m%d%H%M") for t in REFTIMES if t.hour in hours]


def _spectra_truth_steps():
    candidates = list(collect_all_candidates())
    if candidates:
        return RUN_CONFIGS[candidates[0]]["steps"]
    # no candidate runs (baseline-only experiment): fall back to any baseline's steps
    if BASELINE_CONFIGS:
        return next(iter(BASELINE_CONFIGS.values()))["steps"]
    return "0/120/6"


def spectra_participants():
    """participant key -> aggregated spectra path (runs + baselines)."""
    out = {}
    for base in BASELINE_CONFIGS:
        out[base] = OUT_ROOT / f"data/baselines/{base}/spectra_aggregated.nc"
    for run_id, cfg in RUN_CONFIGS.items():
        if cfg.get("_is_candidate", False):
            out[run_id] = OUT_ROOT / f"data/runs/{run_id}/spectra_aggregated.nc"
    return out


SPECTRA_PARTICIPANTS = spectra_participants()


rule spectra_compute:
    input:
        "src/spectra/__init__.py",
        "src/data_input/__init__.py",
        script="workflow/scripts/spectra_compute.py",
        inference_okfile=rules.inference_execute.output.okfile,
        eckit_grids=rules.data_download_eckit_geo_grids.output,
    output:
        OUT_ROOT / "data/runs/{run_id}/{init_time}/spectra.nc",
    log:
        OUT_ROOT / "logs/spectra_compute/{run_id}-{init_time}.log",
    wildcard_constraints:
        run_id=r"[^/]+/[^/]+",
        init_time=r"\d+",
    resources:
        cpus_per_task=8,
        mem_mb=50_000,
        runtime="30m",
    params:
        grib=lambda wc: (
            Path(OUT_ROOT) / f"data/runs/{wc.run_id}/{wc.init_time}/grib"
        ).resolve(),
        steps=lambda wc: RUN_CONFIGS[wc.run_id]["steps"],
        label=lambda wc: RUN_CONFIGS[wc.run_id].get("label", wc.run_id),
    shell:
        """
        export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)
        uv run {input.script} \
            --forecast {params.grib} --reftime {wildcards.init_time} \
            --steps "{params.steps}" --lead_times "{SPECTRA_LEAD_TIMES}" \
            --variables "{SPECTRA_VARIABLES}" --method "{SPECTRA_METHOD}" \
            --label "{params.label}" --output {output} >{log} 2>&1
        """


rule spectra_compute_baseline:
    input:
        "src/spectra/__init__.py",
        "src/data_input/__init__.py",
        script="workflow/scripts/spectra_compute.py",
        forecast=lambda wc: BASELINE_CONFIGS[wc.baseline_id]["root"],
        eckit_grids=rules.data_download_eckit_geo_grids.output,
    output:
        OUT_ROOT / "data/baselines/{baseline_id}/{init_time}/spectra.nc",
    log:
        OUT_ROOT / "logs/spectra_compute_baseline/{baseline_id}-{init_time}.log",
    resources:
        cpus_per_task=8,
        mem_mb=50_000,
        runtime="30m",
    params:
        steps=lambda wc: BASELINE_CONFIGS[wc.baseline_id]["steps"],
        label=lambda wc: BASELINE_CONFIGS[wc.baseline_id].get("label", wc.baseline_id),
    shell:
        """
        export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)
        uv run {input.script} \
            --forecast {input.forecast} --reftime {wildcards.init_time} \
            --steps "{params.steps}" --lead_times "{SPECTRA_LEAD_TIMES}" \
            --variables "{SPECTRA_VARIABLES}" --method "{SPECTRA_METHOD}" \
            --label "{params.label}" --output {output} >{log} 2>&1
        """


rule spectra_compute_truth:
    input:
        "src/spectra/__init__.py",
        "src/data_input/__init__.py",
        script="workflow/scripts/spectra_compute.py",
        truth=config["truth"]["root"],
        eckit_grids=rules.data_download_eckit_geo_grids.output,
    output:
        OUT_ROOT / "data/truth/{init_time}/spectra.nc",
    log:
        OUT_ROOT / "logs/spectra_truth/{init_time}.log",
    resources:
        cpus_per_task=8,
        mem_mb=50_000,
        runtime="30m",
    params:
        steps=_spectra_truth_steps(),
        label=config["truth"]["label"],
    shell:
        """
        export ECCODES_DEFINITION_PATH=$(realpath .venv/share/eccodes-cosmo-resources/definitions)
        uv run {input.script} \
            --truth {input.truth} --reftime {wildcards.init_time} \
            --steps "{params.steps}" --lead_times "{SPECTRA_LEAD_TIMES}" \
            --variables "{SPECTRA_VARIABLES}" --method "{SPECTRA_METHOD}" \
            --label "{params.label}" --output {output} >{log} 2>&1
        """


rule spectra_aggregate:
    input:
        script="workflow/scripts/spectra_aggregate.py",
        spectra_nc=lambda wc: expand(
            rules.spectra_compute.output,
            init_time=_spectra_reftimes(),
            allow_missing=True,
        ),
    output:
        OUT_ROOT / "data/runs/{run_id}/spectra_aggregated.nc",
    log:
        OUT_ROOT / "logs/spectra_aggregate/{run_id}.log",
    wildcard_constraints:
        run_id=r"[^/]+/[^/]+",
    localrule: True
    resources:
        cpus_per_task=2,
        mem_mb=20_000,
        runtime="20m",
    shell:
        "uv run {input.script} {input.spectra_nc} --output {output} >{log} 2>&1"


use rule spectra_aggregate as spectra_aggregate_baseline with:
    input:
        script="workflow/scripts/spectra_aggregate.py",
        spectra_nc=lambda wc: expand(
            rules.spectra_compute_baseline.output,
            init_time=_spectra_reftimes(),
            allow_missing=True,
        ),
    output:
        OUT_ROOT / "data/baselines/{baseline_id}/spectra_aggregated.nc",
    log:
        OUT_ROOT / "logs/spectra_aggregate_baseline/{baseline_id}.log",
    localrule: True


use rule spectra_aggregate as spectra_aggregate_truth with:
    input:
        script="workflow/scripts/spectra_aggregate.py",
        spectra_nc=expand(
            rules.spectra_compute_truth.output,
            init_time=_spectra_reftimes(),
        ),
    output:
        OUT_ROOT / "data/truth/spectra_aggregated.nc",
    log:
        OUT_ROOT / "logs/spectra_aggregate_truth/truth.log",
    localrule: True


rule spectra_plot:
    input:
        script="workflow/scripts/spectra_plot.py",
        truth=OUT_ROOT / "data/truth/spectra_aggregated.nc",
        participants=list(SPECTRA_PARTICIPANTS.values()),
    output:
        report(
            directory(OUT_ROOT / "results/{experiment}/spectra"),
            patterns=["{name}.png"],
        ),
    log:
        OUT_ROOT / "logs/spectra_plot/{experiment}.log",
    resources:
        cpus_per_task=2,
        mem_mb=20_000,
        runtime="20m",
    shell:
        """
        uv run {input.script} \
            --truth {input.truth} --participants {input.participants} \
            --variables "{SPECTRA_VARIABLES}" --lead_times "{SPECTRA_LEAD_TIMES}" \
            --output_dir {output} >{log} 2>&1
        """
