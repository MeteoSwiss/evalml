# ----------------------------------------------------- #
# Publication-grade figures workflow                    #
# ----------------------------------------------------- #
rule publication_figures:
    input:
        script="workflow/scripts/publication_figures.py",
        verif=EXPERIMENT_PARTICIPANTS.values(),
    output:
        report(
            directory("figures"),
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
