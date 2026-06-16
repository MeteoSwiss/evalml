# ----------------------------------------------------- #
# Publication-grade figures workflow                    #
# ----------------------------------------------------- #
rule publication_figures:
    input:
        script="figures/publication_figures.py",
        verif=EXPERIMENT_PARTICIPANTS.values(),
    output:
        report(
            directory("figures/{experiment}"),
            htmlindex="publication_figures.html",
        ),
    log:
        OUT_ROOT / "logs/figures/publication_figures_{experiment}.log",
    localrule: True
    params:
        sources=",".join(list(EXPERIMENT_PARTICIPANTS.keys())),
    shell:
        """
        python {input.script} \
            --verif_files {input.verif} \
            --sources {params.sources} \
            --output {output} >{log} 2>&1
        """