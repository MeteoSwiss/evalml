# Model Evaluation Pipeline

[![Snakemake](https://img.shields.io/badge/snakemake-≥8.0.0-brightgreen.svg)](https://snakemake.github.io)
[![run with conda](http://img.shields.io/badge/run%20with-conda-3EB049?labelColor=000000&logo=anaconda)](https://docs.conda.io/en/latest/)

A Snakemake workflow for anemoi models verification and performance auditing.

## Before you start

This project uses `conda`, download and install it through
[miniforge](https://github.com/conda-forge/miniforge) with:

    curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash Miniforge3-Linux-x86_64.sh

## Installation

Clone the `mch-anemoi-evaluation` repository and navigate to the project root directory:

    git clone git@github.com:MeteoSwiss/mch-anemoi-evaluation.git
    cd mch-anemoi-evaluation

Create and activate the conda environment for Snakemake with:

    mamba env create -f environment.yaml
    conda activate evalenv

## Execution

To run the workflow from command line, change the working directory.

```bash
cd path/to/mch-anemoi-evaluation
```

Adjust options in the default config file `config/config.yaml`.
Before running the complete workflow, you can perform a dry run using:

```bash
snakemake --dry-run
```

To run the workflow using **conda** and **slurm** on Balfrin:

```bash
snakemake --profile workflow/profile/balfrin
```
