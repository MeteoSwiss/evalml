from pathlib import Path
from typing import Dict, List, Any

from pydantic import BaseModel, Field, RootModel, HttpUrl, field_validator

PROJECT_ROOT = Path(__file__).parents[2]


class Dates(BaseModel):
    """Start/stop of the hindcast period and the launch frequency."""

    start: str = Field(
        ...,
        description="First forecast initialisation as an ISO-8601 formatted string.",
    )
    end: str = Field(
        ...,
        description="Last forecast initialisation as an ISO-8601 formatted string.",
    )
    frequency: str = Field(
        ...,
        description="Time between initialisations. Must be a combination of a number and a time unit (h or d).",
        pattern=r"^\d+[hd]$",
    )


class ExplicitDates(RootModel[List[str]]):
    """Explicit list of initialisation dates as ISO-8601 formatted strings."""


class AnemoiInferenceConfig(RootModel[Dict[str, Any]]):
    """Configuration for the Anemoi inference workflow."""


class InferenceResources(BaseModel):
    slurm_partition: str | None = Field(
        None,
        description="The Slurm partition to use for inference jobs, e.g. 'short-shared'.",
    )
    cpus_per_task: int | None = Field(
        None,
        description="Number of CPUs per task to request.",
    )
    mem_mb_per_cpu: int | None = Field(
        None,
        description="Memory (in MB) per CPU to request.",
    )
    runtime: str | None = Field(
        None,
        description="Maximum runtime for the job, e.g. '20m', '2h', '01:30:00'.",
    )
    gpu: int | None = Field(
        None,
        description="Number of GPUs to request.",
    )
    tasks: int | None = Field(
        None,
        description="Number of tasks per submission.",
    )


class RunConfig(BaseModel):
    mlflow_id: str = Field(
        ...,
        min_length=32,
        max_length=32,
        description="The mlflow run ID, as a 32-character hexadecimal string.",
    )
    label: str | None = Field(
        None,
        description="The label for the run that will be used in experiment results such as reports and figures.",
    )
    steps: str = Field(
        ...,
        description=(
            "Forecast lead times in hours, formatted as 'start/end/step'. "
            "The range includes the start lead time and continues with the given step "
            "until reaching or exceeding the end lead time. "
            "Example: '0/120/6' for lead times every 6 hours up to 120 h, "
            "or '0/33/6' up to 30 h."
        ),
    )
    extra_dependencies: List[str] = Field(
        default_factory=list,
        description="List of extra dependencies to install for this model. "
        "These will be added to the pyproject.toml file in the run directory.",
    )
    inference_resources: InferenceResources | None = Field(
        None,
        description="Resource requirements for inference jobs (optional; defaults handled externally).",
    )

    disable_local_eccodes_definitions: bool = Field(
        False,
        description="If true, the ECCODES_DEFINITION_PATH environment variable will not be set to the COSMO local definitions.",
    )

    config: Dict[str, Any] | str

    @field_validator("steps")
    def validate_steps(cls, v: str) -> str:
        if "/" not in v:
            raise ValueError(
                f"Steps must follow the format 'start/stop/step', got '{v}'"
            )
        parts = v.split("/")
        if len(parts) != 3:
            raise ValueError("Steps must be formatted as 'start/end/step'.")
        try:
            start, end, step = map(int, parts)
        except ValueError:
            raise ValueError("Start, end, and step must be integers.")
        if start > end:
            raise ValueError(
                f"Start ({start}) must be less than or equal to end ({end})."
            )
        if step <= 0:
            raise ValueError(f"Step ({step}) must be a positive integer.")
        return v


class ForecasterConfig(RunConfig):
    """Single training run stored in MLflow."""

    config: Dict[str, Any] | str = Field(
        default_factory=lambda _: str(
            PROJECT_ROOT / "resources" / "inference" / "configs" / "forecaster.yaml"
        ),
        description="Configuration for the forecaster run. Can be a dictionary of parameters or a path to a configuration file."
        "By default, it will point to resources/inference/configs/forecaster.yaml in the evalml repository.",
    )


class InterpolatorConfig(RunConfig):
    """Single training run stored in MLflow."""

    config: Dict[str, Any] | str = Field(
        default_factory=lambda _: str(
            PROJECT_ROOT / "resources" / "inference" / "configs" / "interpolator.yaml"
        ),
        description="Configuration for the interpolator run. Can be a dictionary of parameters or a path to a configuration file. "
        "By default, it will point to resources/inference/configs/interpolator.yaml in the evalml repository.",
    )

    forecaster: ForecasterConfig | None = Field(
        None,
        description="Configuration for the forecaster run that this interpolator is based on.",
    )


class BaselineConfig(BaseModel):
    """Configuration for a single baseline to include in the verification."""

    baseline_id: str = Field(
        ...,
        min_length=1,
        description="Identifier for the baseline, e.g. 'COSMO-E'.",
    )
    label: str = Field(
        ...,
        min_length=1,
        description="Label for the baseline that will be used in experiment results such as reports and figures.",
    )
    root: str = Field(
        ...,
        min_length=1,
        description="Root directory where the baseline data is stored.",
    )
    steps: str = Field(
        ...,
        description="Forecast steps to be used from baseline, e.g. '10/120/1'.",
        pattern=r"^\d*/\d*/\d*$",
    )


class AnalysisConfig(BaseModel):
    """Configuration for the analysis data used in the verification."""

    label: str = Field(
        ...,
        min_length=1,
        description="Label for the analysis that will be used in experiment results such as reports and figures.",
    )
    analysis_zarr: str = Field(
        ...,
        min_length=1,
        description="Path to the zarr dataset containing the analysis data.",
    )


class ForecasterItem(BaseModel):
    forecaster: ForecasterConfig


class InterpolatorItem(BaseModel):
    interpolator: InterpolatorConfig


class BaselineItem(BaseModel):
    baseline: BaselineConfig


class Locations(BaseModel):
    """Locations of data and services used in the workflow."""

    output_root: Path = Field(..., description="Root directory for all output files.")
    mlflow_uri: List[HttpUrl] = Field(
        ...,
        description="MLflow tracking URI(s) for the experiment. Can be a list of URIs if using multiple tracking servers.",
    )


class Stratification(BaseModel):
    """Stratification settings for the analysis."""

    regions: List[str] = Field(
        ...,
        description="List of region names for stratification.",
    )
    root: str = Field(
        ...,
        description="Root directory where the region shapefiles are stored.",
    )


class DefaultResources(BaseModel):
    """Default resource settings for job execution."""

    slurm_partition: str = Field(..., description="SLURM partition to use.")
    cpus_per_task: int = Field(..., ge=1, description="Number of CPUs per task.")
    mem_mb_per_cpu: int = Field(..., ge=1, description="Memory per CPU in MB.")
    runtime: str = Field(..., description="Maximum runtime, e.g. '1h'.")

    def parsable(self) -> str:
        """Convert the default resources to a string of key=value pairs."""
        return [f"{key}={value}" for key, value in self.model_dump().items()]


class GlobalResources(BaseModel):
    """
    Define resource limits that apply across all submissions.

    This model is intended to specify global constraints, such as
    the maximum number of GPUs that can be allocated in parallel,
    regardless of individual job settings.
    """

    gpus: int = Field(
        ...,
        ge=1,
        description=(
            "Maximum number of GPUs that may be used concurrently "
            "across all submissions."
        ),
    )

    def parsable(self) -> str:
        """Convert the global resources to a string of key=value pairs."""
        return [f"{key}={value}" for key, value in self.model_dump().items()]


class Profile(BaseModel):
    """Workflow execution profile."""

    executor: str = Field(..., description="Job executor, e.g. 'slurm'.")
    global_resources: GlobalResources
    default_resources: DefaultResources
    jobs: int = Field(..., ge=1, description="Maximum number of parallel jobs.")
    batch_rules: Dict[str, int] = Field(
        default_factory=dict,
        description="Define batches of the same rule that shall be executed within one job submission.",
    )

    def parsable(self) -> Dict[str, str]:
        """Convert the profile to a dictionary of command-line arguments."""
        out = []
        out += ["--executor", self.executor]
        out += ["--resources"] + self.global_resources.parsable()
        out += ["--default-resources"] + self.default_resources.parsable()
        out += ["--jobs", str(self.jobs)]

        # Add rule grouping options if specified
        if self.batch_rules:
            # Groups: rule=rule
            groups = [f"{rule}={rule}" for rule in self.batch_rules.keys()]
            # Group components: rule=<n>
            components = [f"{rule}={n}" for rule, n in self.batch_rules.items()]
            out += ["--groups"] + groups
            out += ["--group-components"] + components
        return out


class ConfigModel(BaseModel):
    """Top-level configuration."""

    description: str = Field(
        ...,
        description="Description of the experiment, e.g. 'Hindcast of the 2023 season.'",
    )
    experiment_label: str | None = Field(
        None,
        description="Optional label for the experiment that will be used in the experiment directory name. Defaults to the config file name if not provided.",
    )
    dates: Dates | ExplicitDates
    runs: List[ForecasterItem | InterpolatorItem] = Field(
        ...,
        description="Dictionary of runs to execute, with run IDs as keys and configurations as values.",
    )
    baselines: List[BaselineItem] = Field(
        ...,
        description="Dictionary of baselines to include in the verification.",
    )
    analysis: AnalysisConfig
    stratification: Stratification
    locations: Locations
    profile: Profile

    model_config = {
        "extra": "forbid",  # fail on misspelled keys
        "populate_by_name": True,
    }


def generate_config_schema() -> str:
    """Generate the JSON schema for the ConfigModel."""
    return ConfigModel.model_json_schema()


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Generate the JSON schema for the evalml configuration"
    )
    parser.add_argument(
        "output", type=str, help="Path to save the generated JSON schema."
    )
    args = parser.parse_args()

    with open(args.output, "w") as f:
        json.dump(generate_config_schema(), f, indent=2)
        f.write("\n")
