from pathlib import Path
from typing import Dict, List, Any, ClassVar, FrozenSet

from pydantic import BaseModel, Field, RootModel, field_validator

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
    # Identity contract: fields that determine the inference ENVIRONMENT (venv, squashfs).
    # Changing any of these requires a new environment to be built.
    ENV_FIELDS: ClassVar[FrozenSet[str]] = frozenset(
        {"checkpoint", "extra_requirements", "disable_local_eccodes_definitions"}
    )
    # Fields excluded from ALL hashing (display/resource metadata only).
    HASH_EXCLUDE: ClassVar[FrozenSet[str]] = frozenset({"label", "inference_resources"})

    checkpoint: str = Field(
        ...,
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
    extra_requirements: List[str] = Field(
        default_factory=list,
        description="List of extra dependencies to install for this model. "
        "These will be added to the requirements.txt file in the run directory.",
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

    model_config = {"extra": "forbid"}

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

    baseline_id: str | None = Field(
        None,
        min_length=1,
        description="Deprecated compatibility field. Workflow baseline IDs are derived from the stem of `root`.",
    )
    label: str = Field(
        ...,
        min_length=1,
        description="Label for the baseline that will be used in experiment results such as reports and figures.",
    )
    root: str = Field(
        ...,
        min_length=1,
        description="Root directory where the baseline data is stored. The workflow derives the baseline ID from the stem of this path.",
    )
    steps: str = Field(
        ...,
        description="Forecast steps to be used from baseline, e.g. '10/120/1'.",
        pattern=r"^\d*/\d*/\d*$",
    )


class TruthConfig(BaseModel):
    """Configuration for the truth data used in the verification."""

    label: str = Field(
        ...,
        min_length=1,
        description="Label that will be used in experiment results such as reports and figures.",
    )
    root: str = Field(
        ...,
        min_length=1,
        description="Path to the root of the dataset.",
    )


class ForecasterItem(BaseModel):
    forecaster: ForecasterConfig


class InterpolatorItem(BaseModel):
    interpolator: InterpolatorConfig


class BaselineItem(BaseModel):
    baseline: BaselineConfig


class ScoreMapsConfig(BaseModel):
    """Parameters controlling which score map plots are produced."""

    params: List[str] = Field(
        default=["T_2M", "TD_2M", "U_10M", "V_10M", "SP_10M", "PS", "PMSL", "TOT_PREC"],
        description=(
            "List of parameters to plot. Supported values: T_2M, TD_2M, U_10M, V_10M, "
            "PS, PMSL, TOT_PREC (native), and SP_10M (derived wind speed from U_10M/V_10M)."
        ),
    )
    leadtimes: List[int] = Field(
        default=list(range(6, 121, 6)),
        description="List of lead times (hours) to plot.",
    )
    metrics: List[str] = Field(
        default=["BIAS", "RMSE", "MAE"],
        description="List of verification metrics to plot.",
    )
    regions: List[str] = Field(
        default=["switzerland", "centraleurope"],
        description="List of regions to plot.",
    )
    seasons: List[str] = Field(
        default=["all", "DJF", "MAM", "JJA", "SON"],
        description="List of seasons to plot.",
    )
    init_hours: List[str] = Field(
        default=["all"],
        description=(
            "List of initialization hours to plot. Use 'all' for the unstratified "
            "view, or zero-padded hour strings like '00', '06', '12', '18'."
        ),
    )


class Locations(BaseModel):
    """Locations of data and services used in the workflow."""

    output_root: Path = Field(..., description="Root directory for all output files.")


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


class Dashboard(BaseModel):
    """Settings for the dashboard"""

    stratification: List[str] = Field(
        ...,
        description="Stratifications to include in the dashboard (any of season, region, init_hour)",
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
    config_label: str | None = Field(
        None,
        description="Optional label for the experiment that will be used in the experiment directory name. Defaults to the config file name if not provided.",
    )
    dates: Dates | ExplicitDates
    runs: List[ForecasterItem | InterpolatorItem | BaselineItem] = Field(
        ...,
        description="List of experiment participants, including forecaster/interpolator ML runs and baselines.",
    )
    baselines: List[BaselineItem] = Field(
        default_factory=list,
        description="Deprecated top-level baselines list. Prefer defining baseline entries directly in `runs`.",
    )
    truth: TruthConfig | None
    stratification: Stratification
    thresholds: Dict[str, Dict[str, List[float]]] = Field(
        default_factory=dict,
        description=(
            "Dictionary mapping parameter names to threshold dicts. "
            "Each dict maps operator keys (gt, ge, lt, le, eq, ne) to lists of threshold values."
        ),
    )

    @field_validator("thresholds")
    @classmethod
    def validate_threshold_operators(
        cls, v: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, Dict[str, List[float]]]:
        _VALID_OPS = {"gt", "ge", "lt", "le", "eq", "ne"}
        for param, op_dict in v.items():
            invalid = set(op_dict) - _VALID_OPS
            if invalid:
                raise ValueError(
                    f"Invalid operator key(s) {invalid!r} for parameter '{param}'. "
                    f"Must be one of {_VALID_OPS}."
                )
        return v

    dashboard: Dashboard
    locations: Locations
    profile: Profile
    score_maps: ScoreMapsConfig = Field(
        default_factory=ScoreMapsConfig,
        description="Parameters for score map plots (used with --maps flag).",
    )

    model_config = {
        "extra": "forbid",  # fail on misspelled keys
        "populate_by_name": True,
    }


def generate_config_schema() -> str:
    """Generate the JSON schema for the ConfigModel."""
    return ConfigModel.model_json_schema()


# Module-level constants for use in Snakemake and elsewhere
RUN_ENV_FIELDS = RunConfig.ENV_FIELDS
RUN_HASH_EXCLUDE = RunConfig.HASH_EXCLUDE


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
