from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, ClassVar, FrozenSet, Optional

from pydantic import BaseModel, Field, RootModel, field_validator, model_validator

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
    blacklist: List[str] = Field(
        default_factory=list,
        description="Optional list of initialisation dates (ISO-8601) to exclude from processing.",
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
    checkpoint: str = Field(
        ...,
        description="The model checkpoint to use. Can be an MLflow run URL, a Hugging Face `.ckpt` URL, or a local checkpoint path.",
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


class TemporalDownscalerConfig(RunConfig):
    """Single training run stored in MLflow."""

    config: Dict[str, Any] | str = Field(
        default_factory=lambda _: str(
            PROJECT_ROOT
            / "resources"
            / "inference"
            / "configs"
            / "temporal_downscaler.yaml"
        ),
        description="Configuration for the temporal downscaler run. Can be a dictionary of parameters or a path to a configuration file. "
        "By default, it will point to resources/inference/configs/temporal_downscaler.yaml in the evalml repository.",
    )

    forecaster: ForecasterConfig | None = Field(
        None,
        description="Configuration for the forecaster run that this temporal downscaler is based on.",
    )


class BaselineConfig(BaseModel):
    """Configuration for a single baseline to include in the verification."""

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
    member: str = Field(
        "000",
        description=(
            "Ensemble member to use: '000'/'control' for control, 'median' for the "
            "pre-computed median, 'mean' to average all members, or any 3-digit member ID. "
            "WARNING: when using 'median' with temporally aggregated parameters (e.g. 6-hourly "
            "precipitation), do NOT aggregate the hourly median — this introduces a significant "
            "negative bias. Instead, read the corresponding pre-aggregated median files "
            "(e.g. ..._median06h). Ensure the baseline data files match the aggregation period."
        ),
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


class TemporalDownscalerItem(BaseModel):
    temporal_downscaler: TemporalDownscalerConfig


class BaselineItem(BaseModel):
    baseline: BaselineConfig


class ScoreMapsConfig(BaseModel):
    """Parameters controlling which score map plots are produced."""

    enabled: bool = Field(
        default=False,
        description="Whether to produce score maps (computationally intensive).",
    )
    params: List[str] = Field(
        default=["T_2M"],
        description=(
            "List of parameters to plot. Supported values: T_2M, TD_2M, U_10M, V_10M, "
            "PS, PMSL, SP_10M (derived from U_10M/V_10M), TOT_PREC1, TOT_PREC6, TOT_PREC24 "
            "(period-accumulated precipitation, period encoded in the name)."
        ),
    )
    leadtimes: List[int] = Field(
        default=[6, 24],
        description="List of lead times (hours) to plot.",
    )
    scores: List[str] = Field(
        default=["BIAS"],
        description="List of verification scores to plot. Supported: BIAS, RMSE, MAE.",
    )
    regions: List[str] = Field(
        default=["switzerland"],
        description="List of regions to plot (e.g. switzerland, centraleurope).",
    )
    seasons: List[str] = Field(
        default=["all"],
        description="List of seasons to plot ('all', 'DJF', 'MAM', 'JJA', 'SON').",
    )
    init_hours: List[str] = Field(
        default=["all"],
        description=(
            "List of initialization hours to plot. Use 'all' for the unstratified "
            "view, or zero-padded hour strings like '00', '06', '12', '18'."
        ),
    )

    model_config = {"extra": "forbid"}


class DomainConfig(BaseModel):
    """A custom map domain defined by name, extent, and projection."""

    name: str = Field(..., description="Name for the custom domain (used as wildcard).")
    extent: List[float] | None = Field(
        None,
        description="Geographic extent as [lon_min, lon_max, lat_min, lat_max] in PlateCarree coordinates. None means full globe.",
    )
    projection: str = Field(
        "orthographic",
        description="Projection name (must be a key in plotting._PROJECTIONS, e.g. 'orthographic').",
    )

    model_config = {"extra": "forbid"}


class MeteogramConfig(BaseModel):
    """Configuration for meteogram generation."""

    enabled: bool = Field(
        default=True,
        description="Whether to generate meteograms (time series plots at stations).",
    )
    stations: List[str] = Field(
        default=["GVE", "KLO", "LUG"],
        description="List of PeakWeather station IDs to generate meteograms for.",
    )


class AnimationsConfig(BaseModel):
    """Configuration for animation generation."""

    enabled: bool = Field(
        default=True,
        description="Whether to generate forecast animations (GIFs per param and region).",
    )
    domains: List[str | DomainConfig] = Field(
        default=["globe", "europe", "switzerland"],
        description=(
            "Domains to generate animations for. Each entry is either a named domain "
            "(e.g. 'globe', 'europe', 'switzerland') defined in plotting.DOMAINS, "
            "or a custom domain dict with 'name', optional 'extent' "
            "[lon_min, lon_max, lat_min, lat_max], and optional 'projection'."
        ),
    )
    frames_per_second: float = Field(
        default=2.0,
        gt=0,
        description="Frames per second for the output GIF animation.",
    )


class ScorecardConfig(BaseModel):
    """Configuration for a single named scorecard."""

    baseline: str = Field(
        ...,
        description="Baseline label to compare against (must match the `label` field of a baseline entry in `runs`).",
    )
    lead_times: str = Field(
        ...,
        description="Lead-time range as start/stop/step (hours).",
    )
    stratification: str = Field(
        ...,
        description="Dimension to use as scorecard columns (e.g. 'region').",
    )
    variables: List[str] = Field(
        ...,
        description=(
            "Variables and metrics as scorecard rows (VAR:M1,M2 format). "
            "An empty list [] is accepted by the schema but falls back to a "
            "hard-coded RMSE-only set (U_10M, V_10M, T_2M, PMSL, TD_2M, TOT_PREC). "
            "Omit ':...' after a variable name to include all available metrics for it."
        ),
    )

    model_config = {"extra": "forbid"}


class ExperimentScorecardConfig(BaseModel):
    """Top-level scorecard block: a single enabled flag plus named scorecard sections."""

    enabled: bool = Field(
        default=True,
        description="Whether to generate scorecards.",
    )
    sections: Dict[str, ScorecardConfig] = Field(
        default_factory=dict,
        description="Named scorecard configurations (e.g. nowcasting, short_range, medium_range).",
    )

    model_config = {"extra": "forbid"}


class ShowcaseConfig(BaseModel):
    """Configuration for the showcase workflow."""

    params: List[str] = Field(
        default=["T_2M", "SP_10M"],
        description="List of parameters to generate animations and meteograms for.",
    )
    meteograms: MeteogramConfig = Field(
        default_factory=MeteogramConfig,
        description="Configuration for meteogram generation.",
    )
    animations: AnimationsConfig = Field(
        default_factory=AnimationsConfig,
        description="Configuration for animation generation.",
    )


class PublicationLeadtimesConfig(BaseModel):
    """Configuration for the publication lead-time figure."""

    enabled: bool = Field(
        default=False,
        description="Whether to generate the lead-time score figures.",
    )

    model_config = {"extra": "forbid"}


class PublicationMeteogramConfig(BaseModel):
    """Case selection for the publication meteogram figure."""

    enabled: bool = Field(
        default=False,
        description="Whether to generate the publication meteogram figure.",
    )
    init_time: str = Field(
        ...,
        pattern=r"^\d{12}$",
        description="Initialisation time (YYYYMMDDHHMM) of the case to plot.",
    )
    station: str = Field(
        ...,
        min_length=1,
        description="PeakWeather station ID to plot (e.g. KLO, GVE, LUG).",
    )
    params: List[str] = Field(
        default=["T_2M", "TOT_PREC", "SP_10M", "DD_10M"],
        min_length=1,
        description="Display parameters (one panel each); SP_10M/DD_10M are derived.",
    )

    model_config = {"extra": "forbid"}


class PublicationScoremapsConfig(BaseModel):
    """Configuration for the publication scoremap figures."""

    enabled: bool = Field(
        default=False,
        description="Whether to generate the publication scoremap figures.",
    )
    steps: Optional[List[int]] = Field(
        default=None,
        description=(
            "Lead times in hours to produce one scoremap figure each. "
            "Defaults to experiment.scoremaps.leadtimes when omitted."
        ),
    )
    params: List[str] = Field(
        default=["T_2M", "SP_10M"],
        description="Parameters to show as rows in the scoremap panel.",
    )
    scores: List[str] = Field(
        default=["MSE_SKILL", "BIAS_CONTRIB"],
        description="Score columns to show in the scoremap panel.",
    )
    baseline_label: str = Field(
        default="ICON-CH1-CTRL",
        description="Label of the baseline run to compare against.",
    )
    season: str = Field(
        default="all",
        description="Season filter passed to the scoremap script.",
    )
    region: str = Field(
        default="switzerland",
        description="Geographic region for the scoremap plot.",
    )

    model_config = {"extra": "forbid"}


class PublicationTeaserConfig(BaseModel):
    """Case selection and styling for the publication teaser ('hero') figure."""

    enabled: bool = Field(
        default=False,
        description="Whether to generate the publication teaser figure.",
    )
    init_time: str = Field(
        ...,
        pattern=r"^\d{12}$",
        description="Initialisation time (YYYYMMDDHHMM) of the case to plot.",
    )
    leadtime: int = Field(
        default=24,
        ge=0,
        description="Lead time in hours for the plotted forecast frame.",
    )
    dark: bool = Field(
        default=False,
        description="Dark variant: black background, white text/ticks, glow.",
    )
    grid: Optional[str] = Field(
        default=None,
        description="ICON grid file for the true-mesh (tripcolor) zoom. "
        "Falls back to a Delaunay mesh when omitted.",
    )

    model_config = {"extra": "forbid"}


class PublicationConfig(BaseModel):
    """Configuration for the publication workflow."""

    leadtimes: Optional[PublicationLeadtimesConfig] = Field(
        default=None,
        description="Lead-time score figure settings (omit to skip).",
    )
    meteogram: Optional[PublicationMeteogramConfig] = Field(
        default=None,
        description="Publication meteogram case selection (omit to skip).",
    )
    scoremaps: Optional[PublicationScoremapsConfig] = Field(
        default=None,
        description="Publication scoremap figure settings (omit to skip). "
        "Requires a gridded (zarr) truth source.",
    )
    teaser: Optional[PublicationTeaserConfig] = Field(
        default=None,
        description="Publication teaser (hero) figure settings (omit to skip).",
    )

    model_config = {"extra": "forbid"}


class Locations(BaseModel):
    """Locations of data and services used in the workflow."""

    output_root: Path = Field(..., description="Root directory for all output files.")


class Stratification(BaseModel):
    """Stratification settings for the analysis."""

    regions: List[str] = Field(
        default_factory=list,
        description="List of region names for stratification. Empty list means no spatial stratification.",
    )
    root: Optional[str] = Field(
        None,
        description="Root directory where the region shapefiles are stored. Required when regions is non-empty.",
    )


class Dashboard(BaseModel):
    """Settings for the dashboard"""

    stratification: List[str] = Field(
        ...,
        description="Stratifications to include in the dashboard (any of season, region, init_hour)",
    )


class ExperimentConfig(BaseModel):
    """Configuration for the experiment workflow outputs."""

    stratification: Stratification = Field(
        ...,
        description="Spatial stratification settings for the analysis.",
    )
    params: List[str] = Field(
        default=["T_2M", "TD_2M", "SP_10M", "PS", "PMSL", "TOT_PREC6"],
        description="List of parameters to compute verification metrics for.",
    )
    thresholds: Dict[str, Dict[str, List[float]]] = Field(
        default_factory=dict,
        description=(
            "Dictionary mapping parameter names to threshold dicts. "
            "Each dict maps operator keys (gt, ge, lt, le, eq, ne) to lists of threshold values."
        ),
    )
    dashboard: Dashboard = Field(
        ...,
        description="Settings for the experiment dashboard.",
    )
    scorecards: Optional[ExperimentScorecardConfig] = Field(
        default=None,
        description="Scorecard generation configuration. Omit or set enabled: false to disable.",
    )
    scoremaps: Optional[ScoreMapsConfig] = Field(
        default=None,
        description="Score map plot configuration. Omit or set enabled: false to disable.",
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


class DefaultResources(BaseModel):
    """Default resource settings for job execution."""

    slurm_partition: str = Field(..., description="SLURM partition to use.")
    cpus_per_task: int = Field(..., ge=1, description="Number of CPUs per task.")
    mem_mb_per_cpu: int = Field(..., ge=1, description="Memory per CPU in MB.")
    runtime: str = Field(..., description="Maximum runtime, e.g. '1h'.")
    slurm_account: str | None = Field(None, description="SLURM account to charge.")
    gpus: int | None = Field(
        None, ge=0, description="Default GPU count per job (0 for non-GPU jobs)."
    )

    def parsable(self) -> list[str]:
        """Convert the default resources to a string of key=value pairs."""
        return [
            f"{key}={value}"
            for key, value in self.model_dump().items()
            if value is not None
        ]


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


class MecConfig(BaseModel):
    """Paths to input observation files for the MEC verification step."""

    ekf_root: str = Field(
        ...,
        description="Root directory for EKF SYNOP files. Files are expected at {ekf_root}/{YYYYMM}/ekfSYNOP_{init}00.nc.",
    )
    mon_synop_root: str = Field(
        ...,
        description="Root directory for monSYNOP files. Files are expected at {mon_synop_root}/{YYYYMMDDH}/monSYNOP.nc.",
    )
    ver_synop_root: str = Field(
        ...,
        description="Root directory for reference verSYNOP files. Files are expected at {ver_synop_root}/verSYNOP_{init}00.nc.",
    )

    model_config = {"extra": "forbid"}


class Ffv2Config(BaseModel):
    """Configuration for the FFV2 scoring pipeline."""

    experiment_ids: str = Field(
        ...,
        description="Comma-separated experiment IDs passed to FFV2.",
    )
    veri_ens_member: str = Field(
        ...,
        description="Comma-separated ensemble member indices passed to FFV2, one per experiment ID (typically -1 for deterministic runs).",
    )
    catthresholds: dict[str, list[float]] = Field(
        ...,
        description="Per-variable categorical thresholds for FFV2, mapping FFV2 variable names to lists of threshold values.",
    )
    pecthresholds: dict[str, dict[str, float]] = Field(
        ...,
        description="Per-variable PEC thresholds for FFV2, mapping FFV2 variable names to a dict with exactly one 'lower' and one 'upper' value.",
    )
    experiment_description: str = Field(
        ...,
        description="Short description of the experiment for FFV2 output files.",
    )
    file_description: str = Field(
        ...,
        description="File description string used in FFV2 output file naming.",
    )
    domain_table: str = Field(
        ...,
        description="Path to the domain table file (polygon) used by FFV2.",
    )
    blacklists: str = Field(
        ...,
        description="Path to the blacklist directory used by FFV2.",
    )

    @field_validator("veri_ens_member", mode="before")
    @classmethod
    def coerce_veri_ens_member_to_str(cls, v):
        return str(v)

    @field_validator("pecthresholds")
    @classmethod
    def validate_pecthresholds(cls, v):
        for var, bounds in v.items():
            invalid = set(bounds) - {"lower", "upper"}
            if invalid:
                raise ValueError(
                    f"pecthresholds[{var!r}] contains invalid keys {invalid}; only 'lower' and 'upper' are allowed."
                )
            if not bounds:
                raise ValueError(
                    f"pecthresholds[{var!r}] must have at least one of 'lower' or 'upper'."
                )
        return v

    model_config = {"extra": "forbid"}


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
    runs: List[ForecasterItem | TemporalDownscalerItem | BaselineItem] = Field(
        ...,
        description="List of experiment participants, including forecaster/temporal downscaler ML runs and baselines.",
    )
    truth: TruthConfig | None
    lapse_rate_correction: bool = Field(
        default=True,
        description="Apply standard-atmosphere lapse-rate correction to T_2M.",
    )
    experiment: ExperimentConfig = Field(
        ...,
        description="Settings for the experiment workflow outputs.",
    )
    locations: Locations
    profile: Profile
    showcase: ShowcaseConfig = Field(
        default_factory=ShowcaseConfig,
        description="Settings for the showcase workflow.",
    )
    publication: PublicationConfig = Field(
        default_factory=PublicationConfig,
        description="Settings for the publication workflow.",
    )
    mec: MecConfig | None = Field(
        None,
        description="Input observation paths for the MEC verification step. Required when running with --mec.",
    )
    ffv2: Ffv2Config | None = Field(
        None,
        description="Configuration for the FFV2 scoring pipeline. Required when running with --ffv2.",
    )

    @model_validator(mode="after")
    def validate_scoremap_leadtimes(self) -> "ConfigModel":
        sm = self.experiment.scoremaps
        if sm is None or not sm.enabled:
            return self
        requested = set(sm.leadtimes)
        for item in self.runs:
            steps = getattr(item, next(iter(type(item).model_fields))).steps
            start, end, step = map(int, steps.split("/"))
            producible = set(range(start, end + 1, step))
            unsupported = requested - producible
            if unsupported:
                raise ValueError(
                    f"scoremaps.leadtimes contains {sorted(unsupported)} h which are not "
                    f"produced by participant with steps '{steps}'."
                )
        return self

    def _enumerated_init_times(self) -> set[str]:
        """All initialisation times as YYYYMMDDHHMM strings (post-blacklist)."""
        fmt_in = "%Y-%m-%dT%H:%M"
        fmt_out = "%Y%m%d%H%M"
        if isinstance(self.dates, ExplicitDates):
            return {
                datetime.strptime(t, fmt_in).strftime(fmt_out) for t in self.dates.root
            }
        start = datetime.strptime(self.dates.start, fmt_in)
        end = datetime.strptime(self.dates.end, fmt_in)
        magnitude, unit = int(self.dates.frequency[:-1]), self.dates.frequency[-1]
        freq = timedelta(days=magnitude) if unit == "d" else timedelta(hours=magnitude)
        blacklist = {datetime.strptime(t, fmt_in) for t in self.dates.blacklist}
        out: set[str] = set()
        t = start
        while t <= end:
            if t not in blacklist:
                out.add(t.strftime(fmt_out))
            t += freq
        return out

    @model_validator(mode="after")
    def validate_publication(self) -> "ConfigModel":
        """Coherence checks for the publication workflow (only when enabled).

        Catches config mistakes that would otherwise surface as cryptic Snakemake
        graph-expansion errors: a meteogram case outside the date range, scoremaps
        against a non-gridded truth, an unknown baseline label, or a lead time that
        no participant produces.
        """
        from evalml.resolution import leadtime_producible

        pub = self.publication
        if pub is None:
            return self

        # Split runs into candidate runs (forecaster/temporal_downscaler) and baselines.
        candidate_steps: list[str] = []
        baseline_steps: dict[str, str] = {}
        for item in self.runs:
            field = next(iter(item.model_fields))
            inner = getattr(item, field)
            if field == "baseline":
                baseline_steps[inner.label] = inner.steps
            else:
                candidate_steps.append(inner.steps)

        # (c) meteogram init_time must be one of the configured initialisation times.
        if pub.meteogram is not None and pub.meteogram.enabled:
            valid_init = self._enumerated_init_times()
            if pub.meteogram.init_time not in valid_init:
                raise ValueError(
                    f"publication.meteogram.init_time {pub.meteogram.init_time!r} is not "
                    f"in the configured initialisation times (check `dates`)."
                )

        sm = pub.scoremaps
        if sm is not None and sm.enabled:
            # (a) scoremaps need a gridded (zarr) truth, not station/jretrieve obs.
            if self.truth is None or "jretrieve" in str(self.truth.root):
                truth_label = self.truth.label if self.truth else "<none>"
                raise ValueError(
                    f"publication.scoremaps requires a gridded (zarr) truth source; "
                    f"configured truth {truth_label!r} is jretrieve/obs and has no "
                    f"spatial scoremaps."
                )
            # (d) the named baseline must exist.
            if sm.baseline_label not in baseline_steps:
                raise ValueError(
                    f"publication.scoremaps.baseline_label {sm.baseline_label!r} not found. "
                    f"Available baseline labels: {list(baseline_steps)}."
                )
            # (b) each lead time must be produced by every candidate AND the baseline.
            # Lead times come from scoremaps.steps, else experiment.scoremaps.leadtimes.
            base_steps = baseline_steps[sm.baseline_label]
            exp_sm = self.experiment.scoremaps
            leadtimes = sm.steps or (exp_sm.leadtimes if exp_sm else None) or [24]
            for leadtime in leadtimes:
                for steps in candidate_steps:
                    if not leadtime_producible(steps, leadtime):
                        raise ValueError(
                            f"publication.scoremaps leadtime {leadtime}h is not produced "
                            f"by candidate run with steps '{steps}'."
                        )
                if not leadtime_producible(base_steps, leadtime):
                    raise ValueError(
                        f"publication.scoremaps leadtime {leadtime}h is not produced by "
                        f"baseline {sm.baseline_label!r} (steps '{base_steps}')."
                    )
        return self

    model_config = {
        "extra": "forbid",  # fail on misspelled keys
        "populate_by_name": True,
    }


def generate_config_schema() -> str:
    """Generate the JSON schema for the ConfigModel."""
    return ConfigModel.model_json_schema()


# Module-level constants for use in Snakemake and elsewhere
RUN_ENV_FIELDS = RunConfig.ENV_FIELDS


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
