"""Query and validate the publication manifest.

``Manifest`` wraps the raw manifest dict and exposes a small, typed API so the
CLI, the Snakemake wrappers and the notebooks all resolve data paths the same
way — turning today's cryptic, hand-assembled ``run_id``/``TRUTH_HASH`` strings
into named lookups, and turning silent config mistakes into clear errors.
"""

from dataclasses import dataclass
from pathlib import Path

from evalml.resolution import leadtime_producible


class ResolutionError(ValueError):
    """Raised when a publication request cannot be coherently resolved."""


@dataclass(frozen=True)
class Participant:
    """A run or baseline that takes part in the publication figures."""

    id: str
    label: str
    role: str  # "candidate" | "baseline"
    model_type: str | None
    steps: str | None
    member: str | None
    source_root: str | None
    is_candidate: bool
    paths: dict

    @classmethod
    def from_dict(cls, d: dict) -> "Participant":
        return cls(
            id=d["id"],
            label=d["label"],
            role=d["role"],
            model_type=d.get("model_type"),
            steps=d.get("steps"),
            member=d.get("member"),
            source_root=d.get("source_root"),
            is_candidate=d.get("is_candidate", d.get("role") == "candidate"),
            paths=d.get("paths", {}),
        )


class Manifest:
    """Typed accessor over a loaded publication manifest dict."""

    def __init__(self, data: dict):
        self._data = data
        self._participants = [
            Participant.from_dict(p) for p in data.get("participants", [])
        ]

    # -- raw fields -------------------------------------------------------
    @property
    def truth(self) -> dict:
        return self._data.get("truth", {})

    @property
    def output_root(self) -> str:
        return self._data.get("output_root", "output")

    @property
    def master_hash(self) -> str | None:
        return self._data.get("master_hash")

    @property
    def init_times(self) -> list[str]:
        return list(self._data.get("dates", {}).get("init_times", []))

    @property
    def publication(self) -> dict:
        return self._data.get("publication", {})

    # -- participant queries ---------------------------------------------
    def participants(self, role: str | None = None) -> list[Participant]:
        if role is None:
            return list(self._participants)
        return [p for p in self._participants if p.role == role]

    def get_candidate(self, label: str | None = None) -> Participant:
        """Return the candidate run, requiring an explicit choice if ambiguous.

        With a single candidate, ``label`` may be omitted. With several, a
        ``label`` is required — replacing the old silent "pick the first
        candidate" behaviour.
        """
        candidates = self.participants(role="candidate")
        if not candidates:
            raise ResolutionError("No candidate run found in the manifest.")
        if label is not None:
            for c in candidates:
                if c.label == label:
                    return c
            raise ResolutionError(
                f"No candidate with label {label!r}. "
                f"Available candidates: {[c.label for c in candidates]}."
            )
        if len(candidates) == 1:
            return candidates[0]
        raise ResolutionError(
            f"Multiple candidates {[c.label for c in candidates]}; "
            f"pass --candidate to choose one."
        )

    def resolve_baseline(self, label: str) -> Participant:
        for p in self.participants(role="baseline"):
            if p.label == label:
                return p
        available = [p.label for p in self.participants(role="baseline")]
        raise ResolutionError(
            f"No baseline with label {label!r} found. "
            f"Available baseline labels: {available}."
        )

    # -- path resolution --------------------------------------------------
    def verif_paths(
        self, include=("candidate", "baseline")
    ) -> list[tuple[str, str]]:
        """(path, label) for the aggregated verification files of participants.

        ``publication_figures`` includes every participant (candidates +
        baselines); pass ``include`` to narrow it.
        """
        out = []
        for p in self._participants:
            if p.role in include:
                out.append((p.paths["verif_aggregated"], p.label))
        return out

    def grib_dir(self, participant: Participant, init_time: str) -> str:
        template = participant.paths.get("grib_dir_template")
        if template is None:
            raise ResolutionError(
                f"Participant {participant.label!r} ({participant.role}) has no GRIB "
                f"directory (only candidate runs do)."
            )
        return template.format(init_time=init_time)

    def scoremap_path(
        self, participant: Participant, param: str, leadtime: int
    ) -> str:
        template = participant.paths.get("scoremap_template")
        if template is None:
            raise ResolutionError(
                f"Participant {participant.label!r} has no scoremap template."
            )
        return template.format(param=param, leadtime=leadtime)

    def meteogram_baseline_specs(self) -> str:
        """Rebuild the ``root|steps|member|label;...`` spec for the meteogram overlay.

        Every configured baseline is overlaid, read according to its ``member``:
        ``control`` (or a numbered member) reads that single member — fast — while
        ``mean`` averages the whole ensemble at plot time (slow). Choose per
        baseline via the ``member`` field in the config.
        """
        specs = []
        for p in self.participants(role="baseline"):
            if p.source_root:
                specs.append(f"{p.source_root}|{p.steps}|{p.member}|{p.label}")
        return ";".join(specs)

    # -- coherence validation --------------------------------------------
    def validate_request(self, figure: str, **opts) -> None:
        """Raise :class:`ResolutionError` if a figure request is incoherent.

        This guards interactive/standalone use, which loads only the manifest and
        never re-runs the pydantic config validation. The rules mirror the
        ConfigModel checks (a–e) using manifest data.
        """
        if figure == "figures":
            if not self.verif_paths():
                raise ResolutionError("No participants found for the figures plot.")
            return

        if figure == "meteogram":
            init_time = opts.get("init_time")
            candidate = opts.get("candidate")  # label or None
            cand = self.get_candidate(candidate)
            if init_time is not None and init_time not in self.init_times:
                raise ResolutionError(
                    f"meteogram init_time {init_time!r} is not in the manifest's "
                    f"initialisation times."
                )
            # GRIB existence is a runtime check (data may not be produced yet).
            if init_time is not None:
                grib = Path(self.grib_dir(cand, init_time))
                if not grib.exists():
                    raise ResolutionError(
                        f"GRIB directory {grib} is missing; run inference for "
                        f"init_time {init_time} first."
                    )
            return

        if figure == "scoremaps":
            if self.truth.get("type") != "zarr":
                raise ResolutionError(
                    f"scoremaps require a gridded (zarr) truth; manifest truth "
                    f"{self.truth.get('label')!r} is {self.truth.get('type')}."
                )
            cand = self.get_candidate(opts.get("candidate"))
            baseline = self.resolve_baseline(opts["baseline"])
            leadtime = int(opts["leadtime"])
            for p in (cand, baseline):
                if p.steps and not leadtime_producible(p.steps, leadtime):
                    raise ResolutionError(
                        f"scoremaps leadtime {leadtime}h is not produced by "
                        f"{p.label!r} (steps '{p.steps}')."
                    )
            return

        raise ResolutionError(f"Unknown figure type {figure!r}.")
