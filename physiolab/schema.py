"""Common result schema shared by every modality engine."""
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from . import __version__

DISCLAIMER = "Research and education prototype. Not for diagnosis or clinical decision-making."


@dataclass
class AnalysisResult:
    modality: str
    source_name: str
    quality: dict
    measurements: dict
    flags: list = field(default_factory=list)
    limitations: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def make_provenance(method: str) -> dict:
    return {
        "platform_version": __version__,
        "method": method,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "disclaimer": DISCLAIMER,
    }