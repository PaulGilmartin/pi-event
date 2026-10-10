"""Event metadata sent with captured camera media."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class CameraEvent:
    """An image and video capture produced by a camera trigger."""

    device_id: str
    motion_score: float
    image_path: Path
    video_path: Path
    event_id: UUID = field(default_factory=uuid4)
    captured_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    event_type: Literal["motion"] = field(default="motion", init=False)

    def __post_init__(self) -> None:
        if not self.device_id.strip():
            raise ValueError("device_id must not be empty")
        if not 0 <= self.motion_score <= 1:
            raise ValueError("motion_score must be between 0 and 1")
        if self.captured_at.tzinfo is None:
            raise ValueError("captured_at must include a timezone")

    def metadata(self) -> dict[str, Any]:
        """Return JSON-compatible event metadata."""
        return {
            "event_id": str(self.event_id),
            "event_type": self.event_type,
            "device_id": self.device_id,
            "captured_at": self.captured_at.astimezone(UTC).isoformat(),
            "motion_score": self.motion_score,
        }
