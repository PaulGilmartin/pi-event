"""Event metadata sent with captured camera media."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CameraEvent:
    """An image and video capture produced by a camera trigger."""

    image_path: Path
    video_path: Path
    captured_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if self.captured_at.tzinfo is None:
            raise ValueError("captured_at must include a timezone")

    def metadata(self) -> dict[str, Any]:
        """Return JSON-compatible event metadata."""
        return {
            "captured_at": self.captured_at.astimezone(UTC).isoformat(),
        }
