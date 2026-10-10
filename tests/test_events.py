from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from pi_event.events import CameraEvent


def test_camera_event_generates_identity_and_serializes_metadata() -> None:
    event = CameraEvent(
        device_id="garden-pi",
        motion_score=0.42,
        image_path=Path("capture.jpg"),
        video_path=Path("capture.h264"),
    )

    metadata = event.metadata()

    assert UUID(metadata["event_id"]) == event.event_id
    assert metadata == {
        "event_id": str(event.event_id),
        "event_type": "motion",
        "device_id": "garden-pi",
        "captured_at": event.captured_at.isoformat(),
        "motion_score": 0.42,
    }
    assert event.captured_at.tzinfo is UTC


def test_camera_event_normalizes_timestamp_to_utc() -> None:
    event = CameraEvent(
        device_id="garden-pi",
        motion_score=0.5,
        image_path=Path("capture.jpg"),
        video_path=Path("capture.h264"),
        captured_at=datetime.fromisoformat("2026-10-10T10:00:00+02:00"),
    )

    assert event.metadata()["captured_at"] == "2026-10-10T08:00:00+00:00"


@pytest.mark.parametrize("motion_score", [-0.01, 1.01])
def test_camera_event_rejects_invalid_motion_score(motion_score: float) -> None:
    with pytest.raises(ValueError, match="motion_score"):
        CameraEvent(
            device_id="garden-pi",
            motion_score=motion_score,
            image_path=Path("capture.jpg"),
            video_path=Path("capture.h264"),
        )
