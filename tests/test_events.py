from datetime import UTC, datetime
from pathlib import Path

from pi_event.events import CameraEvent


def test_camera_event_serializes_metadata() -> None:
    event = CameraEvent(
        image_path=Path("capture.jpg"),
        video_path=Path("capture.h264"),
    )

    metadata = event.metadata()

    assert metadata == {
        "captured_at": event.captured_at.isoformat(),
    }
    assert event.captured_at.tzinfo is UTC


def test_camera_event_normalizes_timestamp_to_utc() -> None:
    event = CameraEvent(
        image_path=Path("capture.jpg"),
        video_path=Path("capture.h264"),
        captured_at=datetime.fromisoformat("2026-10-10T10:00:00+02:00"),
    )

    assert event.metadata()["captured_at"] == "2026-10-10T08:00:00+00:00"
