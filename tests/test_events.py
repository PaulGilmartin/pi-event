from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from pi_event.events import CameraEvent


def test_camera_event_generates_identity_and_serializes_metadata() -> None:
    event = CameraEvent(
        image_path=Path("capture.jpg"),
        video_path=Path("capture.h264"),
    )

    metadata = event.metadata()

    assert UUID(metadata["event_id"]) == event.event_id
    assert metadata == {
        "schema_version": 1,
        "event_id": str(event.event_id),
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
