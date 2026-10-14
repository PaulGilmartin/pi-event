from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import ClassVar, Self
from uuid import UUID

import numpy as np
import pytest

from pi_event import main
from pi_event.events import CameraEvent
from pi_event.imaging import MotionResult


class FakeCamera:
    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass

    def capture_detection_frame(
        self,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.uint8]]:
        return np.zeros((10, 10), dtype=np.uint8)

    def capture_event_media(
        self,
        event_id: UUID,
        output_directory: Path,
    ) -> tuple[Path, Path]:
        event_directory = output_directory / str(event_id)
        return event_directory / "image.jpg", event_directory / "video.h264"


class FakeEventHttpClient:
    sent_events: ClassVar[list[CameraEvent]] = []

    def __init__(self, server_url: str) -> None:
        assert server_url == "http://192.168.8.48:8000/events"

    def send(self, event: CameraEvent) -> None:
        self.sent_events.append(event)


def test_main_prints_motion_result_until_interrupted(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    sleeps: Iterator[None] = iter([None])

    def interrupt_after_one_iteration(seconds: float) -> None:
        assert seconds == main.DETECTION_INTERVAL_SECONDS
        next(sleeps)
        raise KeyboardInterrupt

    class TriggeredDetector:
        def process(
            self,
            frame: np.ndarray[tuple[int, int], np.dtype[np.uint8]],
        ) -> MotionResult:
            return MotionResult(score=0.5, triggered=True)

    class FileWritingCamera(FakeCamera):
        def capture_event_media(
            self,
            event_id: UUID,
            output_directory: Path,
        ) -> tuple[Path, Path]:
            event_directory = output_directory / str(event_id)
            event_directory.mkdir(parents=True)
            image_path = event_directory / "image.jpg"
            video_path = event_directory / "video.h264"
            image_path.write_bytes(b"image")
            video_path.write_bytes(b"video")
            return image_path, video_path

    monkeypatch.setattr(main, "PiCamera", FileWritingCamera)
    monkeypatch.setattr(main, "MotionDetector", TriggeredDetector)
    monkeypatch.setattr(main, "EventHttpClient", FakeEventHttpClient)
    monkeypatch.setattr(main, "sleep", interrupt_after_one_iteration)
    events_directory = tmp_path / "test-events"
    monkeypatch.setattr(main, "EVENTS_DIRECTORY", events_directory)
    FakeEventHttpClient.sent_events.clear()

    main.main()

    assert len(FakeEventHttpClient.sent_events) == 1
    assert list(events_directory.iterdir()) == []
    output = capsys.readouterr().out
    assert "Motion detected! Capturing event..." in output
    assert "Event " in output
    assert "sent successfully" in output
    assert "Stopped." in output
