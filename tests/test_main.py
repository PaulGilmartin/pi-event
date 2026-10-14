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

    monkeypatch.setattr(main, "PiCamera", FakeCamera)
    monkeypatch.setattr(main, "MotionDetector", TriggeredDetector)
    monkeypatch.setattr(main, "EventHttpClient", FakeEventHttpClient)
    monkeypatch.setattr(main, "sleep", interrupt_after_one_iteration)
    monkeypatch.setattr(main, "EVENTS_DIRECTORY", Path("test-events"))
    FakeEventHttpClient.sent_events.clear()

    main.main()

    assert len(FakeEventHttpClient.sent_events) == 1
    output = capsys.readouterr().out
    assert "Motion detected! Capturing event..." in output
    assert "Event " in output
    assert "saved to test-events/" in output
    assert "Stopped." in output
