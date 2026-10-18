from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import Self
from uuid import UUID

import httpx
import numpy as np
import pytest

from pi_event import main
from pi_event.events import CameraEvent
from pi_event.imaging import MotionResult


class FileWritingCamera:
    def __init__(self) -> None:
        self.output_directory: Path | None = None

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
        self.output_directory = output_directory / str(event_id)
        self.output_directory.mkdir(parents=True)
        image_path = self.output_directory / "image.jpg"
        video_path = self.output_directory / "video.h264"
        image_path.write_bytes(b"image")
        video_path.write_bytes(b"video")
        return image_path, video_path


class TriggeredDetector:
    def process(
        self,
        frame: np.ndarray[tuple[int, int], np.dtype[np.uint8]],
    ) -> MotionResult:
        return MotionResult(score=0.5, triggered=True)


class FailingEventHttpClient:
    def __init__(self, server_url: str, api_key: str) -> None:
        pass

    def send(self, event: CameraEvent) -> None:
        request = httpx.Request("POST", "http://example.test/api/event")
        raise httpx.ConnectError("connection failed", request=request)


def test_upload_failure_retains_event_and_continues(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    sleeps: Iterator[None] = iter([None])

    def interrupt_after_one_iteration(seconds: float) -> None:
        next(sleeps)
        raise KeyboardInterrupt

    camera = FileWritingCamera()
    monkeypatch.setattr(main, "PiCamera", lambda: camera)
    monkeypatch.setattr(main, "MotionDetector", TriggeredDetector)
    monkeypatch.setattr(main, "EventHttpClient", FailingEventHttpClient)
    monkeypatch.setattr(main, "sleep", interrupt_after_one_iteration)
    monkeypatch.setattr(main, "EVENTS_DIRECTORY", tmp_path / "events")
    monkeypatch.setenv("PI_EVENT_SERVER_URL", "http://example.test/api/event")
    monkeypatch.setenv("PI_EVENT_API_KEY", "test-api-key")

    main.main()

    assert camera.output_directory is not None
    assert (camera.output_directory / "image.jpg").exists()
    assert (camera.output_directory / "video.h264").exists()
    output = capsys.readouterr().out
    assert "upload failed" in output
    assert "retained at" in output
