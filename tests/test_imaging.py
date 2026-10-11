import sys
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar
from uuid import UUID

import numpy as np
import pytest

from pi_event.imaging import CAPTURE_SIZE, DETECTION_SIZE, MotionDetector, PiCamera


def frame(value: int = 0) -> np.ndarray[tuple[int, int], np.dtype[np.uint8]]:
    return np.full((10, 10), value, dtype=np.uint8)


class FakePicamera2:
    instances: ClassVar[list["FakePicamera2"]] = []

    def __init__(self) -> None:
        self.configuration: dict[str, Any] | None = None
        self.configured_with: dict[str, Any] | None = None
        self.started = False
        self.stopped = False
        self.closed = False
        self.instances.append(self)

    def create_video_configuration(self, **configuration: Any) -> dict[str, Any]:
        self.configuration = configuration
        return configuration

    def configure(self, configuration: dict[str, Any]) -> None:
        self.configured_with = configuration

    def start(self) -> None:
        self.started = True

    def capture_array(self, stream: str) -> np.ndarray[Any, np.dtype[np.uint8]]:
        assert stream == "lores"
        height = DETECTION_SIZE[1]
        width = DETECTION_SIZE[0]
        return np.vstack(
            [
                np.full((height, width), 42, dtype=np.uint8),
                np.full((height // 2, width), 200, dtype=np.uint8),
            ]
        )

    def capture_file(self, path: str, *, name: str) -> None:
        assert name == "main"
        Path(path).write_bytes(b"jpeg-data")

    def start_encoder(self, encoder: Any, *, output: Any, name: str) -> None:
        assert name == "main"
        output.path.write_bytes(b"h264-data")

    def stop_encoder(self, encoder: Any) -> None:
        pass

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[PiCamera, FakePicamera2]:
    FakePicamera2.instances.clear()
    module = ModuleType("picamera2")
    module.Picamera2 = FakePicamera2  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "picamera2", module)
    camera = PiCamera()
    return camera, FakePicamera2.instances[-1]


def test_pi_camera_configures_detection_and_capture_streams(
    fake_camera: tuple[PiCamera, FakePicamera2],
) -> None:
    _, picamera2 = fake_camera
    assert picamera2.configuration == {
        "main": {"size": CAPTURE_SIZE},
        "lores": {"size": DETECTION_SIZE, "format": "YUV420"},
    }
    assert picamera2.configured_with == picamera2.configuration


def test_pi_camera_returns_luminance_plane(
    fake_camera: tuple[PiCamera, FakePicamera2],
) -> None:
    camera, _ = fake_camera
    camera.start()

    captured = camera.capture_detection_frame()

    assert captured.shape == (DETECTION_SIZE[1], DETECTION_SIZE[0])
    assert np.all(captured == 42)


def test_pi_camera_context_manager_releases_camera(
    fake_camera: tuple[PiCamera, FakePicamera2],
) -> None:
    camera, picamera2 = fake_camera

    with camera:
        assert picamera2.started is True

    assert picamera2.stopped is True
    assert picamera2.closed is True


def test_pi_camera_captures_event_media(
    fake_camera: tuple[PiCamera, FakePicamera2],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeEncoder:
        pass

    class FakeOutput:
        def __init__(self, path: str) -> None:
            self.path = Path(path)

    module = ModuleType("picamera2.encoders")
    module.H264Encoder = FakeEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "picamera2.encoders", module)
    module = ModuleType("picamera2.outputs")
    module.FileOutput = FakeOutput  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "picamera2.outputs", module)
    monkeypatch.setattr("pi_event.imaging.sleep", lambda seconds: None)

    camera, _ = fake_camera
    camera.start()
    event_id = UUID("72279045-6017-44c9-b5a5-43829a70d27b")

    image_path, video_path = camera.capture_event_media(event_id, tmp_path)

    assert image_path.read_bytes() == b"jpeg-data"
    assert video_path.read_bytes() == b"h264-data"
    assert image_path.parent == tmp_path / str(event_id)


def test_first_frame_establishes_reference() -> None:
    detector = MotionDetector()

    result = detector.process(frame())

    assert result.score == 0
    assert result.triggered is False


def test_score_is_proportion_of_changed_pixels() -> None:
    detector = MotionDetector()
    detector.process(frame())
    changed = frame()
    changed[:2, :] = 100

    result = detector.process(changed)

    assert result.score == pytest.approx(0.2)


def test_sustained_change_triggers_once() -> None:
    detector = MotionDetector()
    detector.process(frame())
    changed = frame(100)

    results = [detector.process(changed) for _ in range(4)]

    assert [result.triggered for result in results] == [False, False, True, False]


def test_quiet_frame_resets_consecutive_change() -> None:
    detector = MotionDetector()
    quiet = frame()
    changed = frame(100)
    detector.process(quiet)

    detector.process(changed)
    detector.process(changed)
    detector.process(quiet)
    detector.process(changed)
    result = detector.process(changed)

    assert result.triggered is False


def test_gradual_quiet_change_updates_reference() -> None:
    detector = MotionDetector()
    detector.process(frame())

    for brightness in range(1, 21):
        result = detector.process(frame(brightness))

    assert result.score == 0
    assert result.triggered is False


def test_subtraction_does_not_wrap_unsigned_values() -> None:
    detector = MotionDetector()
    detector.process(frame(255))

    result = detector.process(frame(0))

    assert result.score == 1


def test_rejects_non_grayscale_frame() -> None:
    detector = MotionDetector()
    colour_frame = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="two-dimensional"):
        detector.process(colour_frame)


def test_reset_uses_next_frame_as_new_reference() -> None:
    detector = MotionDetector()
    detector.process(frame())
    detector.process(frame(100))

    detector.reset()
    result = detector.process(frame(100))

    assert result.score == 0
    assert result.triggered is False
