import numpy as np
import pytest

from pi_event.imaging import MotionDetector


def frame(value: int = 0) -> np.ndarray[tuple[int, int], np.dtype[np.uint8]]:
    return np.full((10, 10), value, dtype=np.uint8)


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
