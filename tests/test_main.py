from collections.abc import Iterator
from types import TracebackType
from typing import Self

import numpy as np
import pytest

from pi_event import main


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


def test_main_prints_motion_result_until_interrupted(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sleeps: Iterator[None] = iter([None])

    def interrupt_after_one_iteration(seconds: float) -> None:
        assert seconds == main.DETECTION_INTERVAL_SECONDS
        next(sleeps)
        raise KeyboardInterrupt

    monkeypatch.setattr(main, "PiCamera", FakeCamera)
    monkeypatch.setattr(main, "sleep", interrupt_after_one_iteration)

    main.main()

    output = capsys.readouterr().out
    assert "motion_score=0.000 triggered=False" in output
    assert "Stopped." in output
