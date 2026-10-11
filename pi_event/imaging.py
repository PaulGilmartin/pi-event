"""Image capture and motion detection."""

from dataclasses import dataclass
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

GrayscaleFrame = NDArray[np.uint8]

PIXEL_THRESHOLD = 25
MOTION_THRESHOLD = 0.05
REQUIRED_MOTION_FRAMES = 3
REFERENCE_LEARNING_RATE = 0.05
DETECTION_SIZE = (640, 480)
CAPTURE_SIZE = (1920, 1080)


@dataclass(frozen=True, slots=True)
class MotionResult:
    """The measured change in one frame and whether it triggered an event."""

    score: float
    triggered: bool


class PiCamera:
    """Provide grayscale detection frames from a Raspberry Pi camera.

    Picamera2 is imported only when this class is created, so the rest of the
    package and its tests can run on a Mac without Raspberry Pi camera software.
    The low-resolution YUV stream is used for detection while the larger main
    stream remains available for photo and video capture later.
    """

    def __init__(self) -> None:
        try:
            from picamera2 import Picamera2  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError(
                "Picamera2 is required to use PiCamera; install "
                "python3-picamera2 on the Raspberry Pi"
            ) from error

        self._camera: Any = Picamera2()
        configuration = self._camera.create_video_configuration(
            main={"size": CAPTURE_SIZE},
            # Picamera2 calls its optional low-resolution stream "lores".
            # We use it for motion detection while keeping the larger main
            # stream available for captured photos and videos.
            lores={"size": DETECTION_SIZE, "format": "YUV420"},
        )
        self._camera.configure(configuration)
        self._started = False

    def start(self) -> None:
        """Start the configured camera streams."""
        if not self._started:
            self._camera.start()
            self._started = True

    def capture_detection_frame(self) -> GrayscaleFrame:
        """Return the luminance plane of the low-resolution stream."""
        if not self._started:
            raise RuntimeError("camera must be started before capturing frames")

        frame = self._camera.capture_array("lores")
        height = DETECTION_SIZE[1]

        # YUV420 stores brightness in the first 480 rows and colour data after
        # them. Select those rows and every column for a 640×480 grayscale frame.
        return np.asarray(frame[:height, :], dtype=np.uint8)

    def close(self) -> None:
        """Stop the camera and release its resources."""
        if self._started:
            self._camera.stop()
            self._started = False
        self._camera.close()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class MotionDetector:
    """Detect sustained visual change in low-resolution grayscale frames.

    The first frame becomes the reference image for an empty or quiet scene.
    Later frames are compared with that reference, and the proportion of pixels
    that changed becomes the motion score. A large change across several frames
    will trigger an event. During quiet periods, the reference will move slowly
    towards recent frames so gradual changes in daylight are not treated as
    motion.

    In production, frames will come from Picamera2's low-resolution camera
    stream. Tests can pass small, artificial NumPy arrays instead.
    """

    def __init__(self) -> None:
        self._reference_frame: NDArray[np.float32] | None = None
        self._consecutive_motion_frames = 0

    def process(self, frame: GrayscaleFrame) -> MotionResult:
        """Process one frame and return its motion score and trigger state."""
        self._validate_frame(frame)
        current = frame.astype(np.float32)

        if self._reference_frame is None:
            self._reference_frame = current.copy()
            return MotionResult(score=0.0, triggered=False)

        # Each image is a matrix whose rows and columns match pixel locations,
        # with each element holding that pixel's brightness. The current and
        # reference matrices have the same shape, so subtraction produces a
        # matrix of brightness changes at matching locations. abs() keeps the
        # size of each change, whether the pixel became lighter or darker.
        difference = np.abs(current - self._reference_frame)

        # Comparing that matrix with the pixel threshold produces a same-sized
        # boolean matrix: True where a pixel changed enough and False elsewhere.
        # Its mean is the proportion of significantly changed pixels because
        # NumPy treats True as 1 and False as 0.
        score = float(np.mean(difference >= PIXEL_THRESHOLD))

        # Treat the change as significant only when enough of the frame changed.
        significant_motion_detected = score >= MOTION_THRESHOLD

        if significant_motion_detected:
            self._consecutive_motion_frames += 1
        else:
            self._consecutive_motion_frames = 0

            # Move every reference pixel a small fraction of the way towards
            # the matching current pixel. Repeating this during quiet frames
            # lets the reference follow gradual changes such as daylight
            # without immediately replacing the whole reference image.
            self._reference_frame += (
                current - self._reference_frame
            ) * REFERENCE_LEARNING_RATE

        triggered = self._consecutive_motion_frames == REQUIRED_MOTION_FRAMES
        return MotionResult(score=score, triggered=triggered)

    def reset(self) -> None:
        """Forget the current scene and use the next frame as the reference."""
        self._reference_frame = None
        self._consecutive_motion_frames = 0

    @staticmethod
    def _validate_frame(frame: GrayscaleFrame) -> None:
        if frame.ndim != 2:
            raise ValueError("frame must be a two-dimensional grayscale image")
        if frame.size == 0:
            raise ValueError("frame must not be empty")
