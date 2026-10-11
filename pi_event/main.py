"""Command-line entry point."""

from time import sleep

from pi_event.imaging import MotionDetector, PiCamera

DETECTION_INTERVAL_SECONDS = 0.2


def main() -> None:
    """Print motion scores from the Raspberry Pi camera until interrupted."""
    detector = MotionDetector()

    try:
        with PiCamera() as camera:
            while True:
                frame = camera.capture_detection_frame()
                result = detector.process(frame)
                if result.triggered:
                    print("Motion detected!", flush=True)
                sleep(DETECTION_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nStopped.")
