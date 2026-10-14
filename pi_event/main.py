"""Command-line entry point."""

import socket
from pathlib import Path
from shutil import rmtree
from time import sleep
from uuid import uuid4

from pi_event.events import CameraEvent
from pi_event.http_client import EventHttpClient
from pi_event.imaging import MotionDetector, PiCamera

DETECTION_INTERVAL_SECONDS = 0.2
EVENTS_DIRECTORY = Path("events")


def main() -> None:
    """Print motion scores from the Raspberry Pi camera until interrupted."""
    detector = MotionDetector()
    client = EventHttpClient(server_url="http://192.168.8.48:8000/events")
    try:
        with PiCamera() as camera:
            while True:
                frame = camera.capture_detection_frame()
                result = detector.process(frame)
                if result.triggered:
                    print("Motion detected! Capturing event...", flush=True)
                    event_id = uuid4()
                    image_path, video_path = camera.capture_event_media(
                        event_id,
                        EVENTS_DIRECTORY,
                    )
                    event = CameraEvent(
                        device_id=socket.gethostname(),
                        motion_score=result.score,
                        image_path=image_path,
                        video_path=video_path,
                        event_id=event_id,
                    )
                    print("Sending event", flush=True)
                    client.send(event)
                    rmtree(image_path.parent)
                    print(
                        f"Event {event.event_id} sent successfully",
                        flush=True,
                    )
                sleep(DETECTION_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nStopped.")
