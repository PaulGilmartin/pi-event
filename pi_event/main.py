"""Command-line entry point."""

import os
from pathlib import Path
from shutil import rmtree
from time import sleep
from uuid import uuid4

import httpx

from pi_event.events import CameraEvent
from pi_event.http_client import EventHttpClient, EventUploadError
from pi_event.imaging import MotionDetector, PiCamera

DETECTION_INTERVAL_SECONDS = 0.2
EVENTS_DIRECTORY = Path("events")


def required_environment(name: str) -> str:
    """Return a required non-empty environment variable."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def main() -> None:
    """Print motion scores from the Raspberry Pi camera until interrupted."""
    detector = MotionDetector()
    client = EventHttpClient(
        server_url=required_environment("PI_EVENT_SERVER_URL"),
        api_key=required_environment("PI_EVENT_API_KEY"),
    )
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
                        image_path=image_path,
                        video_path=video_path,
                        event_id=event_id,
                    )
                    print("Sending event", flush=True)
                    try:
                        client.send(event)
                    except (EventUploadError, httpx.HTTPError) as error:
                        print(
                            f"Event {event.event_id} upload failed; "
                            f"retained at {image_path.parent}: {error}",
                            flush=True,
                        )
                    else:
                        rmtree(image_path.parent)
                        print(
                            f"Event {event.event_id} sent successfully",
                            flush=True,
                        )
                sleep(DETECTION_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nStopped.")
