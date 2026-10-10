"""HTTP transport for captured camera events."""

import json
from functools import cached_property
from typing import BinaryIO, TypeAlias

import httpx

from pi_event.events import CameraEvent

MultipartValue: TypeAlias = (
    tuple[str | None, BinaryIO | bytes | str, str | None]
)


class EventUploadError(RuntimeError):
    """Raised when a server rejects an event upload."""


class EventHttpClient:
    """Send camera events to an HTTP server as multipart requests.

    Each request contains the event metadata as JSON, the JPEG image bytes,
    and the H.264 video bytes. Local file paths are used only to open the
    captured media; the paths themselves are not sent to the server.
    """

    def __init__(self, server_url: str) -> None:
        if not server_url.strip():
            raise ValueError("server_url must not be empty")
        self._server_url = server_url

    @cached_property
    def client(self) -> httpx.Client:
        """Create and reuse the HTTP client on first use."""
        return httpx.Client()

    def send(self, event: CameraEvent) -> None:
        """Send one event and raise if the server does not accept it."""
        with (
            event.image_path.open("rb") as image,
            event.video_path.open("rb") as video,
        ):
            files = self._multipart_files(event, image, video)

            response = self.client.post(
                self._server_url,
                files=files,
            )

        if not 200 <= response.status_code < 300:
            raise EventUploadError(
                f"server rejected event {event.event_id} "
                f"with status {response.status_code}"
            )

    @staticmethod
    def _multipart_files(
        event: CameraEvent,
        image: BinaryIO,
        video: BinaryIO,
    ) -> dict[str, MultipartValue]:
        return {
            "metadata": (
                None,
                json.dumps(event.metadata()),
                "application/json",
            ),
            "image": (
                event.image_path.name,
                image,
                "image/jpeg",
            ),
            "video": (
                event.video_path.name,
                video,
                "video/h264",
            ),
        }
