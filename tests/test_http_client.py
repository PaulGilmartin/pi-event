import json
from pathlib import Path
from uuid import UUID

import httpx
import pytest

from pi_event.events import CameraEvent
from pi_event.http_client import (
    EventHttpClient,
    EventUploadError,
)


def make_event(tmp_path: Path) -> CameraEvent:
    image_path = tmp_path / "capture.jpg"
    video_path = tmp_path / "capture.h264"
    image_path.write_bytes(b"jpeg-data")
    video_path.write_bytes(b"h264-data")

    return CameraEvent(
        device_id="garden-pi",
        motion_score=0.42,
        image_path=image_path,
        video_path=video_path,
        event_id=UUID("72279045-6017-44c9-b5a5-43829a70d27b"),
    )


def test_send_posts_multipart_event(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []

    def accept(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201)

    client = httpx.Client(transport=httpx.MockTransport(accept))
    event_client = EventHttpClient("http://192.168.8.100/events")
    event_client.__dict__["client"] = client

    event_client.send(make_event(tmp_path))

    request = requests[0]
    body = request.content
    assert request.method == "POST"
    assert request.url == "http://192.168.8.100/events"
    assert "Authorization" not in request.headers
    assert "Idempotency-Key" not in request.headers
    assert b'name="metadata"' in body
    assert b"application/json" in body
    assert b'"device_id": "garden-pi"' in body
    assert b'name="image"; filename="capture.jpg"' in body
    assert b"image/jpeg" in body
    assert b"jpeg-data" in body
    assert b'name="video"; filename="capture.h264"' in body
    assert b"video/h264" in body
    assert b"h264-data" in body


def test_send_raises_when_server_rejects_event(tmp_path: Path) -> None:
    def reject(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request)

    event_client = EventHttpClient("http://192.168.8.100/events")
    event_client.__dict__["client"] = httpx.Client(
        transport=httpx.MockTransport(reject)
    )

    with pytest.raises(EventUploadError, match="status 503"):
        event_client.send(make_event(tmp_path))


def test_metadata_is_valid_json(tmp_path: Path) -> None:
    captured_metadata: dict[str, object] = {}

    def inspect(request: httpx.Request) -> httpx.Response:
        body = request.content.decode()
        json_start = body.index("{")
        json_end = body.index("}", json_start) + 1
        captured_metadata.update(json.loads(body[json_start:json_end]))
        return httpx.Response(200)

    event_client = EventHttpClient("http://192.168.8.100/events")
    event_client.__dict__["client"] = httpx.Client(
        transport=httpx.MockTransport(inspect)
    )

    event_client.send(make_event(tmp_path))

    assert captured_metadata["event_type"] == "motion"
    assert captured_metadata["motion_score"] == 0.42
