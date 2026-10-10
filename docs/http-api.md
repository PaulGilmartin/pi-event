# HTTP event API

`pi-event` sends each camera event as one `multipart/form-data` POST request.

## Request

The client posts to the local-network URL passed to `EventHttpClient`.

The multipart body contains:

- `metadata`: JSON with content type `application/json`
- `image`: JPEG file with content type `image/jpeg`
- `video`: raw H.264 file with content type `video/h264`

Example metadata:

```json
{
  "event_id": "72279045-6017-44c9-b5a5-43829a70d27b",
  "event_type": "motion",
  "device_id": "garden-pi",
  "captured_at": "2026-10-10T08:00:00+00:00",
  "motion_score": 0.42
}
```

`captured_at` is always sent in UTC. `motion_score` is between 0 and 1 and
represents the proportion of the detection area that changed.

## Response

Any HTTP status from 200 through 299 means the server accepted the event.
Every other status is treated as an upload failure.
