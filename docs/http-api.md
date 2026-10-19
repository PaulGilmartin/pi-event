# HTTP event API

`pi-event` sends each camera event as one authenticated
`multipart/form-data` POST request to `POST /api/event`.

## Request

The request includes:

```text
Authorization: Bearer <device API key>
```

The multipart body contains:

- `metadata`: JSON with content type `application/json`
- `image`: JPEG file with content type `image/jpeg`
- `video`: raw H.264 file with content type `video/h264`

Example metadata:

```json
{
  "captured_at": "2026-10-10T08:00:00+00:00"
}
```

`captured_at` is always sent in UTC. The backend identifies the device from
the API key rather than trusting device metadata supplied by the client.

## Response

Any HTTP status from 200 through 299 means the server accepted the event.
Every other status is treated as an upload failure. Missing or invalid API
keys should return `401 Unauthorized`.
