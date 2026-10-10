# pi-event

`pi-event` is a Python application for Raspberry Pi camera events. It is intended to detect motion, capture a photo and short video, and send the event to an HTTP server.

The project is under active development.

## How it works

The camera keeps a small, low-resolution video stream running. Using small images makes it quicker and cheaper for the Raspberry Pi to look for movement.

The application compares each image from the stream with an earlier image. It counts how many pixels have changed by more than a chosen amount. If enough pixels change for several images in a row, the application treats that as an event, such as a bird landing.

When an event is detected, the camera takes a full-resolution photo and records a short video. The application then sends the event details, photo, and video to another server in one HTTP request.

The detection area, amount of change required, and delay between events will be configurable. This helps ignore movement outside the feeder, small lighting changes, and repeated triggers from the same bird.

## Requirements

- Raspberry Pi running Raspberry Pi OS
- A supported Raspberry Pi camera
- Python 3.11 or newer
- Picamera2

Picamera2 is a Python package, but it depends on Raspberry Pi OS camera libraries. Run the following commands on the Raspberry Pi to install it with `apt`:

```bash
sudo apt update
sudo apt install python3-picamera2
```

Confirm that Python can import it:

```bash
python3 -c "from picamera2 import Picamera2; print('Picamera2 is available')"
```

## Install pi-event

Create a virtual environment that can access the system-installed Picamera2 package, then install `pi-event`:

```bash
git clone https://github.com/OWNER/pi-event.git
cd pi-event

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

Replace `OWNER` with the GitHub account or organisation that hosts the repository.

The `--system-site-packages` option is important: without it, the virtual environment cannot see the Picamera2 package installed by Raspberry Pi OS.

The first version sends events to an HTTP endpoint on the local network without authentication. See [`docs/http-api.md`](docs/http-api.md) for the multipart request sent to the server.

## Development

Development can be done on macOS using `uv`. Camera-independent code and tests run locally; camera integration must be tested on a Raspberry Pi.

```bash
brew install uv
uv sync
uv run pytest
```

`uv sync` installs the locked runtime and development dependencies into `.venv`. Picamera2 is deliberately not installed through `uv` or listed as a regular cross-platform Python dependency. The application should keep camera access isolated so development and CI do not require Raspberry Pi hardware.
