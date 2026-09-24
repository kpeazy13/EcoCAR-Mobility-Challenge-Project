# EcoCAR honk detection

Binary audio classification on NVIDIA Jetson Orin, with Raspberry Pi seat feedback and a tablet display. 

```text
Microphone → Jetson: 16 kHz mono → 1 s waveform → log-mel CNN
             → probability every 200 ms → confirmation / hysteresis
             → authenticated UDP over local Ethernet/Wi-Fi
             → Raspberry Pi → motor-driver GPIO inputs → seat motors
                            → HTTP status / tablet browser
```

| Module | Responsibility |
| --- | --- |
| `ecocar/model.py` | 64-band log-mel frontend, compact CNN, binary output |
| `ecocar/data.py`, `train.py` | Resampling, augmentation, group split checks, training, threshold selection, held-out metrics |
| `ecocar/jetson.py` | Microphone capture, CUDA inference, bounded queue, audio gap handling, network output |
| `ecocar/protocol.py` | HMAC authentication, timestamp validation, confirmation and hysteresis |
| `ecocar/pi.py` | UDP receiver, replay rejection, connection monitoring, tablet server |
| `ecocar/haptics.py` | Bounded GPIO pulses, cooldown, shutdown cleanup, dry-run mode |
| `ecocar/static/tablet.html` | Honk alert, listening, audio fault and disconnected states |

## Notification demo — no hardware or ML dependencies

Python 3.10+ is required. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -c "import secrets; print(secrets.token_hex(32))"
export ECOCAR_SECRET='<paste-generated-key>'
python -m ecocar.pi --bind 127.0.0.1
```

In another terminal, activate the environment and set the same key, then run:

```bash
python -m ecocar.demo --seconds 30
```

Open `http://127.0.0.1:8080`. An alert appears after three seconds. GPIO is disabled by default. To use a tablet on the same private network, omit `--bind 127.0.0.1` and open `http://<pi-ip>:8080`. Keep the screen awake using the tablet's kiosk/display settings.

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` and set the key with `$env:ECOCAR_SECRET = '<paste-generated-key>'`. The `python -m ecocar...` commands work unchanged for the dry-run demo.

## Recordings and training

1. Collect representative recordings with the intended microphone placement: horns at different distances, road speeds, wind, HVAC, speech, music, sirens, brakes, motorcycles and silence. Include short and sustained honks and difficult negatives.
2. Export labeled **one-second windows**, ideally every 200 ms to match deployment. Label `1` if a honk is audible anywhere in the window, otherwise `0`. Shorter clips (at least 0.1 s) are zero-padded. Long clips are rejected instead of randomly cropping away an event.
3. Make a CSV with `path,label,split,group`; see `examples/manifest.csv`. Paths are relative to the CSV. Assign whole recording sessions to `train`, `val` or `test` before windowing (e.g. 70/15/15 by session). Each split must contain both classes. All windows from a source/session must share a group. The example references placeholder files, not an included dataset.
4. Install training dependencies on a workstation and train:

```bash
python -m pip install -e '.[train]'
python -m ecocar.train examples/manifest.csv --epochs 30 --output artifacts
```

Outputs: `artifacts/honk.pt` contains weights, threshold and audio metadata. `artifacts/metrics.json` contains precision, recall, F1 and confusion counts. The threshold and best epoch are selected using validation F1 only. Test data is evaluated after selection. CUDA results need not be bitwise reproducible despite the fixed seed. Group leakage checks depend on correct group labels; renamed duplicate recordings cannot be identified automatically.

F1 is a baseline selection objective, not an automotive acceptance criterion. Evaluate event recall, false alerts per driving hour, and onset-to-tablet/motor latency on continuous held-out drives with confirmation and cooldown enabled. No accuracy or latency target has been demonstrated yet.

## Jetson Orin

Install JetPack, then the NVIDIA PyTorch build matching **your JetPack and Python versions** using the [NVIDIA installation guide](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) and [compatibility table](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html). Avoid replacing it with the generic workstation `.[train]` installation. This reference runtime uses PyTorch CUDA; TensorRT conversion is not included.

```bash
# Expose NVIDIA PyTorch if installed at system level:
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
sudo apt-get install libportaudio2 libsndfile1
python -m pip install -e '.[audio]'
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m sounddevice
export ECOCAR_SECRET='<shared-generated-key>'
python -m ecocar.jetson --pi 192.168.1.20 --model artifacts/honk.pt --microphone 0
```

Copy the trained checkpoint into `artifacts/` first. The microphone must support 16 kHz mono through PortAudio; adjust the device index/name. Training accepts other sample rates, but live capture requests 16 kHz directly. CPU testing uses `--device cpu`; there is no silent CUDA fallback.

The first prediction requires one second of continuous audio. By default two consecutive positive windows are required. The release threshold is 0.15 below the learned threshold, with a floor of half that threshold to avoid a permanently latched alert. Short honks can be missed by confirmation; tune `--confirmations` against continuous recordings. Capture/inference older than 500 ms is discarded and a fresh window required. Logs report inference time, not total system latency. Capture uses the [sounddevice input-stream API](https://python-sounddevice.readthedocs.io/en/0.5.3/api/streams.html).

## Raspberry Pi and tablet

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[pi]'
export ECOCAR_SECRET='<shared-generated-key>'
export GPIOZERO_PIN_FACTORY=lgpio
python -m ecocar.pi --jetson-ip 192.168.1.10 --pins 17,27
# Enable GPIO only after motor-driver bench verification:
python -m ecocar.pi --jetson-ip 192.168.1.10 --pins 17,27 --hardware
```

Default pins are placeholders using [gpiozero BCM numbering](https://gpiozero.readthedocs.io/en/stable/api_output.html). Each pin connects to an active-high input on a **3.3 V logic-compatible motor driver**, never directly to a motor. Choose a driver and separate fused supply matching the motor voltage, running current and stall current. Use the required common signal ground, appropriate transient suppression, and hardware pull-downs on enable inputs. LRA motors needing dedicated waveform control require another adapter; this implementation targets on/off driver inputs such as those for ERM motors.

All configured motors activate together because binary classification provides no direction. Default pulses last 250 ms, at most once every two seconds while a honk remains active. Outputs initialize off and switch off on normal shutdown and handled exceptions. Software cannot guarantee off after an OS freeze or sudden power loss; use a hardware pulse timeout/enable circuit if required. Bench-test before seat installation.

The tablet polls `/api/status` every 200 ms after the previous request completes, with a 1.2 s request timeout. The Pi reports a disconnected detector after two seconds without an accepted message. No history or raw audio is stored. The probability is an uncalibrated model score.

## Network and service deployment

Use a private local network. Allow UDP 5005 from Jetson to Pi and TCP 8080 from tablet to Pi. The read-only tablet server has no TLS or login and is intended for that private network. Detection messages use HMAC, session UUIDs, sequences and timestamps to reject tampering, stale messages and replay. Synchronize both clocks within three seconds using NTP or a local time source. UDP is best effort: repeated state updates tolerate some loss, but dropped packets can delay or miss a short event.

`deploy/` provides systemd templates. Create an `ecocar` service account, put the project and virtual environment at `/opt/ecocar`, and place the generated key and Pi address in `/etc/ecocar.env` using the example. Restrict that file to root/service administration. Adjust OS groups (`audio`, `video`, `render` on Jetson; `gpio` on Pi), addresses, microphone and pins. Install the relevant service on each device:

```bash
sudo cp deploy/ecocar-pi.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ecocar-pi
sudo journalctl -u ecocar-pi -f
```

Substitute `ecocar-jetson` on Jetson. The Pi template starts in dry-run mode; add `--hardware` to `ExecStart` after wiring verification. On Jetson termination, the Pi detects missing updates and its independently timed pulses stop. Service files have not been tested on physical devices.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests cover authentication, malformed packets, staleness, replay, hysteresis, connection timeouts, motor cleanup/failure and a real local UDP → Pi → HTTP integration. Installing `.[train]` also enables preprocessing, gradients, checkpoint restoration, split leakage and one-epoch synthetic training tests. With `.[train,audio]`, a simulated microphone exercises the Jetson inference loop through signed UDP output.

Implementation verification: all 11 tests passed on the development machine using CPU inference. Synthetic fixtures are created in temporary directories and are not distributed as trained honk weights. No Jetson CUDA, physical microphone, GPIO/motor, or tablet browser visual check has been performed. These checks do not establish field detection quality or hardware compatibility.
