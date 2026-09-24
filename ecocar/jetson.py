"""Bounded audio capture and inference. Drop backlog instead of reporting old sound."""
import argparse
import logging
import queue
import socket
import threading
import time
import uuid

from .protocol import Gate, encode, secret_from_env


def main():
    import numpy as np
    import sounddevice as sd
    import torch
    from .model import HonkNet, SAMPLE_RATE, WINDOW_SAMPLES, HOP_SAMPLES

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/honk.pt")
    parser.add_argument("--pi", required=True, help="Pi IP/hostname")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--microphone", help="PortAudio device index or name")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confirmations", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    secret = secret_from_env()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable. Install NVIDIA's JetPack-compatible PyTorch or use --device cpu")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    for key, expected in dict(schema=1, sample_rate=SAMPLE_RATE, window_samples=WINDOW_SAMPLES,
                              hop_samples=HOP_SAMPLES).items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"Incompatible model metadata: {key}")
    model = HonkNet().to(args.device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    gate = Gate(checkpoint["threshold"], args.confirmations)
    with torch.inference_mode():
        model(torch.zeros(1, WINDOW_SAMPLES, device=args.device))
    blocks = queue.Queue(maxsize=5)
    fault = threading.Event()

    def capture(indata, frames, timing, status):
        if status or frames != HOP_SAMPLES:
            fault.set()
        try:
            blocks.put_nowait((time.monotonic(), indata[:, 0].copy()))
        except queue.Full:
            fault.set()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    destination = (socket.gethostbyname(args.pi), args.port)
    session = str(uuid.uuid4())
    sequence = 0

    def send(probability=0.0, honk=False, healthy=False):
        nonlocal sequence
        sequence += 1
        try:
            sock.sendto(encode(secret, session, sequence, probability, honk, healthy), destination)
        except OSError:
            logging.exception("Pi send failed")

    mic = int(args.microphone) if args.microphone and args.microphone.isdecimal() else args.microphone
    ring = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    filled = 0
    last_log = 0.0
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=HOP_SAMPLES, device=mic, callback=capture):
            logging.info("Listening at %s Hz; threshold %.2f", SAMPLE_RATE, gate.high)
            while True:
                try:
                    captured, block = blocks.get(timeout=1)
                except queue.Empty:
                    send()
                    fault.set()
                    logging.warning("Microphone stalled")
                    continue
                if fault.is_set() or time.monotonic() - captured > 0.5:
                    fault.clear()
                    while True:
                        try:
                            blocks.get_nowait()
                        except queue.Empty:
                            break
                    filled = 0
                    ring.fill(0)
                    gate = Gate(checkpoint["threshold"], args.confirmations)
                    send()
                    logging.warning("Audio gap/backlog; rebuilding continuous window")
                    continue
                if len(block) != HOP_SAMPLES or not np.isfinite(block).all():
                    fault.set()
                    send()
                    continue
                ring[:-HOP_SAMPLES] = ring[HOP_SAMPLES:]
                ring[-HOP_SAMPLES:] = block
                filled = min(WINDOW_SAMPLES, filled + HOP_SAMPLES)
                if filled < WINDOW_SAMPLES:
                    send()
                    continue
                start = time.monotonic()
                with torch.inference_mode():
                    probability = float(model(torch.from_numpy(ring.copy()).unsqueeze(0)
                                              .to(args.device)).sigmoid().item())
                if not np.isfinite(probability) or time.monotonic() - captured > 0.5 or fault.is_set():
                    fault.set()
                    send()
                    continue
                active = gate.update(probability)
                send(probability, active, True)
                if time.monotonic() - last_log > 5:
                    logging.info("p=%.3f honk=%s inference_ms=%.1f", probability, active,
                                 (time.monotonic() - start) * 1000)
                    last_log = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        send()
        sock.close()


if __name__ == "__main__":
    main()
