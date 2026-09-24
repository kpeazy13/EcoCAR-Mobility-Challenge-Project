"""Run the real inference loop with a simulated microphone and a local UDP receiver."""
import importlib.util
import os
import queue
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HAS_AUDIO = all(importlib.util.find_spec(m) for m in ("torch", "numpy", "sounddevice"))


@unittest.skipUnless(HAS_AUDIO, "Install .[train,audio] for Jetson simulation")
class JetsonTests(unittest.TestCase):
    def test_capture_to_prediction_to_signed_udp(self):
        import numpy as np
        import torch
        from ecocar import jetson
        from ecocar.model import HonkNet, SAMPLE_RATE, WINDOW_SAMPLES, HOP_SAMPLES
        from ecocar.protocol import decode
        torch.set_num_threads(2)
        secret = "simulation-only-01234567890123456789"
        callback = None

        class SimulatedStream:
            def __init__(self, **kwargs):
                nonlocal callback
                callback = kwargs["callback"]
            def __enter__(self): return self
            def __exit__(self, *args): pass

        class CaptureQueue(queue.Queue):
            count = 0
            def get(self, block=True, timeout=None):
                if block:
                    self.count += 1
                    if self.count > 8:
                        raise KeyboardInterrupt
                    callback(np.zeros((HOP_SAMPLES, 1), dtype=np.float32), HOP_SAMPLES, None, False)
                return super().get(block, timeout)

        with tempfile.TemporaryDirectory() as directory, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
            receiver.bind(("127.0.0.1", 0))
            receiver.settimeout(0.1)
            path = Path(directory) / "synthetic.pt"
            model = HonkNet()
            # This fixture emits p=0.5; it is not a trained honk recognizer.
            for parameter in model.parameters():
                parameter.data.zero_()
            torch.save(dict(state_dict=model.state_dict(), threshold=0.4, schema=1,
                            sample_rate=SAMPLE_RATE, window_samples=WINDOW_SAMPLES,
                            hop_samples=HOP_SAMPLES), path)
            argv = ["jetson", "--model", str(path), "--pi", "127.0.0.1", "--port",
                    str(receiver.getsockname()[1]), "--device", "cpu"]
            with patch.dict(os.environ, ECOCAR_SECRET=secret), patch("sys.argv", argv), \
                    patch("sounddevice.InputStream", SimulatedStream), patch("ecocar.jetson.queue.Queue", CaptureQueue):
                jetson.main()
            messages = []
            while True:
                try:
                    packet, _ = receiver.recvfrom(2048)
                    messages.append(decode(packet, secret.encode()))
                except socket.timeout:
                    break
            self.assertEqual(len(messages), 9)
            self.assertTrue(all(not m["healthy"] for m in messages[:4]))
            self.assertTrue(messages[4]["healthy"])
            self.assertFalse(messages[4]["honk"])
            self.assertTrue(messages[5]["honk"])
            self.assertFalse(messages[-1]["healthy"])
            self.assertEqual([m["seq"] for m in messages], list(range(1, 10)))


if __name__ == "__main__":
    unittest.main()
