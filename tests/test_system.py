import json
import socket
import threading
import time
import unittest
import urllib.request
import uuid
from http.server import ThreadingHTTPServer

from ecocar.haptics import Haptics
from ecocar.pi import State, make_handler, receive
from ecocar.protocol import Gate, decode, encode

SECRET = b"test-key-only-012345678901234567890"


class ProtocolTests(unittest.TestCase):
    def packet(self, sequence=1, honk=True, healthy=True):
        return encode(SECRET, str(uuid.uuid4()), sequence, 0.9, honk, healthy)

    def test_authentication_and_staleness(self):
        packet = self.packet()
        self.assertTrue(decode(packet, SECRET)["honk"])
        with self.assertRaises(ValueError):
            decode(packet, b"wrong")
        with self.assertRaises(ValueError):
            decode(packet, SECRET, now=time.time() + 10)
        with self.assertRaises(ValueError):
            decode(packet.replace(b"0.9", b"0.1"), SECRET)

    def test_malformed(self):
        for packet in (b"null", b"[]", b"{}", b"invalid", b"x" * 2049,
                       b'{"body":1,"signature":"x"}', b'\xff'):
            with self.subTest(packet=packet[:30]), self.assertRaises(ValueError):
                decode(packet, SECRET)

    def test_gate_confirmation_and_hysteresis(self):
        gate = Gate(0.7)
        self.assertFalse(gate.update(0.9))
        self.assertFalse(gate.update(0.2))
        self.assertFalse(gate.update(0.9))
        self.assertTrue(gate.update(0.8))
        self.assertTrue(gate.update(0.6))
        self.assertFalse(gate.update(0.4))
        low_threshold = Gate(0.05, confirmations=1)
        self.assertTrue(low_threshold.update(0.1))
        self.assertFalse(low_threshold.update(0.01))

    def test_replay_out_of_order_and_timeout(self):
        haptics = Haptics([17])
        self.addCleanup(haptics.close)
        state = State(haptics)
        message = decode(self.packet(sequence=3), SECRET)
        self.assertTrue(state.accept(message, now=100))
        self.assertFalse(state.accept(message, now=101))
        self.assertFalse(state.accept(dict(message, seq=2), now=101))
        self.assertTrue(state.snapshot(now=100)["honk"])
        self.assertFalse(state.snapshot(now=103)["connected"])
        self.assertFalse(state.snapshot(now=103)["honk"])
        self.assertTrue(state.accept(dict(message, seq=4, healthy=False), now=104))
        self.assertFalse(state.snapshot(now=104)["healthy"])

    def test_haptics_cooldown_shutdown_and_pulse(self):
        class FakeOutput:
            active = False
            def on(self): self.active = True
            def off(self): self.active = False
            def close(self): self.active = False
        haptics = Haptics([17], pulse_seconds=0.02)
        output = FakeOutput()
        haptics.devices = [output]
        self.addCleanup(haptics.close)
        self.assertTrue(haptics.trigger())
        haptics.thread.join(timeout=1)
        self.assertFalse(output.active)
        self.assertFalse(haptics.trigger())
        haptics.close()
        self.assertFalse(haptics.trigger())

    def test_haptic_failure_switches_off(self):
        class BrokenOutput:
            active = False
            def on(self):
                self.active = True
                raise RuntimeError("simulated failure")
            def off(self): self.active = False
            def close(self): pass
        haptics = Haptics([17])
        self.addCleanup(haptics.close)
        output = BrokenOutput()
        haptics.devices = [output]
        with self.assertLogs(level="ERROR"):
            haptics.trigger()
            haptics.thread.join(timeout=1)
        self.assertFalse(output.active)
        self.assertIsNotNone(haptics.fault)
        self.assertFalse(haptics.trigger())

    def test_udp_to_tablet_integration(self):
        haptics = Haptics([17])
        state = State(haptics)
        stop = threading.Event()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        sock.settimeout(0.05)
        receiver = threading.Thread(target=receive, args=(sock, state, SECRET, stop, None))
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
        http = threading.Thread(target=server.serve_forever)
        receiver.start()
        http.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            with urllib.request.urlopen(url) as response:
                self.assertIn(b"Honk detected", response.read())
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                sender.sendto(b"malformed", sock.getsockname())
                sender.sendto(self.packet(), sock.getsockname())
            deadline = time.monotonic() + 2
            while not state.snapshot()["connected"] and time.monotonic() < deadline:
                time.sleep(0.01)
            with urllib.request.urlopen(url + "/api/status") as response:
                result = json.load(response)
            self.assertTrue(result["honk"])
            self.assertTrue(result["healthy"])
            self.assertEqual(result["alert_count"], 1)
        finally:
            stop.set()
            receiver.join(timeout=1)
            sock.close()
            server.shutdown()
            server.server_close()
            http.join(timeout=1)
            haptics.close()


if __name__ == "__main__":
    unittest.main()
