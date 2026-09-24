import argparse
import json
import logging
import signal
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files

from .haptics import Haptics
from .protocol import MAX_PACKET, decode, secret_from_env


class State:
    def __init__(self, haptics, timeout=2.0):
        self.haptics = haptics
        self.timeout = timeout
        self.lock = threading.Lock()
        self.sessions = {}
        self.message = None
        self.received = float("-inf")
        self.last_alert = float("-inf")
        self.alert_count = 0

    def accept(self, message, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            # Keep anti-replay state beyond the protocol's accepted timestamp age.
            self.sessions = {key: value for key, value in self.sessions.items() if now - value[1] < 10}
            previous = self.sessions.get(message["session"])
            if previous and message["seq"] <= previous[0]:
                return False
            if self.message and message["sent"] < self.message["sent"]:
                return False
            self.sessions[message["session"]] = (message["seq"], now)
            self.message, self.received = message, now
            if not message["healthy"]:
                self.last_alert = float("-inf")
            if message["healthy"] and message["honk"]:
                self.last_alert = now
                if self.haptics.trigger():
                    self.alert_count += 1
            return True

    def snapshot(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            connected = self.message is not None and now - self.received < self.timeout
            healthy = connected and self.message["healthy"]
            return dict(connected=connected, healthy=healthy,
                        honk=bool(healthy and now - self.last_alert < 1.5),
                        probability=self.message["probability"] if healthy else None,
                        age_seconds=round(now - self.received, 2) if self.message else None,
                        alert_count=self.alert_count, haptic_fault=self.haptics.fault)


def make_handler(state):
    html = files("ecocar").joinpath("static/tablet.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/status":
                body, content_type = json.dumps(state.snapshot()).encode(), "application/json"
            elif self.path in ("/", "/index.html"):
                body, content_type = html, "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass
    return Handler


def receive(sock, state, secret, stop, allowed_ip):
    while not stop.is_set():
        try:
            packet, peer = sock.recvfrom(MAX_PACKET + 1)
            if allowed_ip and peer[0] != allowed_ip:
                continue
            state.accept(decode(packet, secret))
        except socket.timeout:
            continue
        except ValueError:
            # Untrusted packets are ignored, including malformed and replayed messages.
            continue
        except OSError:
            if not stop.is_set():
                logging.exception("Receiver failed")
            return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--http-port", type=int, default=8080)
    parser.add_argument("--jetson-ip", help="Optional allowed source IPv4 address")
    parser.add_argument("--pins", default="17,27", help="BCM driver input pins")
    parser.add_argument("--hardware", action="store_true", help="Enable physical GPIO; default is dry run")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    secret = secret_from_env()
    haptics = Haptics([int(p) for p in args.pins.split(",")], dry_run=not args.hardware)
    stop = threading.Event()
    state = State(haptics)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server = None
    threads = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    try:
        sock.bind((args.bind, args.port))
        sock.settimeout(0.5)
        server = ThreadingHTTPServer((args.bind, args.http_port), make_handler(state))
        threads = [threading.Thread(target=receive, args=(sock, state, secret, stop, args.jetson_ip), daemon=True),
                   threading.Thread(target=server.serve_forever, daemon=True)]
        for thread in threads:
            thread.start()
        logging.info("Tablet http://<pi-ip>:%s; GPIO %s", args.http_port,
                     "ENABLED" if args.hardware else "dry run")
        while not stop.wait(0.5):
            if any(not thread.is_alive() for thread in threads):
                raise RuntimeError("Pi service worker exited")
    finally:
        stop.set()
        haptics.close()
        sock.close()
        if server:
            if len(threads) == 2 and threads[1].is_alive():
                server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)


if __name__ == "__main__":
    main()
