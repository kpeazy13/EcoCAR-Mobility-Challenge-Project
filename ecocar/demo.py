"""Synthetic notification messages, NOT a simulated accuracy evaluation."""
import argparse
import socket
import time
import uuid

from .protocol import encode, secret_from_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--seconds", type=float, default=30)
    args = parser.parse_args()
    secret = secret_from_env()
    session = str(uuid.uuid4())
    start = time.monotonic()
    sequence = 0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            while time.monotonic() - start < args.seconds:
                active = 3 <= (time.monotonic() - start) % 8 < 4
                sequence += 1
                sock.sendto(encode(secret, session, sequence, 0.95 if active else 0.05, active),
                            (args.pi, args.port))
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            sock.sendto(encode(secret, session, sequence + 1, 0.0, False, False),
                        (args.pi, args.port))


if __name__ == "__main__":
    main()
