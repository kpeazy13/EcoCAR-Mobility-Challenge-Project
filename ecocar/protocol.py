"""Small signed messages; no audio leaves the Jetson."""
import hashlib
import hmac
import json
import math
import time
import uuid

MAX_PACKET = 2048


def secret_from_env():
    import os
    secret = os.environ.get("ECOCAR_SECRET", "")
    if len(secret) < 32 or secret.startswith("REPLACE_"):
        raise ValueError("Set ECOCAR_SECRET to the same random 32+ character key on both devices")
    return secret.encode()


def encode(secret, session, sequence, probability, honk, healthy=True):
    body = json.dumps(dict(version=1, session=session, seq=sequence, sent=time.time(),
                           probability=probability, honk=honk, healthy=healthy),
                      sort_keys=True, separators=(",", ":"), allow_nan=False)
    signature = hmac.new(secret, body.encode(), hashlib.sha256).hexdigest()
    return json.dumps({"body": body, "signature": signature}).encode()


def decode(packet, secret, now=None, max_age=3.0):
    if len(packet) > MAX_PACKET:
        raise ValueError("Oversized packet")
    try:
        envelope = json.loads(packet)
        body, signature = envelope["body"], envelope["signature"]
        expected = hmac.new(secret, body.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise ValueError("Invalid signature")
        message = json.loads(body)
        if message["version"] != 1 or str(uuid.UUID(message["session"])) != message["session"]:
            raise ValueError("Invalid version/session")
        if type(message["seq"]) is not int or message["seq"] < 0:
            raise ValueError("Invalid sequence")
        for key in ("probability", "sent"):
            if type(message[key]) not in (int, float) or not math.isfinite(message[key]):
                raise ValueError("Invalid numeric value")
        if not 0 <= message["probability"] <= 1:
            raise ValueError("Invalid probability")
        if any(type(message[k]) is not bool for k in ("honk", "healthy")):
            raise ValueError("Invalid boolean")
        if abs((time.time() if now is None else now) - message["sent"]) > max_age:
            raise ValueError("Stale message or clocks out of sync")
        return message
    except (KeyError, TypeError, AttributeError, UnicodeError, OverflowError, RecursionError) as exc:
        raise ValueError("Malformed packet") from exc


class Gate:
    """Hysteresis plus consecutive-window confirmation."""
    def __init__(self, threshold, confirmations=2, release_margin=0.15):
        if not 0 < threshold <= 1 or confirmations < 1 or release_margin < 0:
            raise ValueError("Invalid gate settings")
        self.high = threshold
        # A zero release threshold can latch forever because sigmoid never reaches zero.
        self.low = max(threshold * 0.5, threshold - release_margin)
        self.confirmations = confirmations
        self.count = 0
        self.active = False

    def update(self, probability):
        if self.active:
            if probability <= self.low:
                self.active = False
                self.count = 0
        else:
            self.count = self.count + 1 if probability >= self.high else 0
            self.active = self.count >= self.confirmations
        return self.active
