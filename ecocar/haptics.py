"""GPIO drives a motor-driver input, never a motor directly."""
import logging
import threading
import time


class Haptics:
    def __init__(self, pins, dry_run=True, pulse_seconds=0.25, cooldown=2.0):
        if not 0.02 <= pulse_seconds <= 0.5 or cooldown < 1:
            raise ValueError("Pulse must be 20–500 ms; cooldown must be >=1 s")
        if not pins or len(set(pins)) != len(pins) or any(not 0 <= p <= 27 for p in pins):
            raise ValueError("Provide unique BCM GPIO pins in 0..27")
        self.devices = []
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.thread = None
        self.last = float("-inf")
        self.pulse_seconds = pulse_seconds
        self.cooldown = cooldown
        self.fault = None
        if not dry_run:
            from gpiozero import OutputDevice
            try:
                for pin in pins:
                    self.devices.append(OutputDevice(pin, initial_value=False))
            except Exception:
                self.close()
                raise

    def trigger(self):
        with self.lock:
            if (self.stop.is_set() or self.fault or time.monotonic() - self.last < self.cooldown
                    or (self.thread and self.thread.is_alive())):
                return False
            self.last = time.monotonic()
            self.thread = threading.Thread(target=self._pulse, daemon=True)
            self.thread.start()
            return True

    def _pulse(self):
        try:
            for device in self.devices:
                device.on()
            logging.info("Haptic pulse%s", " (dry run)" if not self.devices else "")
            self.stop.wait(self.pulse_seconds)
        except Exception as exc:
            self.fault = str(exc)
            logging.exception("Haptic output failed")
        finally:
            for device in self.devices:
                try:
                    device.off()
                except Exception as exc:
                    self.fault = str(exc)
                    logging.exception("Cannot switch output off")

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=1)
        for device in self.devices:
            try:
                device.off()
            except Exception as exc:
                self.fault = str(exc)
                logging.exception("Output off failed during shutdown")
            try:
                device.close()
            except Exception as exc:
                self.fault = str(exc)
                logging.exception("Output close failed during shutdown")
        self.devices.clear()
