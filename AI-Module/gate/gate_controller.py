"""
gate/gate_controller.py

Simulated gate control. Swap the two marked methods for real hardware
calls later (Raspberry Pi GPIO / servo, or serial commands to an Arduino)
without changing any code that calls this class.

Example (Raspberry Pi, when you have hardware):
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(RELAY_PIN, GPIO.OUT)
    GPIO.output(RELAY_PIN, GPIO.HIGH)   # <- goes where _hardware_open() is

Example (Arduino over serial):
    import serial
    ser = serial.Serial('/dev/ttyUSB0', 9600)
    ser.write(b'OPEN\\n')               # <- goes where _hardware_open() is
"""

import time
import threading


class GateController:
    def __init__(self, open_seconds=5, cooldown_seconds=10):
        """
        open_seconds:      how long the gate stays open before auto-closing
        cooldown_seconds:  minimum time between the SAME plate re-triggering
                            the gate (stops one car re-opening it 5x while idling)
        """
        self.is_open = False
        self.open_seconds = open_seconds
        self.cooldown_seconds = cooldown_seconds
        self._last_opened_for = {}   # plate -> timestamp
        self._lock = threading.Lock()

    # ------------------------------------------------------------
    # Public API — call this from your detection pipeline
    # ------------------------------------------------------------
    def trigger(self, plate):
        """
        Attempt to open the gate for a given (already-authorized) plate.
        Returns "OPENED" or "COOLDOWN" (ignored because it just opened
        for this same plate).
        """
        with self._lock:
            now = time.time()
            last = self._last_opened_for.get(plate, 0)

            if now - last < self.cooldown_seconds:
                print(f"[GATE] Ignoring repeat trigger for {plate} (cooldown)")
                return "COOLDOWN"

            self._last_opened_for[plate] = now

        self._open()
        threading.Timer(self.open_seconds, self._close).start()
        return "OPENED"

    # ------------------------------------------------------------
    # Internal actions — replace bodies with real hardware calls
    # ------------------------------------------------------------
    def _open(self):
        self.is_open = True
        self._hardware_open()
        print(f"[GATE] OPEN  (will auto-close in {self.open_seconds}s)")

    def _close(self):
        self.is_open = False
        self._hardware_close()
        print("[GATE] CLOSED")

    def _hardware_open(self):
        # SIMULATION ONLY. Replace with GPIO.output(PIN, HIGH) or
        # servo.angle = 90, or ser.write(b'OPEN\\n') for real hardware.
        pass

    def _hardware_close(self):
        # SIMULATION ONLY. Replace with GPIO.output(PIN, LOW) or
        # servo.angle = 0, or ser.write(b'CLOSE\\n') for real hardware.
        pass


# Simple manual test
if __name__ == "__main__":
    gate = GateController(open_seconds=3, cooldown_seconds=5)
    gate.trigger("2A-1234")
    time.sleep(1)
    gate.trigger("2A-1234")   # should be ignored (cooldown)
    time.sleep(4)
    gate.trigger("2A-1234")   # should open again
    time.sleep(4)
