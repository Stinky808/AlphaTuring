import time

from config import (
    ARDUINO_PORT,
    ARDUINO_BAUD,
    VALID_MOVES,
    DEFAULT_SAFE_MOVE,
)

try:
    import serial
except ImportError:
    serial = None


def open_arduino():
    if not ARDUINO_PORT:
        print("Arduino serial disabled. Set ARDUINO_PORT to enable movement output.")
        return None

    if serial is None:
        print("pyserial is not installed. Run: pip install pyserial")
        return None

    try:
        arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
        time.sleep(2)
        print(f"Arduino connected on {ARDUINO_PORT} at {ARDUINO_BAUD} baud.")
        return arduino

    except Exception as exc:
        print(f"Could not open Arduino on {ARDUINO_PORT}: {exc}")
        print("Continuing without Arduino movement output.")
        return None


def send_move(arduino, command):
    command = (command or DEFAULT_SAFE_MOVE).strip().upper()

    if command not in VALID_MOVES:
        command = DEFAULT_SAFE_MOVE

    print(f"Arduino command: {command}")

    if arduino is None:
        return

    try:
        arduino.write((command + "\n").encode("utf-8"))

    except Exception as exc:
        print(f"Failed to send Arduino command: {exc}")


def close_arduino(arduino):
    send_move(arduino, "STOP")

    if arduino is not None:
        try:
            arduino.close()
        except Exception:
            pass