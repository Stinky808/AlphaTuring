import os

# =========================
# Gemini model config
# =========================

TEXT_MODEL = os.getenv("TEXT_MODEL", "gemini-3.5-flash")
ROBOTICS_MODEL = os.getenv("ROBOTICS_MODEL", "gemini-robotics-er-1.6-preview")


# =========================
# ElevenLabs config
# =========================

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")


# =========================
# Arduino config
# =========================

ARDUINO_PORT = os.getenv("ARDUINO_PORT", "")
ARDUINO_BAUD = int(os.getenv("ARDUINO_BAUD", "9600"))

VALID_MOVES = {"FORWARD", "LEFT", "RIGHT", "BACKWARD", "STOP"}
DEFAULT_SAFE_MOVE = "STOP"

USE_GEMINI_MOVEMENT = os.getenv("USE_GEMINI_MOVEMENT", "1") == "1"
GEMINI_MOVE_INTERVAL_SECONDS = float(os.getenv("GEMINI_MOVE_INTERVAL_SECONDS", "2.0"))
MOVEMENT_COMMAND_INTERVAL_SECONDS = float(os.getenv("MOVEMENT_COMMAND_INTERVAL_SECONDS", "0.20"))


# =========================
# Camera / detection config
# =========================

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))

POSE_MODEL = os.getenv("POSE_MODEL", "yolo11n-pose.pt")

SHOW_PREVIEW = os.getenv("SHOW_PREVIEW", "1") == "1"

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
DETECTION_COOLDOWN_SECONDS = float(os.getenv("DETECTION_COOLDOWN_SECONDS", "20.0"))

CENTER_TOLERANCE_PIXELS = int(os.getenv("CENTER_TOLERANCE_PIXELS", "80"))
TOO_CLOSE_BODY_HEIGHT_RATIO = float(os.getenv("TOO_CLOSE_BODY_HEIGHT_RATIO", "0.82"))


# =========================
# Audio config
# =========================

AUDIO_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_CHUNK = 512
LISTEN_SECONDS = 7