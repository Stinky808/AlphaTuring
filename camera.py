import asyncio
import io
import os
import subprocess
import tempfile
import time
import wave
import sys

import cv2
import pyaudio
import requests
from ultralytics import YOLO

try:
    import serial
except ImportError:
    serial = None

from google import genai
from google.genai import types


# =========================
# Models / API config
# =========================

# Use a general Gemini model for conversation, transcription, classification,
# and spoken responses.
TEXT_MODEL = os.getenv("TEXT_MODEL", "gemini-3.5-flash")

# Use Gemini Robotics-ER only for camera + robot movement reasoning.
ROBOTICS_MODEL = os.getenv("ROBOTICS_MODEL", "gemini-robotics-er-1.6-preview")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")


# =========================
# Arduino movement config
# =========================

# Set ARDUINO_PORT in your terminal before running, for example:
#   macOS:   export ARDUINO_PORT=/dev/cu.usbmodem1101
#   Linux:   export ARDUINO_PORT=/dev/ttyACM0
#   Windows: set ARDUINO_PORT=COM3
ARDUINO_PORT = os.getenv("ARDUINO_PORT", "")
ARDUINO_BAUD = int(os.getenv("ARDUINO_BAUD", "9600"))

VALID_MOVES = {"FORWARD", "LEFT", "RIGHT", "BACKWARD", "STOP"}
DEFAULT_SAFE_MOVE = "STOP"

# Gemini Robotics movement decisions are rate-limited so you do not call the API every frame.
USE_GEMINI_MOVEMENT = os.getenv("USE_GEMINI_MOVEMENT", "1") == "1"
GEMINI_MOVE_INTERVAL_SECONDS = float(os.getenv("GEMINI_MOVE_INTERVAL_SECONDS", "2.0"))

# Local command throttle.
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

# Movement tuning. This script rotates to center a detected person,
# but it does not drive forward toward them by default.
CENTER_TOLERANCE_PIXELS = int(os.getenv("CENTER_TOLERANCE_PIXELS", "80"))
TOO_CLOSE_BODY_HEIGHT_RATIO = float(os.getenv("TOO_CLOSE_BODY_HEIGHT_RATIO", "0.82"))


# =========================
# Audio config
# =========================

AUDIO_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_CHUNK = 512
LISTEN_SECONDS = 7


# =========================
# Arduino helpers
# =========================

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


# =========================
# Body-part detection
# =========================

def keypoint_visible(kpts, idx, threshold=CONFIDENCE_THRESHOLD):
    return kpts[idx][2] >= threshold


def detect_body_parts_from_pose(result):
    detected = set()

    if result.keypoints is None or result.keypoints.data is None:
        return detected

    keypoints = result.keypoints.data.cpu().numpy()

    for person_kpts in keypoints:
        # COCO keypoints:
        # 0 nose
        # 1 left eye, 2 right eye
        # 3 left ear, 4 right ear
        # 5 left shoulder, 6 right shoulder
        # 7 left elbow, 8 right elbow
        # 9 left wrist, 10 right wrist
        # 11 left hip, 12 right hip
        # 13 left knee, 14 right knee
        # 15 left ankle, 16 right ankle

        if any(keypoint_visible(person_kpts, i) for i in [0, 1, 2, 3, 4]):
            detected.add("face")

        if keypoint_visible(person_kpts, 9) or keypoint_visible(person_kpts, 10):
            detected.add("hand")

        if any(keypoint_visible(person_kpts, i) for i in [5, 6, 7, 8, 9, 10]):
            detected.add("arm")

        torso_points = [5, 6, 11, 12]
        if sum(keypoint_visible(person_kpts, i) for i in torso_points) >= 3:
            detected.add("torso")

        if any(keypoint_visible(person_kpts, i) for i in [11, 12, 13, 14, 15, 16]):
            detected.add("leg")

        # COCO pose does not have toe/foot points; ankle is used as a foot proxy.
        if keypoint_visible(person_kpts, 15) or keypoint_visible(person_kpts, 16):
            detected.add("foot")

    return detected


def estimate_person_position(result):
    """
    Returns approximate person center and height in pixels.
    Uses visible YOLO pose keypoints.
    """
    if result.keypoints is None or result.keypoints.data is None:
        return None

    keypoints = result.keypoints.data.cpu().numpy()
    best = None
    best_visible_count = 0

    for person_kpts in keypoints:
        visible_points = [
            (float(kpt[0]), float(kpt[1]))
            for kpt in person_kpts
            if float(kpt[2]) >= CONFIDENCE_THRESHOLD
        ]

        if len(visible_points) > best_visible_count:
            best_visible_count = len(visible_points)
            best = visible_points

    if not best:
        return None

    xs = [pt[0] for pt in best]
    ys = [pt[1] for pt in best]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    return {
        "center_x": (x_min + x_max) / 2.0,
        "center_y": (y_min + y_max) / 2.0,
        "height": y_max - y_min,
        "visible_points": best_visible_count,
    }


def local_movement_from_pose(result):
    """
    Fast local fallback. It keeps the person centered by rotating left/right.
    It does not drive forward toward the person by default.
    """
    position = estimate_person_position(result)

    if position is None:
        return "STOP"

    if position["height"] >= FRAME_HEIGHT * TOO_CLOSE_BODY_HEIGHT_RATIO:
        return "STOP"

    frame_center_x = FRAME_WIDTH / 2.0

    if position["center_x"] < frame_center_x - CENTER_TOLERANCE_PIXELS:
        return "LEFT"

    if position["center_x"] > frame_center_x + CENTER_TOLERANCE_PIXELS:
        return "RIGHT"

    return "STOP"


# =========================
# Gemini Robotics movement reasoning
# =========================

def encode_frame_as_jpeg_bytes(frame):
    ok, buffer = cv2.imencode(".jpg", frame)

    if not ok:
        return None

    return buffer.tobytes()


def decide_robot_movement(client, frame, detected_parts, local_suggestion):
    """
    Uses Gemini Robotics-ER for high-level movement choice.
    The output is constrained to a tiny command set for Arduino safety.
    """
    image_bytes = encode_frame_as_jpeg_bytes(frame)

    if image_bytes is None:
        return "STOP"

    prompt = f"""
You are controlling a small Arduino robot through serial text commands.

Goal:
Keep a visible person safely in camera view for assistive monitoring.

Detected body parts from local YOLO pose detector:
{", ".join(sorted(detected_parts)) if detected_parts else "none"}

Fast local controller suggestion:
{local_suggestion}

Choose exactly one movement command:
FORWARD
LEFT
RIGHT
BACKWARD
STOP

Safety rules:
- Prefer STOP when uncertain.
- STOP if the person appears close.
- Do not chase, bump, or touch a person.
- Use LEFT or RIGHT only to gently rotate and keep the person centered.
- Avoid FORWARD unless clearly safe and necessary.
- Return only the command. No punctuation. No explanation.
""".strip()

    try:
        response = client.models.generate_content(
            model=ROBOTICS_MODEL,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            ],
        )
    except Exception as exc:
        print(f"Gemini movement decision failed: {exc}")
        return local_suggestion if local_suggestion in VALID_MOVES else "STOP"

    command = (response.text or "").strip().upper()

    if command not in VALID_MOVES:
        return "STOP"

    return command


# =========================
# Audio helpers
# =========================

def pcm_to_wav_bytes(pcm_bytes, sample_rate=16000):
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(AUDIO_CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)

    return buffer.getvalue()


async def record_answer_wav():
    print(f"\nListening for {LISTEN_SECONDS} seconds...")

    p = pyaudio.PyAudio()

    stream = p.open(
        format=pyaudio.paInt16,
        channels=AUDIO_CHANNELS,
        rate=AUDIO_RATE,
        input=True,
        frames_per_buffer=AUDIO_CHUNK,
    )

    frames = []
    chunks = int(AUDIO_RATE / AUDIO_CHUNK * LISTEN_SECONDS)

    try:
        for _ in range(chunks):
            data = await asyncio.to_thread(
                stream.read,
                AUDIO_CHUNK,
                exception_on_overflow=False,
            )
            frames.append(data)

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    pcm_bytes = b"".join(frames)
    return pcm_to_wav_bytes(pcm_bytes, sample_rate=AUDIO_RATE)


# =========================
# Blackboard memory
# =========================

def add_memory(memory, role, text):
    memory.append(
        {
            "role": role,
            "text": text,
        }
    )


def format_memory(memory, max_items=10):
    if not memory:
        return "No previous conversation yet."

    recent = memory[-max_items:]

    return "\n".join(
        f"{item['role']}: {item['text']}" for item in recent
    )


# =========================
# Gemini text / transcription
# =========================

def generate_first_question(client, body_part):
    if body_part == "face":
        prompt = (
            "A local vision detector sees the person's face. "
            "Write one calm first safety-check question. "
            "Ask if they can hear you and speak clearly. "
            "Keep it under 18 words. "
            "Do not say help is on the way."
        )
    else:
        prompt = (
            f"A local vision detector sees the person's {body_part}. "
            f"Write one calm first safety-check question asking whether they can move their {body_part}. "
            "Keep it under 18 words. "
            "Do not say help is on the way."
        )

    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=[prompt],
    )

    return (response.text or "").strip()


def transcribe_answer(client, wav_bytes):
    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=[
            (
                "Transcribe only the main clear user answer. "
                "Ignore filler words like uh, um, okay, and repeated phrases. "
                "If there is no clear speech, return exactly: NO_SPEECH."
            ),
            types.Part.from_bytes(
                data=wav_bytes,
                mime_type="audio/wav",
            ),
        ],
    )

    return (response.text or "").strip()


def classify_user_status(client, transcript, memory):
    conversation_so_far = format_memory(memory)

    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=[
            (
                "Classify the person's latest answer into exactly one label.\n\n"
                "Labels:\n"
                "OK = they say they are okay, safe, can hear, can speak, or can move.\n"
                "NEEDS_HELP = they mention danger, pain, injury, cannot move, cannot breathe, being trapped, or needing help.\n"
                "UNCLEAR = the answer is unclear, unrelated, or no speech.\n\n"
                "Use the conversation memory for context, but classify only the latest answer.\n\n"
                f"Conversation memory:\n{conversation_so_far}\n\n"
                f"Latest answer: {transcript}\n\n"
                "Return only one word: OK, NEEDS_HELP, or UNCLEAR."
            )
        ],
    )

    label = (response.text or "").strip().upper()

    if "NEEDS_HELP" in label:
        return "NEEDS_HELP"

    if "OK" in label:
        return "OK"

    return "UNCLEAR"


def generate_followup(client, transcript, memory):
    conversation_so_far = format_memory(memory)

    if not transcript or transcript.upper() == "NO_SPEECH":
        prompt = (
            "You are a calm assistive safety assistant.\n\n"
            f"Conversation memory:\n{conversation_so_far}\n\n"
            "The person did not give a clear spoken answer. "
            "Write one brief calm sentence asking them to repeat or answer yes/no. "
            "Say help is on the way."
        )
    else:
        prompt = (
            "You are a calm assistive safety assistant.\n\n"
            f"Conversation memory:\n{conversation_so_far}\n\n"
            f"The person just answered: {transcript}\n\n"
            "Respond with one short supportive sentence, then ask one short follow-up safety question. "
            "Also provide a safety tip catered to their situation. "
            "Use the conversation memory so you do not repeat the same question unnecessarily. "
            "Say help is on the way."
        )

    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=[prompt],
    )

    return (response.text or "").strip()


def generate_help_response(client, transcript, memory):
    conversation_so_far = format_memory(memory)

    prompt = (
        "You are a calm assistive safety assistant. "
        "The person may need help.\n\n"
        f"Conversation memory:\n{conversation_so_far}\n\n"
        f"The person just said: {transcript}\n\n"
        "Write a brief spoken response. "
        "Be calm and supportive. "
        "Provide a safety tip catered to their situation. "
        "Say help is on the way. "
        "Use the conversation memory so you do not repeat yourself. "
        "Then ask one short question to understand what help they need next. "
        "Keep it under 35 words."
    )

    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=[prompt],
    )

    return (response.text or "").strip()


# =========================
# ElevenLabs TTS using macOS afplay
# =========================

def speak_with_elevenlabs(text):
    if not ELEVENLABS_API_KEY:
        print("\nMissing ELEVENLABS_API_KEY. Printing instead of speaking.")
        print(f"Assistant: {text}")
        return

    if not ELEVENLABS_VOICE_ID:
        print("\nMissing ELEVENLABS_VOICE_ID. Printing instead of speaking.")
        print(f"Assistant: {text}")
        return

    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/"
        f"{ELEVENLABS_VOICE_ID}?output_format=mp3_44100_128"
    )

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }

    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {
            "stability": 0.55,
            "similarity_boost": 0.75,
        },
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=30,
    )

    if response.status_code != 200:
        print("\nElevenLabs TTS failed.")
        print(response.status_code, response.text)
        print(f"Assistant: {text}")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        f.write(response.content)
        audio_path = f.name

    print(f"\nAssistant: {text}")

    try:
        subprocess.run(["afplay", audio_path], check=False)
    finally:
        try:
            os.remove(audio_path)
        except OSError:
            pass


# =========================
# Conversation task
# =========================

async def run_conversation(gemini_client, body_part):
    """
    Runs one conversation in the background while the camera loop continues.
    Returns when the person is classified as OK, or if the task is cancelled.
    """
    conversation_memory = []

    first_question = await asyncio.to_thread(
        generate_first_question,
        gemini_client,
        body_part,
    )

    add_memory(conversation_memory, "assistant", first_question)

    print(f"\nGemini first question: {first_question}")
    await asyncio.to_thread(speak_with_elevenlabs, first_question)

    while True:
        wav_bytes = await record_answer_wav()

        print("Transcribing answer...")

        transcript = await asyncio.to_thread(
            transcribe_answer,
            gemini_client,
            wav_bytes,
        )

        print(f"\nUser transcript: {transcript}")

        add_memory(conversation_memory, "user", transcript)

        status = await asyncio.to_thread(
            classify_user_status,
            gemini_client,
            transcript,
            conversation_memory,
        )

        print(f"\nUser status: {status}")

        if status == "OK":
            reset_message = (
                "I’m glad you’re okay. I’ll keep monitoring for anyone else who may need help."
            )

            add_memory(conversation_memory, "assistant", reset_message)

            print(f"\nAssistant reset message: {reset_message}")
            await asyncio.to_thread(speak_with_elevenlabs, reset_message)
            return

        if status == "NEEDS_HELP":
            help_message = await asyncio.to_thread(
                generate_help_response,
                gemini_client,
                transcript,
                conversation_memory,
            )

            add_memory(conversation_memory, "assistant", help_message)

            print(f"\nGemini help response: {help_message}")
            await asyncio.to_thread(speak_with_elevenlabs, help_message)
            continue

        followup = await asyncio.to_thread(
            generate_followup,
            gemini_client,
            transcript,
            conversation_memory,
        )

        add_memory(conversation_memory, "assistant", followup)

        print(f"\nGemini follow-up: {followup}")
        await asyncio.to_thread(speak_with_elevenlabs, followup)

        await asyncio.sleep(0.5)


# =========================
# Continuous camera / robot loop
# =========================

async def continuous_camera_loop(gemini_client, arduino):
    print("Detector loop started. Camera will stay on until you quit.")

    if sys.platform == "darwin":
        cap_backend = cv2.CAP_AVFOUNDATION
    else:
        cap_backend = cv2.CAP_ANY

    cap = cv2.VideoCapture(CAMERA_INDEX, cap_backend)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at CAMERA_INDEX={CAMERA_INDEX}")

    model = YOLO(POSE_MODEL)

    active_parts = set()
    last_announced = {}
    priority_parts = ["hand", "arm", "leg", "foot", "face", "torso"]

    conversation_task = None
    movement_task = None

    last_gemini_move_time = 0.0
    last_sent_command = None
    last_sent_time = 0.0
    current_command = "STOP"

    try:
        while True:
            ret, frame = await asyncio.to_thread(cap.read)

            if not ret:
                print("Could not read webcam frame")
                send_move(arduino, "STOP")
                await asyncio.sleep(1)
                continue

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            results = await asyncio.to_thread(
                model.predict,
                frame,
                verbose=False,
                conf=0.35,
            )

            result = results[0]
            detected_parts = detect_body_parts_from_pose(result)
            local_command = local_movement_from_pose(result)
            now = time.monotonic()

            # Clean up finished conversation task.
            if conversation_task is not None and conversation_task.done():
                try:
                    conversation_task.result()
                except Exception as exc:
                    print(f"Conversation task ended with error: {exc}")
                conversation_task = None

            # Detect newly visible body part to trigger a conversation.
            chosen_part = None

            for part in priority_parts:
                newly_visible = part in detected_parts and part not in active_parts
                cooldown_ok = (
                    now - last_announced.get(part, 0)
                ) >= DETECTION_COOLDOWN_SECONDS

                if newly_visible and cooldown_ok:
                    chosen_part = part
                    last_announced[part] = now
                    break

            active_parts = detected_parts

            # Start a background conversation without stopping the camera.
            if chosen_part and conversation_task is None:
                print(f"\nLocal detector: START_CONVERSATION: {chosen_part}")

                send_move(arduino, "STOP")
                last_sent_command = "STOP"
                last_sent_time = now

                conversation_task = asyncio.create_task(
                    run_conversation(gemini_client, chosen_part)
                )

            # Movement control.
            if detected_parts:
                current_command = local_command

                if USE_GEMINI_MOVEMENT:
                    # Collect completed Gemini movement result.
                    if movement_task is not None and movement_task.done():
                        try:
                            current_command = movement_task.result()
                        except Exception as exc:
                            print(f"Movement task ended with error: {exc}")
                            current_command = local_command
                        movement_task = None

                    # Start new Gemini Robotics movement decision at a safe interval.
                    if (
                        movement_task is None
                        and now - last_gemini_move_time >= GEMINI_MOVE_INTERVAL_SECONDS
                    ):
                        last_gemini_move_time = now
                        frame_for_gemini = frame.copy()
                        parts_for_gemini = set(detected_parts)

                        movement_task = asyncio.create_task(
                            asyncio.to_thread(
                                decide_robot_movement,
                                gemini_client,
                                frame_for_gemini,
                                parts_for_gemini,
                                local_command,
                            )
                        )
            else:
                current_command = "STOP"

            # Extra safety: while talking to a person, do not move forward/backward.
            # Rotation is still allowed for camera centering.
            if conversation_task is not None and not conversation_task.done():
                if current_command in {"FORWARD", "BACKWARD"}:
                    current_command = "STOP"

            # Send movement command to Arduino.
            if (
                current_command != last_sent_command
                or now - last_sent_time >= MOVEMENT_COMMAND_INTERVAL_SECONDS
            ):
                send_move(arduino, current_command)
                last_sent_command = current_command
                last_sent_time = now

            # Preview window.
            if SHOW_PREVIEW:
                annotated = result.plot()

                label = ", ".join(sorted(detected_parts)) or "none"

                cv2.putText(
                    annotated,
                    f"Detected: {label}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    annotated,
                    f"Move: {current_command}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    annotated,
                    f"Conversation: {'ON' if conversation_task else 'OFF'}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("Body-Part Detector", annotated)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt

            await asyncio.sleep(0.03)

    finally:
        send_move(arduino, "STOP")

        if conversation_task is not None and not conversation_task.done():
            conversation_task.cancel()

        if movement_task is not None and not movement_task.done():
            movement_task.cancel()

        cap.release()

        if SHOW_PREVIEW:
            cv2.destroyAllWindows()


# =========================
# Main
# =========================

async def main():
    print("Starting assistive vision prototype with continuous camera + Arduino movement...")
    print(f"Text model: {TEXT_MODEL}")
    print(f"Robotics model: {ROBOTICS_MODEL}")

    gemini_client = genai.Client()
    arduino = open_arduino()

    try:
        await continuous_camera_loop(gemini_client, arduino)
    finally:
        send_move(arduino, "STOP")

        if arduino is not None:
            try:
                arduino.close()
            except Exception:
                pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopping...")