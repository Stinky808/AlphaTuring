import asyncio
import io
import os
import subprocess
import tempfile
import time
import wave

import cv2
import pyaudio
import requests
from ultralytics import YOLO

from google import genai
from google.genai import types


# =========================
# Models / API config
# =========================

GEMINI_MODEL = "gemini-3.5-flash"

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

ELEVENLABS_API_KEY="sk_c86bd17cac9cc74804a275f011179c40142e901601e2791e"
ELEVENLABS_VOICE_ID="s3TPKV1kjDlVtZbl4Ksh"

ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"


# =========================
# Camera / detection config
# =========================

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

POSE_MODEL = "yolo11n-pose.pt"

SHOW_PREVIEW = True

CONFIDENCE_THRESHOLD = 0.45
DETECTION_COOLDOWN_SECONDS = 20.0


# =========================
# Audio config
# =========================

AUDIO_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_CHUNK = 512
LISTEN_SECONDS = 7


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


async def wait_for_first_body_part():
    print("Detector loop started...")

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at CAMERA_INDEX={CAMERA_INDEX}")

    model = YOLO(POSE_MODEL)

    active_parts = set()
    last_announced = {}

    priority_parts = ["hand", "arm", "leg", "foot", "face", "torso"]

    try:
        while True:
            ret, frame = await asyncio.to_thread(cap.read)

            if not ret:
                print("Could not read webcam frame")
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
            now = time.monotonic()

            chosen_part = None

            for part in priority_parts:
                newly_visible = part in detected_parts and part not in active_parts
                cooldown_ok = (
                    now - last_announced.get(part, 0)
                ) >= DETECTION_COOLDOWN_SECONDS

                if newly_visible and cooldown_ok:
                    chosen_part = part
                    break

            active_parts = detected_parts

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

                cv2.imshow("Body-Part Detector", annotated)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt

            if chosen_part:
                print(f"\nLocal detector: START_CONVERSATION: {chosen_part}")
                return chosen_part

            await asyncio.sleep(0.03)

    finally:
        cap.release()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()


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
        model=GEMINI_MODEL,
        contents=[prompt],
    )

    return (response.text or "").strip()


def transcribe_answer(client, wav_bytes):
    response = client.models.generate_content(
        model=GEMINI_MODEL,
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
        model=GEMINI_MODEL,
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
            "Use the conversation memory so you do not repeat the same question unnecessarily. "
            "Say help is on the way."
        )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
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
        "Say help is on the way. "
        "Use the conversation memory so you do not repeat yourself. "
        "Then ask one short question to understand what help they need next. "
        "Keep it under 35 words."
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
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
# Main conversation
# =========================

async def main():
    print("Starting assistive vision prototype...")

    gemini_client = genai.Client()

    while True:
        print("\nScanning for visible body parts...")

        # New session memory starts here.
        # This resets whenever the previous user is classified as OK.
        conversation_memory = []

        body_part = await wait_for_first_body_part()

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

                # Reset memory by leaving this inner loop.
                # The outer loop starts a fresh conversation_memory list.
                conversation_memory = []
                break

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

                # Keep talking to the same person with memory preserved.
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


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopping...")