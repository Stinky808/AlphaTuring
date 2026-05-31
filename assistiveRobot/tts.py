import os
import subprocess
import tempfile
import sys

import requests

from config import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_VOICE_ID,
    ELEVENLABS_MODEL_ID,
)


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
        if sys.platform == "darwin":
            subprocess.run(["afplay", audio_path], check=False)
        else:
            subprocess.run(["mpg123", "-q", audio_path], check=False)

    finally:
        try:
            os.remove(audio_path)
        except OSError:
            pass
