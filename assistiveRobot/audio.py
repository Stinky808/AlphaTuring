import asyncio
import io
import os
import wave

import pyaudio

from config import (
    AUDIO_RATE,
    AUDIO_CHANNELS,
    AUDIO_CHUNK,
    LISTEN_SECONDS,
)


AUDIO_INPUT_DEVICE_INDEX = os.getenv("AUDIO_INPUT_DEVICE_INDEX")


def list_input_devices():
    p = pyaudio.PyAudio()

    try:
        print("\nAvailable audio input devices:")

        found = False

        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)

            if int(info.get("maxInputChannels", 0)) > 0:
                found = True
                print(
                    f"{i}: {info.get('name')} "
                    f"inputs={info.get('maxInputChannels')} "
                    f"rate={info.get('defaultSampleRate')}"
                )

        if not found:
            print("No audio input devices found. Plug in a USB microphone.")

    finally:
        p.terminate()


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
    stream = None

    input_device_index = None

    if AUDIO_INPUT_DEVICE_INDEX not in (None, ""):
        input_device_index = int(AUDIO_INPUT_DEVICE_INDEX)

    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=AUDIO_CHANNELS,
            rate=AUDIO_RATE,
            input=True,
            input_device_index=input_device_index,
            frames_per_buffer=AUDIO_CHUNK,
        )

        frames = []
        chunks = int(AUDIO_RATE / AUDIO_CHUNK * LISTEN_SECONDS)

        for _ in range(chunks):
            data = await asyncio.to_thread(
                stream.read,
                AUDIO_CHUNK,
                exception_on_overflow=False,
            )
            frames.append(data)

        pcm_bytes = b"".join(frames)

        return pcm_to_wav_bytes(
            pcm_bytes,
            sample_rate=AUDIO_RATE,
        )

    except Exception as exc:
        print("\nCould not open/read microphone.")
        print(f"Error: {exc}")
        list_input_devices()

        # Return silence so the robot does not crash.
        silence = b"\x00\x00" * AUDIO_RATE * LISTEN_SECONDS

        return pcm_to_wav_bytes(
            silence,
            sample_rate=AUDIO_RATE,
        )

    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()

        p.terminate()