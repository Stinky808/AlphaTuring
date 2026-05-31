import asyncio
import ctypes
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


# ---------------------------------------------------------------------------
# Suppress the flood of ALSA "Unknown PCM" errors on Raspberry Pi OS.
# ---------------------------------------------------------------------------

def _suppress_alsa_errors():
    try:
        asound = ctypes.cdll.LoadLibrary("libasound.so.2")
        handler_type = ctypes.CFUNCTYPE(
            None, ctypes.c_char_p, ctypes.c_int,
            ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        )
        # Assign to function attribute — keeps it alive for the process lifetime
        _suppress_alsa_errors._handler = handler_type(lambda *_: None)
        asound.snd_lib_error_set_handler(_suppress_alsa_errors._handler)
    except Exception:
        pass

_suppress_alsa_errors()


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
                    f"  {i}: {info.get('name')} "
                    f"inputs={info.get('maxInputChannels')} "
                    f"rate={info.get('defaultSampleRate')}"
                )
        if not found:
            print("  No audio input devices found. Plug in a USB microphone.")
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

    input_device_index = (
        int(AUDIO_INPUT_DEVICE_INDEX)
        if AUDIO_INPUT_DEVICE_INDEX not in (None, "")
        else None
    )

    try:
        # p.open() is blocking — wrap it so the event loop stays free.
        stream = await asyncio.to_thread(
            p.open,
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
        return pcm_to_wav_bytes(pcm_bytes, sample_rate=AUDIO_RATE)

    except Exception as exc:
        print("\nCould not open/read microphone.")
        print(f"Error: {exc}")
        list_input_devices()
        # Return silence so the robot does not crash.
        silence = b"\x00\x00" * AUDIO_RATE * LISTEN_SECONDS
        return pcm_to_wav_bytes(silence, sample_rate=AUDIO_RATE)

    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        p.terminate()
