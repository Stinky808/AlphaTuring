import asyncio
import io
import wave

import pyaudio

from config import (
    AUDIO_RATE,
    AUDIO_CHANNELS,
    AUDIO_CHUNK,
    LISTEN_SECONDS,
)


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

    return pcm_to_wav_bytes(
        pcm_bytes,
        sample_rate=AUDIO_RATE,
    )