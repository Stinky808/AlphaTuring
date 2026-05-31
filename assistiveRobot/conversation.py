import asyncio

from audio import record_answer_wav
from gemini_logic import (
    add_memory,
    generate_first_question,
    transcribe_answer,
    classify_user_status,
    generate_followup,
    generate_help_response,
)
from tts import speak_with_elevenlabs


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

    await asyncio.to_thread(
        speak_with_elevenlabs,
        first_question,
    )

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
                "I’m glad you’re okay. "
                "I’ll keep monitoring for anyone else who may need help."
            )

            add_memory(conversation_memory, "assistant", reset_message)

            print(f"\nAssistant reset message: {reset_message}")

            await asyncio.to_thread(
                speak_with_elevenlabs,
                reset_message,
            )

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

            await asyncio.to_thread(
                speak_with_elevenlabs,
                help_message,
            )

            continue

        followup = await asyncio.to_thread(
            generate_followup,
            gemini_client,
            transcript,
            conversation_memory,
        )

        add_memory(conversation_memory, "assistant", followup)

        print(f"\nGemini follow-up: {followup}")

        await asyncio.to_thread(
            speak_with_elevenlabs,
            followup,
        )

        await asyncio.sleep(0.5)