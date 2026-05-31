from google.genai import types

from config import TEXT_MODEL, ROBOTICS_MODEL, VALID_MOVES
from camera import encode_frame_as_jpeg_bytes


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
                "NEEDS_HELP = they mention danger, pain, injury, cannot move, cannot breathe, "
                "being trapped, or needing help.\n"
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


def decide_robot_movement(client, frame, detected_parts, local_suggestion):
    """
    Uses Gemini Robotics-ER for high-level movement choice.

    Behavior:
    - Search when no person is visible.
    - Turn toward a visible person.
    - Approach when centered.
    - Stop if close or uncertain.

    The Arduino ultrasonic sensor is the final safety layer.
    """
    image_bytes = encode_frame_as_jpeg_bytes(frame)

    if image_bytes is None:
        return "STOP"

    person_visible = bool(detected_parts)

    prompt = f"""
You are controlling a small Arduino robot through serial text commands.

Goal:
Search for a person. When a person is visible, approach them slowly but do not get too close.

Detected body parts from local YOLO pose detector:
{", ".join(sorted(detected_parts)) if detected_parts else "none"}

Person visible:
{"yes" if person_visible else "no"}

Fast local controller suggestion:
{local_suggestion}

Choose exactly one movement command:
FORWARD
LEFT
RIGHT
BACKWARD
STOP

Movement policy:
- If no person is visible, choose LEFT or RIGHT to slowly search.
- If a person is visible on the left side of the image, choose LEFT.
- If a person is visible on the right side of the image, choose RIGHT.
- If a person is visible and centered, choose FORWARD.
- If the person appears very close, choose STOP.
- If uncertain, choose STOP.
- Do not use BACKWARD unless there is a clear reason.
- The Arduino has an ultrasonic sensor and will block FORWARD if the robot is too close.

Return only one word:
FORWARD, LEFT, RIGHT, BACKWARD, or STOP.
""".strip()

    try:
        response = client.models.generate_content(
            model=ROBOTICS_MODEL,
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                ),
            ],
        )

    except Exception as exc:
        print(f"Gemini movement decision failed: {exc}")
        return local_suggestion if local_suggestion in VALID_MOVES else "STOP"

    command = (response.text or "").strip().upper()

    if command not in VALID_MOVES:
        return "STOP"

    return command


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