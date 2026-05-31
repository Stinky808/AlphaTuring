import asyncio
import sys
import time

import cv2
from google import genai
from ultralytics import YOLO

from arduino import (
    open_arduino,
    send_move,
    close_arduino,
)
from config import (
    TEXT_MODEL,
    ROBOTICS_MODEL,
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    POSE_MODEL,
    SHOW_PREVIEW,
    DETECTION_COOLDOWN_SECONDS,
    USE_GEMINI_MOVEMENT,
    GEMINI_MOVE_INTERVAL_SECONDS,
    MOVEMENT_COMMAND_INTERVAL_SECONDS,
)
from conversation import run_conversation
from gemini_logic import decide_robot_movement
from camera import (
    detect_body_parts_from_pose,
    local_movement_from_pose,
    draw_preview,
)


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

    priority_parts = [
        "hand",
        "arm",
        "leg",
        "foot",
        "face",
        "torso",
    ]

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

            frame = cv2.resize(
                frame,
                (FRAME_WIDTH, FRAME_HEIGHT),
            )

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

            # Detect newly visible body part to trigger conversation.
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

            # Start background conversation without stopping the camera.
            if chosen_part and conversation_task is None:
                print(f"\nLocal detector: START_CONVERSATION: {chosen_part}")

                send_move(arduino, "STOP")
                last_sent_command = "STOP"
                last_sent_time = now

                conversation_task = asyncio.create_task(
                    run_conversation(
                        gemini_client,
                        chosen_part,
                    )
                )

            # Movement control.
            #
            # local_movement_from_pose() now handles both cases:
            # - no person visible -> SEARCH_DIRECTION, usually LEFT
            # - person visible -> LEFT / RIGHT / FORWARD / STOP
            current_command = local_command

            if USE_GEMINI_MOVEMENT and detected_parts:
                # Collect completed Gemini Robotics movement decision.
                if movement_task is not None and movement_task.done():
                    try:
                        gemini_command = movement_task.result()

                        # Only trust Gemini movement if a person is still visible.
                        # This prevents an old delayed Gemini result from moving the robot
                        # after the person disappears.
                        if detected_parts:
                            current_command = gemini_command
                        else:
                            current_command = local_command

                    except Exception as exc:
                        print(f"Movement task ended with error: {exc}")
                        current_command = local_command

                    movement_task = None

                # Start a new movement decision at a safe interval,
                # but only when a person is visible.
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

            # Extra safety while speaking to a person:
            # do not drive forward/backward during conversation.
            #
            # This means the robot can search and approach before conversation,
            # but once it starts talking, it will not keep driving toward the person.
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
                annotated = draw_preview(
                    result=result,
                    detected_parts=detected_parts,
                    current_command=current_command,
                    conversation_active=conversation_task is not None,
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


async def main():
    print("Starting assistive vision prototype with continuous camera + Arduino movement...")
    print(f"Text model: {TEXT_MODEL}")
    print(f"Robotics model: {ROBOTICS_MODEL}")

    gemini_client = genai.Client()
    arduino = open_arduino()

    try:
        await continuous_camera_loop(
            gemini_client,
            arduino,
        )

    finally:
        close_arduino(arduino)


if __name__ == "__main__":
    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        print("\nStopping...")