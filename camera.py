import cv2

from config import (
    CONFIDENCE_THRESHOLD,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    CENTER_TOLERANCE_PIXELS,
    TOO_CLOSE_BODY_HEIGHT_RATIO,
)


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

    best_visible_points = None
    best_visible_count = 0

    for person_kpts in keypoints:
        visible_points = [
            (float(kpt[0]), float(kpt[1]))
            for kpt in person_kpts
            if float(kpt[2]) >= CONFIDENCE_THRESHOLD
        ]

        if len(visible_points) > best_visible_count:
            best_visible_count = len(visible_points)
            best_visible_points = visible_points

    if not best_visible_points:
        return None

    xs = [pt[0] for pt in best_visible_points]
    ys = [pt[1] for pt in best_visible_points]

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
    Fast local movement fallback.

    It rotates left/right to keep a person centered.
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


def encode_frame_as_jpeg_bytes(frame):
    ok, buffer = cv2.imencode(".jpg", frame)

    if not ok:
        return None

    return buffer.tobytes()


def draw_preview(result, detected_parts, current_command, conversation_active):
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
        f"Conversation: {'ON' if conversation_active else 'OFF'}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    return annotated