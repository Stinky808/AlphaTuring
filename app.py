import cv2
import os
import tempfile
import threading
from dotenv import load_dotenv
from google import genai
from google.genai import types
import sounddevice as sd
import scipy.io.wavfile as wavfile
from flask import Flask, Response, render_template, jsonify
from ultralytics import YOLO

load_dotenv()

MODEL_PATH = "best.pt"
RECORD_SECONDS = 6
SAMPLE_RATE = 16000

gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

app = Flask(__name__)
model = YOLO(MODEL_PATH)

lock = threading.Lock()
camera = cv2.VideoCapture(0)
victim_detected = False
is_recording = False
latest_transcript = ""
latest_response = ""


def record_and_analyze():
    global is_recording, latest_transcript, latest_response

    audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()  # close handle so wavfile and os.unlink can access it on Windows
    wavfile.write(tmp.name, SAMPLE_RATE, audio)

    try:
        with open(tmp.name, "rb") as f:
            audio_bytes = f.read()

        result = gemini.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                "Transcribe exactly what is being said in this audio. "
                "Then on a new line write 'Response:' followed by a short, calm, reassuring message "
                "that a rescuer could say to help this person."
            ]
        )
        text = result.text
        if "Response:" in text:
            parts = text.split("Response:", 1)
            latest_transcript = parts[0].strip()
            latest_response = parts[1].strip()
        else:
            latest_transcript = text
            latest_response = ""
    finally:
        os.unlink(tmp.name)
        with lock:
            is_recording = False


def generate_frames():
    global victim_detected, is_recording
    prev_detected = False

    while True:
        success, frame = camera.read()
        if not success:
            break

        results = model.predict(frame, conf=0.70, verbose=False)
        annotated = results[0].plot()

        detected = len(results[0].boxes) > 0
        with lock:
            victim_detected = detected
            should_record = detected and not prev_detected and not is_recording
            if should_record:
                is_recording = True

        if should_record:
            threading.Thread(target=record_and_analyze, daemon=True).start()

        prev_detected = detected

        _, buffer = cv2.imencode(".jpg", annotated)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    with lock:
        return jsonify({"detected": victim_detected})


@app.route("/transcript")
def transcript():
    with lock:
        return jsonify({"transcript": latest_transcript, "response": latest_response})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
