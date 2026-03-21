from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import math
import numpy as np

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

app = Flask(__name__)

running = False
current_volume = 0

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None
)

volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
minVol, maxVol, _ = volume_interface.GetVolumeRange()

mpHands = mp.solutions.hands
hands = mpHands.Hands()
draw = mp.solutions.drawing_utils


def generate_frames():

    global running
    global current_volume

    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        if running and result.multi_hand_landmarks:

            for hand in result.multi_hand_landmarks:

                draw.draw_landmarks(
                    frame,
                    hand,
                    mpHands.HAND_CONNECTIONS
                )

                lm = hand.landmark

                x1 = int(lm[4].x * w)
                y1 = int(lm[4].y * h)

                x2 = int(lm[8].x * w)
                y2 = int(lm[8].y * h)

                cv2.circle(frame, (x1, y1), 8, (0, 0, 255), -1)
                cv2.circle(frame, (x2, y2), 8, (0, 255, 255), -1)

                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                dist = math.hypot(x2 - x1, y2 - y1)

                vol_percent = np.interp(dist, [20, 200], [0, 100])
                vol_percent = max(0, min(100, vol_percent))

                current_volume = int(vol_percent)

                vol = np.interp(current_volume, [0, 100], [minVol, maxVol])

                volume_interface.SetMasterVolumeLevel(vol, None)

                cv2.putText(
                    frame,
                    f"Volume: {current_volume}%",
                    (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

                bar_height = int(np.interp(current_volume, [0, 100], [400, 150]))

                cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3)
                cv2.rectangle(frame, (50, bar_height), (85, 400), (0, 255, 0), -1)

        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/start')
def start():
    global running
    running = True
    return jsonify({"status": "started"})


@app.route('/stop')
def stop():
    global running
    running = False
    return jsonify({"status": "stopped"})


@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/volume')
def volume():
    return jsonify({"volume": current_volume})
if __name__ == "__main__":
    app.run(debug=True)