import cv2
import mediapipe as mp
import math
import numpy as np

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


class GestureController:

    def __init__(self):
        self.running = False
        self.volume = 0

        # Initialize Windows audio controller
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_,
            CLSCTX_ALL,
            None
        )

        self.volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
        self.minVol, self.maxVol, _ = self.volume_interface.GetVolumeRange()

        # Mediapipe hands setup
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.draw = mp.solutions.drawing_utils


    def start(self):
        self.running = True


    def stop(self):
        self.running = False


    def generate_frames(self):

        self.cap = cv2.VideoCapture(0)

        while True:

            success, frame = self.cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = self.hands.process(rgb)

            if self.running and result.multi_hand_landmarks:

                for hand in result.multi_hand_landmarks:

                    self.draw.draw_landmarks(
                        frame,
                        hand,
                        self.mpHands.HAND_CONNECTIONS
                    )

                    lm = hand.landmark

                    # Thumb tip
                    x1 = int(lm[4].x * w)
                    y1 = int(lm[4].y * h)

                    # Index finger tip
                    x2 = int(lm[8].x * w)
                    y2 = int(lm[8].y * h)

                    cv2.circle(frame, (x1, y1), 8, (0, 0, 255), -1)
                    cv2.circle(frame, (x2, y2), 8, (0, 255, 255), -1)

                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    # Distance between fingers
                    dist = math.hypot(x2 - x1, y2 - y1)

                    # Convert distance → volume %
                    vol_percent = np.interp(dist, [20, 200], [0, 100])

                    vol_percent = max(0, min(100, vol_percent))

                    self.volume = int(vol_percent)

                    # Convert percentage → Windows volume level
                    vol = np.interp(self.volume, [0, 100], [self.minVol, self.maxVol])

                    # Set system volume
                    self.volume_interface.SetMasterVolumeLevel(vol, None)

                    # Display volume text
                    cv2.putText(
                        frame,
                        f"Volume: {self.volume}%",
                        (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2
                    )

                    # Draw volume bar
                    bar_height = int(np.interp(self.volume, [0, 100], [400, 150]))

                    cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3)

                    cv2.rectangle(frame, (50, bar_height), (85, 400), (0, 255, 0), -1)

            ret, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )