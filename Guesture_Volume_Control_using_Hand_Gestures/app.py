from flask import Flask, render_template, Response, redirect, url_for, jsonify
import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

app = Flask(__name__)

camera_running = False
current_volume = 0

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(False,1,0,0.6,0.6)

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

min_vol,max_vol = volume.GetVolumeRange()[0],volume.GetVolumeRange()[1]

distance_history=[]
SMOOTHING_FRAMES=5
prev_vol_percent=-1


def generate_frames():

    global camera_running
    global current_volume
    global prev_vol_percent

    while camera_running:

        success,frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame,1)

        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:

            for hand_landmarks in result.multi_hand_landmarks:

                mp_draw.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)

                h,w,_ = frame.shape

                thumb = hand_landmarks.landmark[4]
                index = hand_landmarks.landmark[8]

                x1,y1 = int(thumb.x*w),int(thumb.y*h)
                x2,y2 = int(index.x*w),int(index.y*h)

                distance = math.hypot(x2-x1,y2-y1)

                distance_history.append(distance)

                if len(distance_history)>SMOOTHING_FRAMES:
                    distance_history.pop(0)

                smooth_distance = sum(distance_history)/len(distance_history)

                vol = np.interp(smooth_distance,[30,200],[min_vol,max_vol])
                vol_bar = np.interp(smooth_distance,[30,200],[400,150])
                vol_percent = int(np.interp(smooth_distance,[30,200],[0,100]))

                if abs(vol_percent-prev_vol_percent)>1:
                    volume.SetMasterVolumeLevel(vol,None)
                    prev_vol_percent=vol_percent

                current_volume = vol_percent

                cv2.circle(frame,(x1,y1),10,(0,255,0),-1)
                cv2.circle(frame,(x2,y2),10,(0,255,0),-1)
                cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),3)

                cv2.putText(frame,f"Volume: {vol_percent}%",(30,50),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

                cv2.putText(frame,f"Distance: {int(smooth_distance)} px",(30,90),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

                cv2.rectangle(frame,(50,150),(80,400),(255,255,255),3)
                cv2.rectangle(frame,(50,int(vol_bar)),(80,400),(0,255,0),-1)

        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()

        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')


@app.route('/')
def index():
    return render_template('index.html',running=camera_running)


@app.route('/video')
def video():
    return Response(generate_frames(),
    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start')
def start():
    global camera_running
    camera_running=True
    return redirect(url_for('index'))


@app.route('/stop')
def stop():
    global camera_running
    camera_running=False
    return redirect(url_for('index'))


@app.route('/volume')
def volume_data():
    return jsonify(volume=current_volume)


if __name__ == "__main__":
    app.run(debug=True)