from flask import Flask, render_template, Response, jsonify
from gesture import GestureController

app = Flask(__name__)

controller = GestureController()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start():
    controller.start()
    return jsonify({"status":"started"})

@app.route('/stop')
def stop():
    controller.stop()
    return jsonify({"status":"stopped"})

@app.route('/video')
def video():
    return Response(controller.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/volume')
def volume():
    return jsonify({"volume":controller.volume})

if __name__ == "__main__":
    app.run(debug=True)