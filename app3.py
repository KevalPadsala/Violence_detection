from flask import Flask, render_template, Response, request
import cv2
import os
import numpy as np
from keras.models import load_model
from collections import deque

app = Flask(__name__)
file_path = None
camera = None
model = load_model("modelnew.h5")
Q = deque(maxlen=128)

def initialize_camera():
    global camera
    global file_path
    if file_path is not None:
        camera = cv2.VideoCapture(file_path)
    else: 
        camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        # read the camera frame
        success, frame = camera.read()
        if not success:
            break

        # process the frame for prediction
        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128)).astype("float32")
        frame = frame.reshape(128, 128, 3) / 255

        # make predictions on the frame and then update the predictions queue
        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        Q.append(preds)

        # perform prediction averaging over the current history of previous predictions
        results = np.array(Q).mean(axis=0)
        label = (results > 0.50)[0]

        # add text to the frame
        text = "Violence: {}".format(label)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        text_color = (0, 255, 0) # default : green

        if label: # Violence prob
            text_color = (0, 0, 255) # red
        else:
            text_color = (0, 255, 0)
        text_org = (10, 50)  # text origin in the frame
        cv2.putText(output, text, text_org, font, font_scale, text_color, thickness)

        # encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', output)
        frame = buffer.tobytes()

        # yield the frame as a byte string
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index_final1.html')


@app.route('/video')
def video():
    initialize_camera()
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        global file_path
        file_path = os.path.join('upload.html', file.filename)
        file.save(file_path)
        # process the file
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
    