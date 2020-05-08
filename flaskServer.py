import os
import cv2
import numpy as np
import json
from flask import Flask, request
from werkzeug.utils import secure_filename

Upload_folder = '/code/uploadedimages/'
Allowed_extensions = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Upload_folder

classes = []
with open('/code/coco.names', "r") as f:
    classes = [line.strip() for line in f.readlines()]


class Model:
    label = ""
    accuracy = ""

    def __init__(self, label, accuracy):
        self.label = label
        self.accuracy = accuracy



@app.route('/')
def functions():
    return "welcome to object detection"


@app.route('/api/object_detection', methods=['POST'])
def objectDetection():
    file = request.files['image']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    img = cv2.imread('/code/uploadedimages/'+ filename)
    net = cv2.dnn.readNet('/code/yolov3-tiny.weights','/code/yolov3-tiny.cfg')

    layers_names = net.getLayerNames()

    net.setInput(cv2.dnn.blobFromImage(img, 0.000392, (416,416), (0, 0, 0), True, crop=False))
    outs = net.forward([layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()])

    class_ids = []
    confidences = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                confidences.append(float(confidence))
                class_ids.append(class_id)

    combined = []
    for i in range(len(confidences)):
        combined.append(Model(str(classes[class_ids[i]]), str("%0.2f" % (100 * float(confidences[i])))))

    json_arr = []
    for i in range(len(confidences)):
        json_arr.append(json.dumps(combined[i].__dict__))

    return json.dumps({"object":json_arr}, indent = 6).replace('\\"','"') .replace('"{','{').replace('}"','}')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', threaded=True, port=5000)

