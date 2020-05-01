import os
import cv2
import numpy as np
import argparse
import json
from flask import url_for, jsonify, Flask, request
from werkzeug.utils import secure_filename, redirect

Upload_folder = '/code/uploadedimages/'
Allowed_extensions = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Upload_folder
args = argparse.ArgumentParser()


class Model:
    label = ""
    accuracy = ""

    def __init__(self, label, accuracy):
        self.label = label
        self.accuracy = accuracy


args.add_argument('-w', required=False, help="path to weights file is required")
args.add_argument('-c', required=False, help="path to cfg file is required")
args.add_argument('-n', required=False, help="path to names file is required")

options = args.parse_args()

weights_fileP = str(options.w)
cfg_fileP = str(options.c)
names_fileP = str(options.n)



@app.route('/')
def functions():
    return "welcome to object detection"


@app.route('/api/object_detection', methods=['POST'])
def objectDetection():
    file = request.files['imagefile']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    img = cv2.imread('/code/uploadedimages/'+ filename)

    #net = cv2.dnn.readNet(weights_fileP,cfg_fileP)
    net = cv2.dnn.readNet('/code/yolov3.weights','/code/yolov3.cfg')
    classes = []

    with open('/code/coco.names', "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layers_names = net.getLayerNames()
    outputlayers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # imgUMat = np.float32(img)
    # gray = cv2.cvtColor(imgUMat, cv2.COLOR_BGR2GRAY)
    blob = cv2.dnn.blobFromImage(img, 0.000392, (416,416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(outputlayers)

    class_ids = []
    confidences = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # center_x = int(detection[0] * width)
                # center_y = int(detection[1] * height)
                # w = int(detection[2] * width)
                # h = int(detection[3] * height)
                #
                # x = int(center_x - w / 2)
                # y = int(center_y - h / 2)
                #
                #
                #
                # cv2.rectangle(img, (x,y), (x+w,y+h), (0, 255, 0), 1)

                # boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    list = []

    for i in range(len(confidences)):
        list.append(Model(str(classes[class_ids[i]]), str("%0.2f" % (100 * float(confidences[i])))))

    json_arr = []

    for i in range(len(confidences)):
        json_arr.append(json.dumps(list[i].__dict__))
        # obj = json.dumps(list[i].__dict__)

    return json.dumps(json_arr,indent = 4).replace('\\"','"') .replace('"{','{').replace('}"','}')

    #obj = ""
    #for i in range(len(json_arr)):
    #    if i == 0:
    #        obj = obj + "[" + json_arr[i] + ","
    #    if 0 < i < (len(json_arr) - 1):
    #        obj = obj + json_arr[i] + ','
    #    elif i == len(json_arr) - 1:
    #        obj = obj + json_arr[i] + "]"

    #return obj


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

