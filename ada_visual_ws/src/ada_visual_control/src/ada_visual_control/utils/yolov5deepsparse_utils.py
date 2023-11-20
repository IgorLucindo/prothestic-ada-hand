import numpy as np
import os
import cv2
from deepsparse import Pipeline


file_path = os.path.dirname(os.path.abspath(__file__))

# load model
model = Pipeline.create(
    task="yolo",
    model_path=file_path + "/models/yolov5-s-coco-pruned75_quantized.onnx",
    # model_path="zoo:yolov5-s-coco-pruned75",
    # model_path="zoo:yolov5-m-coco-pruned55.4block",
    class_names="coco",
    num_cores=4,
    image_size=(640, 480)
)


# convert from ROS to OpenCV image format
def convertFrameFormat(frameMsg):
    np_arr = np.frombuffer(frameMsg, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # rotate image to desirable angle
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    processable_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame, processable_frame

# run inference
def runModel(processable_frame):
    # inference
    results = model(images=processable_frame, iou_thres=0.2, conf_thres=0.1)

    # get atributes
    boxes = results.boxes[0]
    classes = results.labels[0]
    scores = results.scores[0]

    return boxes, classes, scores