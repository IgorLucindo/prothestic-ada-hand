import numpy as np
import cv2
from deepsparse import Pipeline


models_path = "/home/bioinlab/Desktop/carlosIgor/prothestic_ada_hand/ada_visual_ws/src/ada_visual_control/src/models/"

# load model
model = Pipeline.create(
    task="yolo",
    # model_path=models_path + "yolov5-s-coco-pruned75_quantized.onnx",
    # model_path=models_path + "yolov5-s-coco-pruned75.onnx",
    # model_path=models_path + "yolov5-m-coco-pruned55.4block.onnx",
    model_path=models_path + "yolov5-l-coco-pruned90_quantized.onnx",
    class_names="coco",
    num_cores=4,
    # image_size=(640, 480)
    image_size=(416, 416)
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