import numpy as np
import cv2
import torch
import torchvision.transforms as T
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from PIL import Image


COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# load model
model = ssdlite320_mobilenet_v3_large(pretrained=True)
model.eval()

transform = T.Compose([T.ToTensor()])


# convert from ROS to OpenCV image format
def convertFrameFormat(frameMsg):
    np_arr = np.frombuffer(frameMsg, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # rotate image to desirable angle
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame2 = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    processable_frame = transform(frame2)

    return frame, processable_frame

# run inference
def runModel(processable_frame):
    # inference
    with torch.no_grad():
        results = model([processable_frame])

    # get atribute
    boxes = results[0]['boxes'].numpy()
    labels = results[0]['labels'].numpy()
    classes = []
    for i in range(len(labels)):
        classes.append(COCO_CLASSES[labels[i]])
    scores = results[0]['scores'].numpy()
    
    return boxes, classes, scores