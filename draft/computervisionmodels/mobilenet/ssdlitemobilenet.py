import cv2
import time
import torch
import torchvision.transforms as T
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from PIL import Image


# load model
model = ssdlite320_mobilenet_v3_large(pretrained=True)
model.eval()

transform = T.Compose([T.ToTensor()])

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

objects = {
    'bottle': {'width': 60}
}


# get processable frame for inference
def getProcessableFrame(frame):
    frame2 = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return transform(frame2)


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


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    previousTime = 0

    # loop
    while True:
        # get deltaTime
        currentTime = time.time()
        deltaTime = currentTime - previousTime
        previousTime = currentTime

        _, frame = cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        processable_frame = getProcessableFrame(frame)

        # run inference
        boxes, classes, scores = runModel(processable_frame)

        # set atributes
        detected = False
        box = []
        name = "nothing"
        score = 0
        for i in range(len(classes)):
            if classes[i] not in objects:
                    continue

            if scores[i] > max(score, 0.25):
                detected = True
                score = scores[i]
                name = classes[i]
                box = boxes[i]

        if detected:
            startPoint = (int(box[0]), int(box[1]))
            finishPoint = (int(box[2]), int(box[3]))
            cv2.rectangle(frame, startPoint, finishPoint, (0, 255, 0))

        # debug
        print("name: ", name, "    score: ", score, '    FPS: ', round(1/deltaTime, 2), ' '*20, end='\r')
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break