import os
import cv2
import time
from deepsparse import Pipeline


file_path = os.path.dirname(os.path.abspath(__file__))

# load model with deepsparse
model = Pipeline.create(
    task="yolo",
    model_path=file_path + "/models/yolov5-s-coco-pruned75_quantized.onnx",
    class_names="coco",
    num_cores=4,
    image_size=(640, 480)
)

objects = {
    'bottle': {'width': 0},
    'banana': {'width': 0},
    'apple': {'width': 0},
}


# get processable frame for inference
def getProcessableFrame(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# run inference
def runModel(processable_frame):
    # inference
    results = model(images=processable_frame, iou_thres=0.2, conf_thres=0.1)

    # get atribute
    boxes = results.boxes[0]
    classes = results.labels[0]
    scores = results.scores[0]
    
    return boxes, classes, scores


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    previousTime = 0

    # loop
    while True:
        # get fps
        currentTime = time.time()
        deltaTime = currentTime - previousTime
        previousTime = currentTime

        # get frame
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