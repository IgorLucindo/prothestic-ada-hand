import cv2
import time
from deepsparse import Pipeline


# load model with deepsparse
model = Pipeline.create(
    task="yolo",
    model_path="models/yolov5-s-coco-pruned75_quantized.onnx",
    class_names="coco",
    num_cores=4,
    image_size=(640, 480)
)

cap = cv2.VideoCapture(0)
previousTime = 0

# loop
while True:
    # get fps
    currentTime = time.time()
    deltaTime = currentTime - previousTime
    previousTime = currentTime
    print('FPS: ',round(1/deltaTime, 2), end='\r')

    _, frame = cap.read()

    # run inference
    results = model(images=frame)

    boxes = results.boxes[0]
    labels = results.labels[0]
    scores = results.scores[0]

    score = 0
    for i in range(len(labels)):
        if scores[i] > max(score, 0.25):
            # set current object atributes
            score = scores[i]
            name = labels[i]
            box = boxes[i]
            # print(name, score, box)



    cv2.imshow('frame', frame)


    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break