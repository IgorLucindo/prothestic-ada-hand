import cv2
from ultralytics import YOLO
import time
import numpy as np

batch_size = 2


model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)
frames = np.zeros((batch_size, 480, 640, 3), np.uint8)

frame_count = 0
previousTime = 0


while True:
    currentTime = time.time()
    deltaTime = currentTime - previousTime
    if frame_count % batch_size == 0:
        previousTime = currentTime
        print(batch_size/deltaTime)


    frame_count += 1


    _, frame = cap.read()
    frames = np.insert(frames, 0, frame, axis=0)
    frames = frames[:batch_size]


    if frame_count % batch_size == 0:
        results = model([frames[0], frames[1]], max_det=1, verbose=False)[0]

        for i in range(results.boxes.shape[0]):
            box = results.boxes.xywh[i]
            score = results.boxes.conf[i].item()
            name = results.names[results.boxes.cls[i].item()]
            # print(name, score)

    
    # cv2.imshow('frame', frame)


    if cv2.waitKey(15) == 27:
        cv2.destroyAllWindows()
        break