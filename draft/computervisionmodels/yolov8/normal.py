from ultralytics import YOLO
import cv2
import time


model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

previousTime = 0


while True:
    currentTime = time.time()
    deltaTime = currentTime - previousTime
    previousTime = currentTime
    print(1/deltaTime)


    _, frame = cap.read()


    results = model(frame, max_det=1, verbose=False)[0]


    for i in range(results.boxes.shape[0]):
        box = results.boxes.xyxy[i].numpy()
        score = results.boxes.conf[i].item()
        name = results.names[results.boxes.cls[i].item()]
        # print(name, score)

    
    # cv2.imshow('frame', frame)


    if cv2.waitKey(15) == 27:
        cv2.destroyAllWindows()
        break