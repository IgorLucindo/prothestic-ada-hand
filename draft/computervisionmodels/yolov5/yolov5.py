import torch
import cv2
import time


model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

cap = cv2.VideoCapture(0)

previousTime = 0


while True:
    currentTime = time.time()
    deltaTime = currentTime - previousTime
    previousTime = currentTime
    # print(1/deltaTime)


    _, frame = cap.read()


    results = model(frame)


    for i in range(results.xyxy[0].shape[0]):
        box = results.xyxy[0][i][:4].numpy()
        score = results.xyxy[0][i][-2].item()
        name = model.names[results.xyxy[0][i][-1].item()]
        print(name, score)

    
    cv2.imshow('frame', frame)


    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break