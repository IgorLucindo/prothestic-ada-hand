import torch
import torchvision.transforms as T
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import cv2
from PIL import Image


classes = [
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
    'bottle': {'width_mm': 60}
}

focal_length_mm = 510 # Focal length of the camera in millimeters


model = ssdlite320_mobilenet_v3_large(pretrained=True)
model.eval()

transform = T.Compose([T.ToTensor()])

cap = cv2.VideoCapture(0)


while True:
    _, frame = cap.read()

    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    frame2 = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    img = transform(frame2)

    with torch.no_grad():
        prediction = model([img])

    # The 'boxes' key contains the bounding boxes of the detected objects
    boxes = prediction[0]['boxes']

    # The 'labels' key contains the class labels of the detected objects
    labels = prediction[0]['labels']

    # The 'scores' key contains the confidence scores of the detected objects
    scores = prediction[0]['scores']

    maxScore = 0
    name = "nothing"
    box = boxes[0]
    for i in range(len(labels)):
        if scores[i].item() > maxScore:
            maxScore = scores[i].item()
            name = classes[labels[i].item()]
            box = boxes[i]
            width = int(box[2].item()) - int(box[0].item())
        
    # calculate distance
    if name in objects:
        distance_mm = (objects[name]['width_mm'] * focal_length_mm) / width
        print(distance_mm)

        startPoint = (int(box[0].item()), int(box[1].item()))
        finishPoint = (int(box[2].item()), int(box[3].item()))
        cv2.rectangle(frame, startPoint, finishPoint, (0, 255, 0))
    else:
        name = "nothing"

    print("name: ", name, "     score: ", maxScore)

    cv2.imshow('img', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break