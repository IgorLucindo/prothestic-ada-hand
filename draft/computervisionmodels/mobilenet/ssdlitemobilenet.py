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
    processable_frame = transform(frame2)

    # run inference
    with torch.no_grad():
        prediction = model([processable_frame])

    # set atribute
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']

    score = 0
    name = "nothing"
    box = boxes[0]
    for i in range(len(labels)):
        if classes[labels[i].item()] not in classes:
                continue

        if scores[i].item() > max(score, 0.25):
            score = scores[i].item()
            name = classes[labels[i].item()]
            box = boxes[i]
            width = int(box[2].item()) - int(box[0].item())
        
    # calculate distance
    distance_mm = 0
    if name in objects:
        distance_mm = (objects[name]['width_mm'] * focal_length_mm) / width

        startPoint = (int(box[0].item()), int(box[1].item()))
        finishPoint = (int(box[2].item()), int(box[3].item()))
        cv2.rectangle(frame, startPoint, finishPoint, (0, 255, 0))
    else:
        name = "nothing"

    print("name: ", name, "    score: ", score, '    distance: ', distance_mm, ' '*20, end='\r')

    cv2.imshow('img', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break