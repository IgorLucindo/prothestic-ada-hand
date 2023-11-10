import torch
import torchvision.transforms as T
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import cv2
from object_detector import *
from PIL import Image

# Load Object Detector
detector = HomogeneousBgDetector()

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
    detected = False
    for i in range(len(labels)):
        if classes[labels[i].item()] not in objects:
            continue
        score = scores[i].item()
        if score > maxScore and score > 0.25:
            maxScore = score
            name = classes[labels[i].item()]
            box = boxes[i]
            detected = True
        
    # calculate distance
    if detected:
        width = int(box[2].item()) - int(box[0].item())
        distance_mm = (objects[name]['width_mm'] * focal_length_mm) / width

        pt1 = (int(box[0].item()), int(box[1].item()))
        pt2 = (int(box[2].item()), int(box[3].item()))

        frame_crop = frame[pt1[1]:pt2[1] , pt1[0]:pt2[0]]

        contours = detector.detect_objects(frame_crop)
        maxArea = 0
        if len(contours):
            contour = contours[0]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > maxArea:
                maxArea = area
                contour = cnt
        # Fit an ellipse to the contour
        ellipse = cv2.fitEllipse(contour)
        
        # Extract the angle of rotation from the ellipse
        angle = ellipse[2]

        cv2.ellipse(frame_crop, ellipse, (0,0,255), 3)


    print("name: ", name, "     score: ", maxScore)

    if detected:
        cv2.rectangle(frame, pt1, pt2, (0, 255, 0))
        # cv2.imshow('crop', frame_crop)
    cv2.imshow('img', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break