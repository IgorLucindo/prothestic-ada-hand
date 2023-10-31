import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
import cv2
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import torchvision.transforms as T
from PIL import Image
import time


focal_length_mm = 510 # Focal length of the camera in millimeters

window_size = 30
stride = 10
detection_counts = {}
framesUntilDetection = 8
start_time = time.time()
object_positions = {}
frameMsg = ""
newMsg = False
processing = False

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
    'bottle': {'width_mm': 60, 'grasp': 'Power'},
    'cell phone': {'width_mm': 70, 'grasp': 'Power'},
    'banana': {'width_mm': 30, 'grasp': 'Pinch'},
    'apple': {'width_mm': 50, 'grasp': 'Power'},
}

curr_obj = {
    'name': "nothing",
    'grasp': "None",
    'prev_grasp': "None",
    'score': 0,
    'box': [],
    'dist_mm': 0,
    'prev_dist_mm': 0,
    'vel': 0
}

# Load pre-trained MobileNetV2 model
model = ssdlite320_mobilenet_v3_large(pretrained=True)
model.eval()

transform = T.Compose([T.ToTensor()])


# show image with displayed information
def showImage(frame):
    cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)
    # write informations
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = .7
    text = str(round(curr_obj["score"], 3)) + ' ' + curr_obj["name"]
    cv2.putText(frame, text, (50, 50), font, font_scale,(0,255,0), thickness=2)
    cv2.putText(frame, f'Grasp Type: {curr_obj["grasp"]}', (50, 100), font, font_scale, (0, 0, 255), thickness=2)
    cv2.putText(frame, f'Distance: {round(curr_obj["dist_mm"], 3)}', (50, 150), font, font_scale, (255, 0, 0), thickness=2)
    # draw box
    if curr_obj['name'] != "nothing":
                startPoint = (int(curr_obj['box'][0].item()), int(curr_obj['box'][1].item()))
                finishPoint = (int(curr_obj['box'][2].item()), int(curr_obj['box'][3].item()))
                cv2.rectangle(frame, startPoint, finishPoint, (0, 255, 0))
    # show
    cv2.imshow('frame', frame)
    cv2.waitKey(1)


# reset current object atributes
def resetObject():
    curr_obj['name'] = "nothing"
    curr_obj['score'] = 0
    curr_obj['box'] = []
    curr_obj['dist_mm'] = 0


# choose the object with highest score
def setCurrentObject(scores, labels, boxes):
    for i in range(len(labels)):
        if classes[labels[i].item()] not in objects:
            continue

        if scores[i].item() > max(curr_obj['score'], 0.25):
            # set current object atributes
            curr_obj['score'] = scores[i].item()
            curr_obj['name'] = classes[labels[i].item()]
            curr_obj['box'] = boxes[i]


# check if object is in 'objects' dictionary
def checkObject(frame_count, deltaTime):
    name = curr_obj['name']
    if name in objects:
        # set current object distance and velocity
        width = int(curr_obj['box'][2].item()) - int(curr_obj['box'][0].item())
        curr_obj['dist_mm'] = (objects[name]['width_mm'] * focal_length_mm) / width
        curr_obj['vel'] = -(curr_obj['dist_mm'] - curr_obj['prev_dist_mm'])/deltaTime

        # Update detection counts and positions for the current object
        if name in detection_counts:
            detection_counts[name] += 1
            object_positions[name].append(frame_count)
        else:
            detection_counts[name] = 1
            object_positions[name] = [frame_count]
    else:
        # reset current object atributes
        resetObject()


# function that is called every time the message arrives
def cbImageCompressed(message):
    global frameMsg, newMsg, processing
    if not processing:
        newMsg = True

        frameMsg = message.data


# loop
def loop():
    global newMsg, processing

    frame_count = 0
    rate = rospy.Rate(30)

    currentTime = 0
    previousTime = 0

    while not rospy.is_shutdown():
        if newMsg:
            # get delta time
            currentTime = time.time()
            deltaTime = currentTime - previousTime
            previousTime = currentTime


            # set processing to True
            processing = True
            newMsg = False
            # increment frame
            frame_count += 1


            # reset current object atributes every start of loop
            resetObject()


            # convert from ROS to OpenCV image format
            np_arr = np.frombuffer(frameMsg, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # rotate image to desirable angle
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # Convert image to PIL format for transformation
            frame2 = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            newFrame = transform(frame2)


            # Process the frame with the model
            with torch.no_grad():
                prediction = model([newFrame])
            # The 'boxes' key contains the bounding boxes of the detected objects
            boxes = prediction[0]['boxes']
            # The 'labels' key contains the class labels of the detected objects
            labels = prediction[0]['labels']
            # The 'scores' key contains the confidence scores of the detected objects
            scores = prediction[0]['scores']


            # choose the object with highest score
            setCurrentObject(scores, labels, boxes)


            # check if object is in 'objects' dictionary
            checkObject(frame_count, deltaTime)


            # Check if the window is complete
            if frame_count >= window_size:
                # reset current object grasp type
                curr_obj['grasp'] = "None"
                # Process the detection counts and send grasp types
                for obj, count in detection_counts.items():
                    if count >= framesUntilDetection:
                        curr_obj['grasp'] = objects[obj]['grasp']

                # Slide the window by 'stride' frames
                for obj, positions in object_positions.items():
                    positions_to_remove = [p for p in positions if p <= stride]
                    count_removed = len(positions_to_remove)
                    for p in positions_to_remove:
                        positions.remove(p)
                    for i in range(len(positions)):
                        positions[i] -= stride
                    detection_counts[obj] -= count_removed

                frame_count -= stride

                for obj in list(detection_counts.keys()):
                    if obj not in object_positions:
                        del detection_counts[obj]
                    # if len(positions) < stride:
                    #     del detection_counts[obj]
                    #     del object_positions[obj]


            # publish to grasp detection topic
            dist_vel_array = Float32MultiArray()
            dist_vel_array.data = [curr_obj['dist_mm'], curr_obj['vel']]
            grasp_detection_pub.publish(dist_vel_array)


            # publish to grasp type topic
            if curr_obj['grasp'] != "None" and curr_obj['grasp'] != curr_obj['prev_grasp']:
                grasp_type_pub.publish(curr_obj['grasp'])
            

            # show image with displayed information
            showImage(frame)


            # set current object atributes
            curr_obj['prev_grasp'] = curr_obj['grasp']
            curr_obj['prev_dist_mm'] = curr_obj['dist_mm']


            # set processing to False
            processing = False
        rate.sleep()

	
if __name__ == "__main__":
    # initialize the subscriber node
    rospy.init_node('object_detection', anonymous=True)
    
    # create grasp detection topic and publisher
    grasp_detection_pub = rospy.Publisher("grasp_detection", Float32MultiArray, queue_size=10)

    # create grasp type topic and publisher
    grasp_type_pub = rospy.Publisher("grasp_type", String, queue_size=10)
    
    # subscribe
    rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, cbImageCompressed)

    loop()

    cv2.destroyAllWindows()