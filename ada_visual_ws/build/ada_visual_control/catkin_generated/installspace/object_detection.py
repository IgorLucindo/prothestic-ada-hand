# all distances in mm
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
from ada_visual_control.utils.logger import log



# camera focal length
focal_length = 510

# load model
model = ssdlite320_mobilenet_v3_large(pretrained=True)
model.eval()

transform = T.Compose([T.ToTensor()])

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

dict_objects = {
    'bottle': {'width': 60, 'grasp': 'Power'},
    'cell phone': {'width': 70, 'grasp': 'Power'},
    'banana': {'width': 30, 'grasp': 'Pinch'},
    'apple': {'width': 50, 'grasp': 'Power'},
}


class CurrentObject:
    name = "nothing"
    grasp = "None"
    prev_grasp = "None"
    score = 0
    box = []
    dist = 0
    prev_dist = 0
    vel = 0
    detected = False
    time = 0

    # reset current object atributes
    def resetObject(self, deltaTime, resetGraspTimer):
        self.name = "nothing"
        if self.time < resetGraspTimer: self.time += deltaTime
        else:
            self.time = 0
            self.grasp = "None"
        self.score = 0
        self.box = []
        self.dist= 0
        self.vel = 0

    # choose the object with highest score
    def setObject(self, results, deltaTime, resetGraspTimer):
        self.detected = False

        # get objects atributes
        boxes = results[0]['boxes']
        labels = results[0]['labels']
        scores = results[0]['scores']

        # choose current object
        self.score = 0
        for i in range(len(labels)):
            if classes[labels[i].item()] not in dict_objects:
                continue

            if scores[i].item() > max(self.score, 0.25):
                # set current object atributes
                self.detected = True
                self.score = scores[i].item()
                self.name = classes[labels[i].item()]
                self.box = boxes[i]

        if self.detected:
            # set current object atributes
            self.time = 0
            dict_obj = dict_objects[self.name]
            self.grasp = dict_obj['grasp']
            width = int(self.box[2].item()) - int(self.box[0].item())
            self.dist= (dict_obj['width'] * focal_length)/width
            self.vel = -(self.dist - self.prev_dist)/deltaTime
        else:
            # reset current object atributes
            self.resetObject(deltaTime, resetGraspTimer)

    # set current object previous atributes
    def setPrevObject(self):
        self.prev_grasp = self.grasp
        self.prev_dist = self.dist

curr_obj = CurrentObject()


# convert from ROS to OpenCV image format
def convertFrameFormat(frameMsg):
    np_arr = np.frombuffer(frameMsg, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # rotate image to desirable angle
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Convert image to PIL format for transformation
    frame2 = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    processable_frame = transform(frame2)

    return frame, processable_frame


# publish to ros topics
def publish():
    # grasp detection topic
    dist_vel_array = Float32MultiArray()
    dist_vel_array.data = [curr_obj.dist, curr_obj.vel]
    grasp_detection_pub.publish(dist_vel_array)

    # grasp type topic
    if curr_obj.grasp != "None" and curr_obj.grasp != curr_obj.prev_grasp:
        grasp_type_pub.publish(curr_obj.grasp)


# show image with displayed information
def showImage(frame):
    cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)
    # write informations
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = .7
    text = str(round(curr_obj.score, 3)) + ' ' + curr_obj.name
    cv2.putText(frame, text, (50, 50), font, font_scale,(0,255,0), thickness=2)
    cv2.putText(frame, f'Grasp Type: {curr_obj.grasp}', (50, 100), font, font_scale, (0, 0, 255), thickness=2)
    cv2.putText(frame, f'Distance: {round(curr_obj.dist, 3)}', (50, 150), font, font_scale, (255, 0, 0), thickness=2)
    # draw box
    if curr_obj.name != "nothing":
                startPoint = (int(curr_obj.box[0].item()), int(curr_obj.box[1].item()))
                finishPoint = (int(curr_obj.box[2].item()), int(curr_obj.box[3].item()))
                cv2.rectangle(frame, startPoint, finishPoint, (0, 255, 0))
    # show
    cv2.imshow('frame', frame)
    cv2.waitKey(1)


# function that is called every time the message arrives
def cbImageCompressed(message):
    global frameMsg, newMsg, processing
    if not processing:
        newMsg = True

        frameMsg = message.data


# loop
def loop():
    global newMsg, processing
    rate = rospy.Rate(30)
    previousTime = 0

    while not rospy.is_shutdown():
        if newMsg:
            log()
            # get deltaTime
            currentTime = time.time()
            deltaTime = currentTime - previousTime
            previousTime = currentTime

            # set processing to True
            processing = True
            newMsg = False

            # convert from ROS to OpenCV image format
            frame, processable_frame = convertFrameFormat(frameMsg)
            
            # run inference
            with torch.no_grad():
                results = model([processable_frame])

            # choose the object with highest score
            curr_obj.setObject(results, deltaTime, resetGraspTimer=3)

            # publish to ros topics
            publish()

            # show image with displayed information
            showImage(frame)

            # set current object previous atributes
            curr_obj.setPrevObject()

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