import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import torch
from torchvision.models.mobilenet import mobilenet_v2
import torchvision.transforms as T
from PIL import Image
import json
import time


window_size = 30
stride = 10
detection_counts = {}
grasp_type = "None"
framesUntilDetection = 15

grasp_mapping = {
    "fountain_pen": 'Pinch',
    "ballpoint": 'Pinch',
    "beer_bottle": 'Cylinder',
    "water_bottle": 'Cylinder',
    "remote_control": 'Point',
    "wine_bottle": 'Cylinder',
    "mouse": 'Mouse Grip',
    "cup": 'Power',
    "coffee_mug": 'Cylinder',
    "Granny_Smith": 'Power',
    "banana": 'Power'
}

start_time = time.time()
object_positions = {}
imgMsg = ""
newMsg = False
processing = False


# Load pre-trained MobileNetV2 model
model = mobilenet_v2(pretrained=True).eval()

# Load class index mappings
with open('/home/bioinlab/Desktop/carlosIgor/computer vision/imagenet_class_index.json') as f:
    raw_class = json.load(f)
with open('/home/bioinlab/Desktop/carlosIgor/computer vision/imagenet_subclass.json') as f:
    new_class = json.load(f)

# Configure confidence threshold
confidence_threshold = 0.5


transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def which_class(raw_class, new_class, idxs, scores):
    idxs = idxs.reshape(-1)
    scores = scores.reshape(-1)
    
    for idx, score in zip(idxs, scores):
        # if score.item() >= confidence_threshold:
        if str(idx.item()) in list(new_class.keys()):
            return new_class[str(idx.item())][1], str(score.item())[:5]

    return 'nothing', 0


# class specific object detection
def objectDetection(img, frame_count):
    pass


# show image with displayed information
def showImage(img, score_final, name_final):
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    text = str(score_final) + ' ' + name_final
    text_offset_x = 50
    text_offset_y = 50
    cv2.putText(img, text, (text_offset_x, text_offset_y), font, font_scale,(0,255,0), thickness=2)
    cv2.putText(img, f'Grasp Type: {grasp_type}', (50, 100), font, font_scale, (0, 0, 255), thickness=2)
    cv2.imshow('RealSense', img)
    cv2.waitKey(1)


# function that is called every time the message arrives
def callbackFunction(message):
    global imgMsg, newMsg, processing
    if not processing:
        newMsg = True

        imgMsg = message.data
        # print the message
        rospy.loginfo("received a video message/frame")


# loop
def loop():
    global grasp_type, newMsg, processing

    frame_count = 0
    rate = rospy.Rate(30)
    
    while not rospy.is_shutdown():
        if newMsg:
            # set processing to True
            processing = True
            newMsg = False

            # increment frame
            frame_count += 1

            # convert from ROS to OpenCV image format
            np_arr = np.fromstring(imgMsg, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Convert image to PIL format for transformation
            frame2 = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            newFrame = transform(frame2)

            # Process the frame with the model
            out = model(torch.Tensor(newFrame).unsqueeze(0))
            out = torch.nn.functional.softmax(out).reshape(-1)
            score, predicted_idx = torch.topk(out, 5)
            name_final, score_final = which_class(raw_class, new_class, predicted_idx, score)
            if score_final == 0:
                score_final = ''

            # Update detection counts and positions for the current object
            if name_final != 'nothing':
                if name_final in detection_counts:
                    detection_counts[name_final] += 1
                    object_positions[name_final].append(frame_count)
                else:
                    detection_counts[name_final] = 1
                    object_positions[name_final] = [frame_count]

            # Check if the window is complete
            if frame_count >= window_size:
                print (grasp_type)
                # Process the detection counts and send grasp types
                sawObj = False
                for obj, count in detection_counts.items():
                    # print("object:", obj)
                    # print("count:", count)
                    if count >= framesUntilDetection:
                        # Send grasp type related to the object 'obj' to the hand
                        # print(f"Sending grasp type for object '{obj}' to the hand")
                        sawObj = True
                        grasp_type = grasp_mapping[obj]
                        # print (grasp_type)

                        # Update the shared variable
                        # shared_var.value = b"grasp_type"
                if not sawObj:
                    grasp_type = "None"

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

            # publish to grasp type topic
            graspDetectionPub.publish(name_final)

            # publish to grasp type topic
            graspTypePub.publish(grasp_type)
            
            # show image with displayed information
            showImage(frame, score_final, name_final)

            # set processing to False
            processing = False
            rate.sleep()

	
if __name__ == "__main__":
    # initialize the subscriber node
    rospy.init_node('camera_sensor_subscriber', anonymous=True)
    
    # create grasp detection topic and publisher
    graspDetectionPub = rospy.Publisher("grasp_detection", String, queue_size=10)

    # create grasp type topic and publisher
    graspTypePub = rospy.Publisher("grasp_type", String, queue_size=10)
    
    # subscribe
    rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, callbackFunction)

    loop()

    cv2.destroyAllWindows()