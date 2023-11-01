# all distances in mm
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
import cv2
import torch
from deepsparse import compile_model
from ada_visual_control.utils.deepsparse_utils import (
    modify_yolo_onnx_input_shape,
    postprocess_nms
)
from typing import Union
import time
from ada_visual_control.classes.currentObject import CurrentObject


image_shape = (416, 416)
# camera focal length
focal_length = 510

# load model
model_filepath, _ = modify_yolo_onnx_input_shape("/home/bioinlab/Desktop/carlosIgor/prothestic_ada_hand/ada_visual_ws/src/ada_visual_control/src/models/householdobjects.onnx", image_shape)
model = compile_model(model_filepath, batch_size=1)

frameMsg = ""
newMsg = False
processing = False

classes = [
    'bottle'
]

dict_objects = {
    'bottle': {'width': 60, 'grasp': 'Power'},
    'cell phone': {'width': 70, 'grasp': 'Power'},
    'banana': {'width': 30, 'grasp': 'Pinch'},
    'apple': {'width': 50, 'grasp': 'Power'},
}

curr_obj = CurrentObject(dict_objects, classes, focal_length)


def _preprocess_batch(batch: np.ndarray) -> Union[np.ndarray, torch.Tensor]:
    if len(batch.shape) == 3:
        batch = batch.reshape(1, *batch.shape)
    batch = np.ascontiguousarray(batch)
    return batch


# convert from ROS to OpenCV image format
def convertFrameFormat(frameMsg):
    np_arr = np.frombuffer(frameMsg, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # rotate image to desirable angle
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame2 = cv2.resize(frame, image_shape)
    processable_frame = frame2[:, :, ::-1].transpose(2, 0, 1)

    return frame, processable_frame


# run inference
def runModel(processable_frame):
    # pre processing
    batch = _preprocess_batch(processable_frame)
    # inference
    results = model.run([batch])
    # NMS
    results = postprocess_nms(results[0])[0]
    # get atributes
    boxes = results[:, 0:4]
    labels = results[:, 5].astype(int)
    scores = results[:, 4]

    return boxes, labels, scores


# publish to ros topics
def publish():
    # grasp detection topic
    dist_vel_array = Float32MultiArray()
    dist_vel_array.data = [curr_obj.dist, curr_obj.vel]
    grasp_detection_pub.publish(dist_vel_array)

    # grasp type topic
    if curr_obj.grasp != curr_obj.prev_grasp:
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
            boxes, labels, scores = runModel(processable_frame)

            # choose the object with highest score
            curr_obj.setObject(boxes, labels, scores, deltaTime, resetGraspTimer=3)

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