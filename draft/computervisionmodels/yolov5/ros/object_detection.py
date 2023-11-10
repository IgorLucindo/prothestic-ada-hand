# all distances in mm
import numpy as np
import rospy
import cv2
import time
from deepsparse import Pipeline
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from ada_visual_control.classes.currentObject import CurrentObject


# camera focal length
focal_length = 510

# load model
model = Pipeline.create(
    task="yolo",
    model_path="models/yolov5-s-coco-pruned75_quantized.onnx",
    class_names="coco",
    num_cores=4,
    image_size=(640, 480)
)

frameMsg = ""
newMsg = False
processing = False

dict_objects = {
    'bottle': {'width': 60, 'grasp': 'Power'},
    'cell phone': {'width': 70, 'grasp': 'Power'},
    'banana': {'width': 30, 'grasp': 'Pinch'},
    'apple': {'width': 50, 'grasp': 'Power'},
}

curr_obj = CurrentObject(dict_objects, focal_length)


# convert from ROS to OpenCV image format
def convertFrameFormat(frameMsg):
    np_arr = np.frombuffer(frameMsg, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # rotate image to desirable angle
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return frame


# run inference
def runModel(frame):
    # inference
    results = model(images=frame)

    return results


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
            frame = convertFrameFormat(frameMsg)
            
            # run inference
            results = runModel(frame)

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