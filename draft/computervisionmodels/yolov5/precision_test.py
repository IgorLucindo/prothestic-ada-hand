# all distances in mm
import cv2
import os
import time
from deepsparse import Pipeline
from ros.classes.currentObject import CurrentObject
from ros.classes.kalmanfilter import KalmanFilter


# camera focal length
focal_length = 510

# load model
model = Pipeline.create(
    task="yolo",
    model_path="/home/bioinlab/Desktop/carlosIgor/prothestic_ada_hand/ada_visual_ws/src/ada_visual_control/src/models/yolov5-s-coco-pruned75_quantized.onnx",
    class_names="coco",
    num_cores=4,
    image_size=(640, 480)
)

dict_objects = {
    'bottle': {'width': 60, 'grasp': 'Power'},
    'cell phone': {'width': 70, 'grasp': 'Power'},
    'banana': {'width': 30, 'grasp': 'Pinch'},
    'apple': {'width': 50, 'grasp': 'Power'},
}

curr_obj = CurrentObject(dict_objects, focal_length)

detect_distance = 80
statesNum = 10

prevDist = 0
predDistArray = []
count = 0

# Kalman Filter
kalmanFilter = KalmanFilter()

ada_hand_videos_path = "../../ada_hand_videos"

# List all files in the folder
video_folders = os.listdir(ada_hand_videos_path)


# run inference
def runModel(frame):
    # inference
    results = model(images=frame)

    return results


# return true if grasp
def graspDetection():
    global prevDist, predDistArray, count
    predDist = 0

    # if there is an object
    if curr_obj.dist:
        # update and predict
        kalmanFilter.update(curr_obj.dist, curr_obj.vel)
        predDist, _ = kalmanFilter.predict()

        # reset count when object is detected
        count = 0
    # if there isn't any object
    else:
        # set predicted distances array
        if curr_obj.dist != prevDist:
            predDistArray = kalmanFilter.getNextStates(statesNum)
        
        # set 'predDist' according to predicted distance
        if count < statesNum and len(predDistArray):
            predDist = predDistArray[count]
            count += 1

    # set previous message
    prevDist = curr_obj.dist

    return (predDist <= detect_distance) and (predDist != 0)


# loop
def loop(cap):
    previousTime = 0

    while True:
        # get deltaTime
        currentTime = time.time()
        deltaTime = currentTime - previousTime
        previousTime = currentTime

        # get frame
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imshow('frame', frame)

        # run inference
        results = runModel(frame)

        # choose the object with highest score
        curr_obj.setObject(results, deltaTime, resetGraspTimer=3)

        # detects grasp
        if graspDetection():
            return 1

        cv2.waitKey(1)

    return 0


if __name__ == "__main__":
    # Iterate over each video
    for video_folder in video_folders:
        video_folder_path = os.path.join(ada_hand_videos_path, video_folder)
        videos = os.listdir(video_folder_path)

        # set atributes for getting precision
        i = 0
        objectPrecision = 0

        for video in videos:
            video_path = os.path.join(video_folder_path, video)

            # get capture
            cap = cv2.VideoCapture(video_path)

            # set precision for each object
            grasped = loop(cap)
            objectPrecision = (objectPrecision*i + grasped)/(i+1)
            i += 1

            print(video_folder, ': ', round(objectPrecision*100, 2), end='\r')




            
