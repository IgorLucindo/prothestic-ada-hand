import numpy as np
import cv2
import rospy
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
# from kalmanfilter import KalmanFilter


detect_distance_mm = 80
statesNum = 5

graspMsg = ""
predDist = 0
predVel = 0
predDistArray = []
prevDist = 0
count = 0


class KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)


    def predict(self):
        ''' This function estimates the position of the object'''
        predicted = self.kf.predict()
        x, y = np.float32(predicted[0][0]), np.float32(predicted[1][0])
        return x, y
    

    def update(self, coordX, coordY):
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)


    def getNextStates(self, statesNum):
        predictions = []
        for _ in range(statesNum):
            # Predict the next state
            prediction = self.kf.predict()
            # Get the updated state estimate
            state = self.kf.statePost
            predictions.append(prediction[0][0])
            # Set the current state as the new initial state for the next prediction
            self.kf.statePre = state
        return predictions


# Kalman Filter
kalmanFilter = KalmanFilter()

predictions = []


# function that is called every time the message arrives
def cbGraspDetection(message):
    global predDist, predVel, predDistArray, prevDist, count
    predDist = 0

    # if there is an object
    if message.data[0]:
        # update and predict
        kalmanFilter.update(message.data[0], message.data[1])
        predDist, predVel = kalmanFilter.predict()

        # reset count when object is detected
        count = 0
    # if there isn't any object
    else:
        # set predicted distances array
        if message.data[0] != prevDist:
            predDistArray = kalmanFilter.getNextStates(statesNum)

            # set 'predDistArray' in predictions array for debug purposes
            predictions.insert(0, predDistArray)
            if len(predictions) > statesNum:
                predictions.pop()
        
        # set 'predDist' according to predicted distance
        if count < statesNum and len(predDistArray):
            predDist = predDistArray[count]
            count += 1

    # set previous message
    prevDist = message.data[0]
    
    # publish distance and predicted distance for debug
    predict = Float32MultiArray()
    predict.data = [message.data[0], predDist]
    predict_pub.publish(predict)

    cb()


# function that is called every time the message arrives
def cbGraspType(message):
    global graspMsg
    graspMsg = message.data

    cb()


def cb():
    # conditions to grasp
    if predDist <= detect_distance_mm and predDist:
        # publish to grasp topic
        grasp_pub.publish(graspMsg)


if __name__ == "__main__":
    # initialize the subscriber node
    rospy.init_node('validation', anonymous=True)

    # create grasp type topic and publisher
    grasp_pub = rospy.Publisher("grasp", String, queue_size=10)

    predict_pub = rospy.Publisher("predict", Float32MultiArray, queue_size=10)
    
    # subscribe
    rospy.Subscriber("grasp_detection", Float32MultiArray, cbGraspDetection)
    rospy.Subscriber("grasp_type", String, cbGraspType)

    rospy.spin()