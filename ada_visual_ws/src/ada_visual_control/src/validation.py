# all distances in mm
import rospy
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from ada_visual_control.classes.kalmanfilter import KalmanFilter


detect_distance = 80
statesNum = 10

graspMsg = ""
predDist = 0
predVel = 0
predDistArray = []
prevDist = 0
count = 0

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
    if (predDist <= detect_distance) and (predDist != 0):
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