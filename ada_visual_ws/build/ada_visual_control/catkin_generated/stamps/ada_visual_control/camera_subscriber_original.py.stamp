import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
import cv2


topicName = 'video_topic'

# function that is called every time the message arrives
def callbackFunction(message):
	# print the message
	rospy.loginfo("received a video message/frame")
	# convert from ROS to OpenCV image format
	np_arr = np.fromstring(message.data, np.uint8)
	frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
	
	cv2.imshow("camera", frame)
	cv2.waitKey(1)
	
# initialize the subscriber node
rospy.init_node('camera_sensor_subscriber', anonymous=True)
# subscribe
rospy.Subscriber(topicName, CompressedImage, callbackFunction)

rospy.spin()

cv2.destroyAllWindows()
