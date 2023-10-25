import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
import cv2


topicName = 'video_topic'

# initialize the node
rospy.init_node('camera_sensor_publisher', anonymous=True)
# create video topic and publisher
publisher = rospy.Publisher(topicName, CompressedImage, queue_size=60)

rate = rospy.Rate(30)

# create the video capture object
cap = cv2.VideoCapture(0)


while not rospy.is_shutdown():
	ret, frame = cap.read()
	
	if ret:
		# print the message
		rospy.loginfo("video frame captured and published")
		# convert from OpenCV to ROS image format
		compressedFrame = CompressedImage()
		compressedFrame.header.stamp = rospy.Time.now()
		compressedFrame.format = "jpeg"
		compressedFrame.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
		# publish
		publisher.publish(compressedFrame)
		
	rate.sleep()