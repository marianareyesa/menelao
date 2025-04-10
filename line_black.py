#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import datetime
import os

class LineFollower:
    def __init__(self):
        rospy.init_node('line_follower', anonymous=True)
        
        # Initialize CV Bridge and Twist message
        self.bridge = CvBridge()
        self.twist = Twist()

        # Publisher for velocity commands
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # Subscriber for camera images
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

        # Define HSV range for black (adjust these if needed)
        self.lower_black = np.array([0, 0, 0])
        self.upper_black = np.array([180, 255, 50])

        # Create directory to save images if it doesn't exist
        self.image_save_dir = '/tmp/line_follower_images'
        if not os.path.exists(self.image_save_dir):
            os.makedirs(self.image_save_dir)

        # Interval for saving images (in seconds)
        self.save_interval = 5  # Save an image every 5 seconds
        self.last_saved_time = rospy.Time.now()

        rospy.loginfo("Line follower node started, waiting for images...")

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("CV Bridge error: %s", e)
            return

        # Convert image to HSV color space
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        height, width, _ = cv_image.shape

        # Crop the image to the lower half (where the line should be)
        crop_img = hsv[int(height * 0.5):height, :]

        # Create a mask for black color using the defined range
        mask = cv2.inRange(crop_img, self.lower_black, self.upper_black)

        # Use Canny edge detection to detect the edges of the line
        edges = cv2.Canny(mask, threshold1=100, threshold2=200)

        # Calculate moments of the edges to get the center of mass
        M = cv2.moments(edges)
        if M['m00'] > 0:
            # Calculate the centroid (center of mass) of the line
            cx = int(M['m10'] / M['m00'])
            error = cx - (width // 2)
            # Set forward speed and use error for turning
            self.twist.linear.x = 0.15
            self.twist.angular.z = -float(error) / 200.0
        else:
            # If no line is detected, rotate in place to search for it
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.3

        # Publish the command
        self.cmd_pub.publish(self.twist)

        # Optionally, show the masked image and edges for debugging
        cv2.imshow("Mask", mask)
        cv2.imshow("Edges", edges)
        cv2.waitKey(1)

        # Save image every save_interval seconds
        current_time = rospy.Time.now()
        if current_time - self.last_saved_time >= rospy.Duration(self.save_interval):
            self.save_image(cv_image)
            self.last_saved_time = current_time

    def save_image(self, image):
        # Generate a unique filename based on the current time
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        image_filename = os.path.join(self.image_save_dir, f"image_{timestamp}.jpg")
        
        # Save the image
        cv2.imwrite(image_filename, image)
        rospy.loginfo(f"Image saved: {image_filename}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = LineFollower()
        node.run()
    except rospy.ROSInterruptException:
        pass
