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
        rospy.init_node('line_follower_fixed', anonymous=True)
        
        # Initialize CV Bridge and Twist message
        self.bridge = CvBridge()
        self.twist = Twist()

        # Publisher for velocity commands
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # Subscriber for camera images
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

        # Define HSV range for black line detection (adjust if needed)
        self.lower_black = np.array([0, 0, 0])
        self.upper_black = np.array([180, 255, 50])

        # Movement parameters
        self.linear_vel_base = 0.15       # forward speed when centered
        self.angular_vel_base = 0.5       # max turning speed
        self.min_lin = 0.05               # minimum forward speed
        self.min_ang = 0.1                # minimum turning speed
        self.last_turn = 0                # last turn direction: 1 left, -1 right

        # Create directory to save images if it doesn't exist
        self.image_save_dir = '/local/student/catkin_ws/src/menelao_challenge/tmp/'
        os.makedirs(self.image_save_dir, exist_ok=True)

        # Interval for saving images (in seconds)
        self.save_interval = 5  # Save an image every 5 seconds
        self.last_saved_time = rospy.Time.now()

        rospy.loginfo("Fixed line follower node started, waiting for images...")

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("CV Bridge error: %s", e)
            return

        # Convert image to HSV and crop bottom for line
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        height, width = hsv.shape[:2]
        crop_img = hsv[int(height * 0.8):, :]

        # Mask for black line
        mask = cv2.inRange(crop_img, self.lower_black, self.upper_black)

        # Moments to find centroid of line
        M = cv2.moments(mask)
        if M['m00'] > 0:
            # Centroid in the cropped region
            cx = int(M['m10'] / M['m00'])
            # Adjust error relative to full width
            error = (cx - width / 2)
            # Normalize error to [-1,1]
            error_norm = error / (width / 2)

            # Proportional control for linear and angular velocities
            lin = self.linear_vel_base * (1 - abs(error_norm))
            ang = -error_norm * self.angular_vel_base

            # Enforce minimum speeds
            self.twist.linear.x = max(self.min_lin, lin)
            # If turning, ensure minimum angular rate
            if abs(ang) < self.min_ang:
                ang = np.sign(ang) * self.min_ang
            self.twist.angular.z = ang

            # Update last turn direction
            self.last_turn = 1 if ang > 0 else -1 if ang < 0 else 0
        else:
            # No line detected: search by rotating in last known direction
            self.twist.linear.x = 0.0
            if self.last_turn >= 0:
                self.twist.angular.z = -self.angular_vel_base
            else:
                self.twist.angular.z = self.angular_vel_base

        # Publish velocity command
        self.cmd_pub.publish(self.twist)

        # Optional: display for debugging
        cv2.imshow("Mask", mask)
        cv2.waitKey(1)

        # Save image at interval
        current_time = rospy.Time.now()
        if current_time - self.last_saved_time >= rospy.Duration(self.save_interval):
            self.save_image(cv_image)
            self.last_saved_time = current_time

    def save_image(self, image):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        image_filename = os.path.join(self.image_save_dir, f"image_{timestamp}.jpg")
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
