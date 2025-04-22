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
        
        # Initialize CV Bridge and Twist
        self.bridge = CvBridge()
        self.twist = Twist()

        # Publisher for velocity commands
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # Subscriber for camera images
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

        # Define HSV range for line detection
        self.lower_black = np.array([0, 0, 0])
        self.upper_black = np.array([180, 255, 80])

        # Movement parameters
        self.linear_vel_base = 0.15  # forward speed when on center
        self.turn_linear = 0.05      # small forward movement during turns
        self.angular_vel_base = 0.3  # turning speed
        self.last_turn = 1           # last turn direction

        # Save images directory
        self.image_save_dir = '/local/student/catkin_ws/src/menelao_challenge/tmp/'
        os.makedirs(self.image_save_dir, exist_ok=True)
        self.save_interval = 5
        self.last_saved_time = rospy.Time.now()

        rospy.loginfo("Basic-zone line follower node started...")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("CV Bridge error: %s", e)
            return

        # Preprocess
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        # Crop bottom 30% of image
        crop = hsv[int(h*0.7):, :]

        # Threshold to detect black
        mask = cv2.inRange(crop, self.lower_black, self.upper_black)

        # Divide mask into three vertical zones
        zone_width = w // 3
        left_zone = mask[:, :zone_width]
        center_zone = mask[:, zone_width:2*zone_width]
        right_zone = mask[:, 2*zone_width:]

        # Sum of white pixels in each zone
        left_sum = np.sum(left_zone)
        center_sum = np.sum(center_zone)
        right_sum = np.sum(right_zone)

        # Decide direction based on max zone
        max_sum = max(left_sum, center_sum, right_sum)
        if max_sum == center_sum and center_sum > 0:
            # Line is centered
            self.twist.linear.x = self.linear_vel_base
            self.twist.angular.z = 0.0
            self.last_turn = 0
        elif max_sum == left_sum and left_sum > 0:
            # Line to left
            self.twist.linear.x = self.turn_linear
            self.twist.angular.z = self.angular_vel_base
            self.last_turn = 1
        elif max_sum == right_sum and right_sum > 0:
            # Line to right
            self.twist.linear.x = self.turn_linear
            self.twist.angular.z = -self.angular_vel_base
            self.last_turn = -1
        else:
            # Line lost: rotate in last direction
            self.twist.linear.x = 0.0
            self.twist.angular.z = self.angular_vel_base * self.last_turn

        # Publish command
        self.cmd_pub.publish(self.twist)

        # Debug display
        cv2.imshow("Mask", mask)
        cv2.waitKey(1)

        # Save image periodically
        now = rospy.Time.now()
        if now - self.last_saved_time >= rospy.Duration(self.save_interval):
            self.save_image(frame)
            self.last_saved_time = now

    def save_image(self, img):
        ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        path = os.path.join(self.image_save_dir, f"img_{ts}.jpg")
        cv2.imwrite(path, img)
        rospy.loginfo(f"Saved {path}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        LineFollower().run()
    except rospy.ROSInterruptException:
        pass
