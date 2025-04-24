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

        # HSV range for line detection
        self.lower_black = np.array([0, 0, 0])
        self.upper_black = np.array([180, 255, 80])

        # Movement parameters
        self.linear_vel_base = 0.15  # forward speed when centered
        self.angular_vel_base = 0.3  # turning speed
        self.turn_linear = 0.05      # small forward during turns

        # Last direction for recovery (-1 right, +1 left)
        self.last_dir = 1

        # Morphological kernel to reduce noise
        self.kernel = np.ones((5,5), np.uint8)

        # Save images directory
        self.image_save_dir = '/local/student/catkin_ws/src/menelao_challenge/tmp/'
        os.makedirs(self.image_save_dir, exist_ok=True)
        self.save_interval = 5
        self.last_saved_time = rospy.Time.now()

        rospy.loginfo("Contour-based adaptive line follower started...")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("CV Bridge error: %s", e)
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]

        # ROI: bottom 25% for main line detection
        y0 = int(h * 0.75)
        crop = hsv[y0:, :]

        # Mask and clean-up
        mask = cv2.inRange(crop, self.lower_black, self.upper_black)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        # Find contours in ROI
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Select the largest contour by area
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if area > 100:  # ignore small blobs
                # Compute centroid of contour
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10']/M['m00'])
                    # Normalize error
                    err = (cx - w/2) / (w/2)
                    # Velocity commands
                    self.twist.linear.x = self.linear_vel_base * (1 - abs(err))
                    self.twist.angular.z = -err * self.angular_vel_base
                    # Minor forward during sharp turns
                    if abs(self.twist.angular.z) > 0.1:
                        self.twist.linear.x = self.turn_linear
                    # Update last_dir
                    self.last_dir = 1 if self.twist.angular.z > 0 else -1
                else:
                    self._recover(hsv, w)
            else:
                self._recover(hsv, w)
        else:
            self._recover(hsv, w)

        # Publish command
        self.cmd_pub.publish(self.twist)

        # Debug view
        cv2.imshow("ROI Mask", mask)
        cv2.waitKey(1)

        # Save image periodically
        now = rospy.Time.now()
        if now - self.last_saved_time >= rospy.Duration(self.save_interval):
            self.save_image(frame)
            self.last_saved_time = now

    def _recover(self, hsv, width):
        # Lost-line: detect branch in full frame
        full_mask = cv2.inRange(hsv, self.lower_black, self.upper_black)
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, self.kernel)
        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # choose largest contour
            cnt = max(contours, key=cv2.contourArea)
            x, _, w_cnt, _ = cv2.boundingRect(cnt)
            cx = x + w_cnt/2
            # Decide direction
            self.last_dir = 1 if cx < width/2 else -1
        # Rotate in place towards last_dir
        self.twist.linear.x = 0.0
        self.twist.angular.z = self.angular_vel_base * self.last_dir

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
