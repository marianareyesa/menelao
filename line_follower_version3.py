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
        self.linear_vel_base = 0.15
        self.min_lin = 0.06
        self.angular_vel_base = 0.4

        # Morphological kernel to reduce noise
        self.kernel = np.ones((5,5), np.uint8)

        # Save images directory
        self.image_save_dir = '/local/student/catkin_ws/src/menelao_challenge/tmp/'
        os.makedirs(self.image_save_dir, exist_ok=True)
        self.save_interval = 5
        self.last_saved_time = rospy.Time.now()

        rospy.loginfo("Contour-based adaptive line follower v3 started...")

    def image_callback(self, msg):
        # Initialize twist for this frame
        self.twist = Twist()
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("CV Bridge error: %s", e)
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]

        # ROI: bottom 35% for main line detection
        y0 = int(h * 0.65)
        crop = hsv[y0:, :]

        # Mask and clean-up
        mask = cv2.inRange(crop, self.lower_black, self.upper_black)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        # Try main ROI
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if area > 200:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10']/M['m00'])
                    err = (cx - w/2) / (w/2)
                    # Straight proportional control
                    lin = self.linear_vel_base * (1 - abs(err))
                    ang = -err * self.angular_vel_base
                    # Minimum linear
                    self.twist.linear.x = max(self.min_lin, lin)
                    self.twist.angular.z = ang
                else:
                    # fallback
                    self._recover(hsv, w)
            else:
                self._recover(hsv, w)
        else:
            self._recover(hsv, w)

        # Publish command
        self.cmd_pub.publish(self.twist)

        # Debug
        cv2.imshow("ROI Mask", mask)
        cv2.waitKey(1)

        # Periodic save
        now = rospy.Time.now()
        if now - self.last_saved_time >= rospy.Duration(self.save_interval):
            self._save_image(frame)
            self.last_saved_time = now

    def _recover(self, hsv, width):
        # When line segment ends (e.g., square corner), use full-frame centroid to find new direction
        full_mask = cv2.inRange(hsv, self.lower_black, self.upper_black)
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, self.kernel)
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, self.kernel)
        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10']/M['m00'])
                err = (cx - width/2) / (width/2)
                # Rotate in place towards the line
                self.twist.linear.x = 0.0
                self.twist.angular.z = -err * self.angular_vel_base
                return
        # If still no contour, rotate slowly to search
        self.twist.linear.x = 0.0
        self.twist.angular.z = self.angular_vel_base * 0.5

    def _save_image(self, img):
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
