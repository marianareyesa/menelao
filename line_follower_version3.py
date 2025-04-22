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

        # HSV range for line detection (black)
        self.lower_black = np.array([0, 0, 0])
        self.upper_black = np.array([180, 255, 60])

        # Movement parameters
        self.linear_vel_base = 0.15
        self.turn_linear = 0.08
        self.angular_vel_base = 0.4
        self.min_lin = 0.05

        # Morphological kernel to reduce noise
        self.kernel = np.ones((5,5), np.uint8)

        # Lost-line debounce
        self.lost_count = 0
        self.lost_threshold = 3  # frames before recovery

        # Save images directory
        self.image_save_dir = '/local/student/catkin_ws/src/menelao_challenge/tmp/'
        os.makedirs(self.image_save_dir, exist_ok=True)
        self.save_interval = 5
        self.last_saved_time = rospy.Time.now()

        # Previous turn direction
        self.last_dir = 1

        rospy.loginfo("Line follower v5 with debounce started...")

    def image_callback(self, msg):
        self.twist = Twist()
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("CV Bridge error: %s", e)
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]

        # ROI: bottom 20% for line detection to detect only close edges
        y0 = int(h * 0.8)
        crop = hsv[y0:, :]

        mask_roi = cv2.inRange(crop, self.lower_black, self.upper_black)
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, self.kernel)
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, self.kernel)

        contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected = False
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt) > 200:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10']/M['m00'])
                    err = (cx - w/2) / (w/2)
                    lin = self.linear_vel_base * (1 - abs(err))
                    ang = -err * self.angular_vel_base
                    self.twist.linear.x = max(self.min_lin, lin)
                    self.twist.angular.z = ang
                    if abs(ang) > 0.2:
                        self.twist.linear.x = self.turn_linear
                    self.last_dir = 1 if ang > 0 else -1
                    detected = True

        # Debounce lost-line: only recover after threshold
        if detected:
            self.lost_count = 0
        else:
            self.lost_count += 1

        if self.lost_count >= self.lost_threshold:
            self._recover(hsv, w)

        self.cmd_pub.publish(self.twist)

        # Debug view
        cv2.imshow("ROI Mask", mask_roi)
        cv2.waitKey(1)

        # Periodic save
        now = rospy.Time.now()
        if now - self.last_saved_time >= rospy.Duration(self.save_interval):
            self._save_image(frame)
            self.last_saved_time = now

    def _recover(self, hsv, width):
        full_mask = cv2.inRange(hsv, self.lower_black, self.upper_black)
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, self.kernel)
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, self.kernel)

        mid = width // 2
        left_sum = np.sum(full_mask[:, :mid])
        right_sum = np.sum(full_mask[:, mid:])

        if left_sum > right_sum and left_sum > 1000:
            self.twist.linear.x = self.turn_linear
            self.twist.angular.z = self.angular_vel_base
            self.last_dir = 1
        elif right_sum > left_sum and right_sum > 1000:
            self.twist.linear.x = self.turn_linear
            self.twist.angular.z = -self.angular_vel_base
            self.last_dir = -1
        else:
            self.twist.linear.x = self.turn_linear * 0.5
            self.twist.angular.z = self.angular_vel_base * self.last_dir * 0.5

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
