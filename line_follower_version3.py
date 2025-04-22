#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import datetime
import os
from collections import deque

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
        self.upper_black = np.array([180, 255, 80])  # widened to handle shadows

        # Movement parameters
        self.linear_vel_base = 0.15
        self.angular_vel_base = 0.5
        self.min_lin = 0.05
        self.min_ang = 0.1
        self.deadband_frac = 0.05  # 5% of frame width
        self.last_turn = 1

        # Centroid smoothing
        self.cx_buffer = deque(maxlen=5)

        # Morphological kernel
        self.kernel = np.ones((5,5), np.uint8)

        # Directory for saved images
        self.image_save_dir = '/local/student/catkin_ws/src/menelao_challenge/tmp/'
        os.makedirs(self.image_save_dir, exist_ok=True)
        self.save_interval = 5
        self.last_saved_time = rospy.Time.now()

        rospy.loginfo("Enhanced line follower node started...")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("CV Bridge error: %s", e)
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        # Crop bottom 30% for line
        crop = hsv[int(h*0.7):, :]

        mask = cv2.inRange(crop, self.lower_black, self.upper_black)
        # Morphological clean-up
        mask = cv2.erode(mask, self.kernel, iterations=1)
        mask = cv2.dilate(mask, self.kernel, iterations=2)

        # Find moments
        M = cv2.moments(mask)
        if M['m00'] > 0:
            raw_cx = int(M['m10']/M['m00'])
            # smooth centroid
            self.cx_buffer.append(raw_cx)
            cx = int(sum(self.cx_buffer)/len(self.cx_buffer))

            # Compute error relative to center
            err = cx - w/2
            # Deadband
            if abs(err) < self.deadband_frac * w:
                err_norm = 0
            else:
                err_norm = err/(w/2)

            # Velocity command via proportional control
            lin = self.linear_vel_base * (1 - abs(err_norm))
            ang = -err_norm * self.angular_vel_base

            # Enforce minimums
            self.twist.linear.x = max(self.min_lin, lin)
            if abs(ang) < self.min_ang and err_norm != 0:
                ang = np.sign(ang) * self.min_ang
            self.twist.angular.z = ang

            self.last_turn = 1 if ang > 0 else -1 if ang < 0 else self.last_turn
        else:
            # Lost line: rotate slowly in last direction
            self.twist.linear.x = 0.0
            self.twist.angular.z = self.min_ang * self.last_turn

        # Publish
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
