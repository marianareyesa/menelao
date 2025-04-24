#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import datetime
import os
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from std_msgs.msg import UInt16, Float32

class GripperControl:
    def __init__(self):
        self.servo_pub = rospy.Publisher('/servo', UInt16, queue_size=1)
        rospy.Subscriber('/servoLoad', Float32, self._load_cb)
        self.current_load = 0.0

    def _load_cb(self, msg):
        self.current_load = msg.data

    def open(self):
        rospy.loginfo('Opening gripper')
        self.servo_pub.publish(UInt16(0))

    def close(self):
        rospy.loginfo('Closing gripper')
        self.servo_pub.publish(UInt16(170))

class LineFollower:
    def __init__(self):
        rospy.init_node('smart_line_follower', anonymous=True)
        self.bridge = CvBridge()
        self.twist = Twist()
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

        self.gripper = GripperControl()

        self.lower_black = np.array([0, 0, 0])
        self.upper_black = np.array([180, 255, 80])

        self.lower_red1 = np.array([0, 100, 100])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([160, 100, 100])
        self.upper_red2 = np.array([180, 255, 255])
        self.lower_green = np.array([50, 100, 100])
        self.upper_green = np.array([70, 255, 255])
        self.lower_blue = np.array([100, 100, 100])
        self.upper_blue = np.array([130, 255, 255])
        self.lower_yellow = np.array([25, 100, 100])
        self.upper_yellow = np.array([35, 255, 255])

        self.linear_vel_base = 0.15
        self.angular_vel_base = 0.3
        self.turn_linear = 0.05
        self.kernel = np.ones((5,5), np.uint8)

        self.image_save_dir = '/local/student/catkin_ws/src/menelao_challenge/tmp/'
        os.makedirs(self.image_save_dir, exist_ok=True)
        self.save_interval = 5
        self.last_saved_time = rospy.Time.now()

        self.last_dir = 1
        self.state = "FOLLOW_LINE"
        self.current_puck_color = None

        rospy.loginfo("Smart Line Follower ready.")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("CV Bridge error: %s", e)
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]

        if self.state == "FOLLOW_LINE":
            self.follow_line(hsv, h, w)
            puck_color = self.detect_puck(hsv)
            if puck_color:
                self.current_puck_color = puck_color
                self.state = "APPROACH_PUCK"

        elif self.state == "APPROACH_PUCK":
            puck_color = self.detect_puck(hsv)
            if puck_color:  # still visible and close
                self.move_forward(duration=0.3)
            else:  # lost it = very close
                self.pick_puck()
                if self.current_puck_color == "green":
                    self.state = "TURN_LEFT"
                else:
                    self.state = "DISPLACE_PUCK"


        elif self.state == "DISPLACE_PUCK":
            self.rotate(90)
            self.drop_puck()
            self.rotate(-90)
            self.state = "FOLLOW_LINE"

        elif self.state == "TURN_LEFT":
            self.rotate(90)
            self.state = "SEARCH_YELLOW"

        elif self.state == "SEARCH_YELLOW":
            self.move_forward()
            if self.detect_yellow(hsv):
                self.state = "DROP_GREEN"

        elif self.state == "DROP_GREEN":
            self.drop_puck()
            self.state = "RETURN_TO_LINE"

        elif self.state == "RETURN_TO_LINE":
            self.move_backward()
            self.rotate(-90)
            self.state = "FOLLOW_LINE"

        self.cmd_pub.publish(self.twist)
        cv2.imshow("Camera", hsv)
        cv2.waitKey(1)

        now = rospy.Time.now()
        if now - self.last_saved_time >= rospy.Duration(self.save_interval):
            self.save_image(frame)
            self.last_saved_time = now

    def follow_line(self, hsv, h, w):
        crop = hsv[int(h * 0.75):, :]
        mask = cv2.inRange(crop, self.lower_black, self.upper_black)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10']/M['m00'])
                err = (cx - w/2) / (w/2)
                self.twist.linear.x = self.linear_vel_base * (1 - abs(err))
                self.twist.angular.z = -err * self.angular_vel_base
                if abs(self.twist.angular.z) > 0.1:
                    self.twist.linear.x = self.turn_linear
                self.last_dir = 1 if self.twist.angular.z > 0 else -1
            else:
                self._recover(hsv, w)
        else:
            self._recover(hsv, w)

    def detect_puck(self, hsv):
        def get_largest_contour(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(cnt)
                return (x, y, w, h)
            return None

        red_mask = cv2.inRange(hsv, self.lower_red1, self.upper_red1) | cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        green_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        blue_mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)

        for color, mask in [("red", red_mask), ("green", green_mask), ("blue", blue_mask)]:
            bbox = get_largest_contour(mask)
            if bbox:
                x, y, w, h = bbox
                if h > 40:  # threshold: puck is "close enough"
                    return color
        return None


    def detect_yellow(self, hsv):
        yellow_mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        return cv2.countNonZero(yellow_mask) > 500

    def pick_puck(self):
        rospy.loginfo("Picking up puck")
        self.twist.linear.x = 0.05
        self.twist.angular.z = 0
        self.cmd_pub.publish(self.twist)
        rospy.sleep(1.5)
        self.gripper.close()
        rospy.sleep(1)
        self.twist.linear.x = -0.05
        self.cmd_pub.publish(self.twist)
        rospy.sleep(1.5)

    def drop_puck(self):
        rospy.loginfo("Dropping puck")
        self.twist.linear.x = 0
        self.twist.angular.z = 0
        self.cmd_pub.publish(self.twist)
        rospy.sleep(1)
        self.gripper.open()
        rospy.sleep(1)

    def move_forward(self, duration=1.5):
        self.twist.linear.x = 0.1
        self.twist.angular.z = 0
        self.cmd_pub.publish(self.twist)
        rospy.sleep(duration)

    def move_backward(self, duration=1.5):
        self.twist.linear.x = -0.1
        self.twist.angular.z = 0
        self.cmd_pub.publish(self.twist)
        rospy.sleep(duration)

    def rotate(self, degrees):
        direction = 1 if degrees > 0 else -1
        duration = abs(degrees) / 45.0  # 45° per second approx
        self.twist.linear.x = 0
        self.twist.angular.z = direction * 0.5
        self.cmd_pub.publish(self.twist)
        rospy.sleep(duration)

    def _recover(self, hsv, width):
        mask = cv2.inRange(hsv, self.lower_black, self.upper_black)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, _, w_cnt, _ = cv2.boundingRect(cnt)
            cx = x + w_cnt/2
            self.last_dir = 1 if cx < width/2 else -1
        self.twist.linear.x = 0
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
