#!/usr/bin/env python3
import rospy
import tf2_ros
import numpy as np
import cv2
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, String
from cv_bridge import CvBridge

class RobotController:
    def __init__(self):
        rospy.init_node('robot_controller')

        # Publishers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.gripper_pub = rospy.Publisher('/gripper_servo_controller/command', Float64, queue_size=10)
        self.twist = Twist()
        self.bridge = CvBridge()

        # Subscribers
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        rospy.Subscriber('/detected_object', String, self.object_callback)

        # TF Buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Robot Size (For obstacle avoidance)
        self.safe_distance = 0.5  

        # Target color can be red or green
        self.target_color = rospy.get_param('~target_color', 'red') 

        # HSV Color Ranges
        # Black range is for line detection; red/green for object detection
        self.color_ranges = {
            'black': ([0, 0, 0], [180, 255, 50]),   # Line detection
            'red': ([0, 100, 100], [10, 255, 255]),
            'green': ([40, 100, 100], [70, 255, 255])
        }

    def image_callback(self, data):
        """Processes camera feed for line following & object detection."""
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Crop Image to Exclude Gripper Interference
        height, width, _ = cv_image.shape
        cropped_hsv = hsv[int(height * 0.6):height, :]

        # 1️⃣ Line Following (Black Line)
        lower_black, upper_black = self.color_ranges['black']
        mask_line = cv2.inRange(cropped_hsv, np.array(lower_black), np.array(upper_black))
        M_line = cv2.moments(mask_line)

        if M_line['m00'] > 0:
            cx_line = int(M_line['m10'] / M_line['m00'])
            error = cx_line - width // 2
            self.twist.linear.x = 0.15
            self.twist.angular.z = -float(error) / 200
        else:
            # If line is lost, rotate to search for it
            self.twist.linear.x = 0
            self.twist.angular.z = 0.3

        # 2️⃣ Object Detection (Red, Green)
        # Loop through just 'red' and 'green' instead of 'blue'
        for color in ['red', 'green']:
            lower, upper = self.color_ranges[color]
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                biggest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(biggest_contour)

                # Only process significant objects
                if area > 600:
                    x, y, w, h = cv2.boundingRect(biggest_contour)
                    obj_center_x = x + w // 2
                    offset = obj_center_x - width // 2

                    # If it’s the target color, move it to the deploy zone
                    if color == self.target_color:
                        rospy.loginfo(f"🎯 {color.upper()} object detected! Moving to deploy zone...")
                        self.pickup_object()
                    else:
                        # If it's not the target color, push it away
                        rospy.loginfo(f"🚫 {color.upper()} is NOT the target! Pushing it away...")
                        self.push_object(offset)

        # Publish movement command
        self.cmd_pub.publish(self.twist)

    def push_object(self, offset):
        """Pushes non-target objects out of the way."""
        if abs(offset) > 20:
            self.twist.angular.z = -0.002 * offset  # Align to object
        else:
            self.twist.angular.z = 0

        # Move forward slowly to push
        self.twist.linear.x = 0.05
        self.cmd_pub.publish(self.twist)
        rospy.sleep(2)
        self.twist.linear.x = 0.0
        self.cmd_pub.publish(self.twist)

    def lidar_callback(self, msg):
        """Handles obstacle detection using LiDAR."""
        min_distance = min(msg.ranges)

        if min_distance < self.safe_distance:
            rospy.logwarn("🚧 Obstacle detected! Adjusting path...")
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.5  # Turn away
        else:
            # Continue moving if no obstacle is too close
            self.twist.linear.x = 0.15

        self.cmd_pub.publish(self.twist)

    def pickup_object(self):
        """Activates gripper to pick up the correct object."""
        rospy.loginfo("🤖 Picking up object...")
        self.gripper_pub.publish(1.0)  # Close gripper
        rospy.sleep(2)
        self.deposit_object()

    def deposit_object(self):
        """Moves robot to deposit zone and releases object."""
        rospy.loginfo("📦 Moving to deploy zone...")
        self.twist.linear.x = 0.3
        self.cmd_pub.publish(self.twist)
        rospy.sleep(3)

        rospy.loginfo("📤 Releasing object...")
        self.gripper_pub.publish(0.0)  # Open gripper
        rospy.sleep(2)

    def object_callback(self, msg):
        """Optional callback if you have a separate node publishing detected object info."""
        rospy.loginfo(f"Object callback received: {msg.data}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = RobotController()
        node.run()
    except rospy.ROSInterruptException:
        pass
