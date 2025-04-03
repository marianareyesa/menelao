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

        # 🌟 Target color dynamically assigned before competition
        self.target_color = rospy.get_param('color', 'red') 

        # HSV Color Ranges
        self.color_ranges = {
            'black': ([0, 0, 0], [180, 255, 50]),  # Line detection
            'red': ([0, 100, 100], [10, 255, 255]),
            'green': ([40, 100, 100], [70, 255, 255]),
            'blue': ([90, 100, 100], [130, 255, 255])
        }

    # def object_callback(self, msg):
    #     detected_object = msg.data.lower()
    #     rospy.loginfo(f"Detected object: {detected_object}")
    #     # If an object is detected and is the target color, pickup the object
    #     if detected_object == self.target_color:
    #         rospy.loginfo(f"🎯 Target {detected_object.upper()} detected!")
    #         self.pickup_object()
    #     else:
    #         rospy.loginfo(f"🚫 {detected_object.upper()} is not the target color.")

    def image_callback(self, data):
        """Processes camera feed for line following & object detection."""
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Crop Image to Exclude Gripper Interference
        height, width, _ = cv_image.shape
        cropped_hsv = hsv[int(height * 0.6):height, :]

        # 🔹 1️⃣ Line Following (Black Line)
        lower_black, upper_black = self.color_ranges['black']
        mask_line = cv2.inRange(cropped_hsv, np.array(lower_black), np.array(upper_black))
        M_line = cv2.moments(mask_line)

        # Prioritize line following
        if M_line['m00'] > 0:
            cx_line = int(M_line['m10'] / M_line['m00'])
            error = cx_line - width // 2
            self.twist.linear.x = 0.15  # Move forward slowly
            self.twist.angular.z = -float(error) / 200  # Adjust angular velocity based on line error
        else:
            self.twist.linear.x = 0
            self.twist.angular.z = 0.3  # Rotate to find line again if not detected

        for color in ['red', 'green']
        

        # 🔹 2️⃣ Object Detection (Red, Green, Blue) - Secondary Task
        self.detect_objects(hsv, cv_image)

        # 🔹 3️⃣ Display Camera Feed
        cv2.imshow("Camera Feed", cv_image)  # Show the image in a window called "Camera Feed"
        cv2.waitKey(1)

        # Publish movement command
        self.cmd_pub.publish(self.twist)

    def detect_objects(self, hsv, cv_image):
        """Detect objects (red, green, blue) and handle them."""
        height, width, _ = cv_image.shape

        for color in ['red', 'green', 'blue']:
            lower, upper = self.color_ranges[color]
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                biggest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(biggest_contour)

                if area > 600:  # Only process significant objects
                    x, y, w, h = cv2.boundingRect(biggest_contour)
                    obj_center_x = x + w // 2
                    offset = obj_center_x - width // 2

                    # If it's the target color, pick it up
                    if color == self.target_color:
                        rospy.loginfo(f"🎯 {color.upper()} object detected! Moving to deploy zone...")
                        self.pickup_object()
                    else:
                        rospy.loginfo(f"🚫 {color.upper()} is NOT the target! Pushing it away...")
                        self.push_object(offset)

                    # Draw bounding box for visual feedback
                    cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def push_object(self, offset):
        """Pushes non-target objects out of the way."""
        if abs(offset) > 20:
            self.twist.angular.z = -0.002 * offset  # Align to object
        else:
            self.twist.angular.z = 0

        # Push object aside
        self.twist.linear.x = 0.05  # Move forward slowly
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
            self.twist.linear.x = 0.15  # Continue moving

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

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = RobotController()
        node.run()
    except rospy.ROSInterruptException:
        pass
