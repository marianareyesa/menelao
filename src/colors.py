#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from cv_bridge import CvBridge
 #comment

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

        # 🌟 Target color dynamically assigned before competition
        self.target_color = rospy.get_param('color', 'red') 

        # HSV Color Ranges for red, green, and blue
        self.color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'green': ([40, 100, 100], [70, 255, 255]),
            'blue': ([90, 100, 100], [130, 255, 255])
        }

    def image_callback(self, data):
        """Processes camera feed for object detection."""
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Detect objects based on color
        self.detect_objects(hsv, cv_image)

        # Display Camera Feed
        cv2.imshow("Camera Feed", cv_image)  # Show the image in a window called "Camera Feed"
        cv2.waitKey(1)

        # Publish movement command
        self.cmd_pub.publish(self.twist)

    def detect_objects(self, hsv, cv_image):
        """Detect objects (red, green, blue) and handle them."""
        height, width, _ = cv_image.shape

        # Check for each color
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

                    # If it's the target color, pick it up and turn left, else turn right
                    if color == self.target_color:
                        rospy.loginfo(f"🎯 {color.upper()} object detected! Picking it up and turning left...")
                        self.pickup_object()
                        self.turn_left()
                    else:
                        rospy.loginfo(f"🚫 {color.upper()} detected! Picking it up and turning right...")
                        self.pickup_object()
                        self.turn_right()

                    # Draw bounding box for visual feedback
                    cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def pickup_object(self):
        """Activates gripper to pick up the correct object."""
        rospy.loginfo("🤖 Picking up object...")
        self.gripper_pub.publish(1.0)  # Close gripper
        rospy.sleep(2)

    def release_object(self):
        """Releases the object."""
        rospy.loginfo("📤 Releasing object...")
        self.gripper_pub.publish(0.0)  # Open gripper
        rospy.sleep(2)

    def turn_left(self):
        """Turns the robot 90 degrees to the left."""
        rospy.loginfo("🌀 Turning left 90 degrees...")
        self.twist.angular.z = 1.57  # 90 degrees in radians
        self.cmd_pub.publish(self.twist)
        rospy.sleep(2)
        self.release_object()

    def turn_right(self):
        """Turns the robot 90 degrees to the right."""
        rospy.loginfo("🌀 Turning right 90 degrees...")
        self.twist.angular.z = -1.57  # -90 degrees in radians
        self.cmd_pub.publish(self.twist)
        rospy.sleep(2)
        self.release_object()

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = RobotController()
        node.run()
    except rospy.ROSInterruptException:
        pass
