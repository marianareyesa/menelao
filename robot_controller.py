#!/usr/bin/env python3
import rospy
import tf2_ros
import numpy as np
import cv2
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, String
from cv_bridge import CvBridge

# rosrun your_package robot_controller.py _target_color:=red

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

        # Robot State
        self.safe_distance = 0.5
        self.target_color = rospy.get_param('~target_color', 'green')  # Default to green
        self.is_handling_object = False
        self.object_detected = False

        # HSV Color Ranges
        self.color_ranges = {
            'black': ([0, 0, 0], [180, 255, 50]),    # Line detection
            'red': ([0, 100, 100], [10, 255, 255]),
            'green': ([40, 100, 100], [80, 255, 255])
        }

        # Object detection parameters
        self.min_object_area = 1000  # Minimum area to consider as an object
        self.max_object_area = 50000  # Maximum area to consider as an object
        self.object_center_threshold = 30  # Pixels from center to consider object centered

    def image_callback(self, data):
        """Processes camera feed for line following & object detection."""
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        height, width, _ = cv_image.shape

        # First priority: Line following
        if not self.is_handling_object:
            # Crop image to focus on line detection area
            line_roi = hsv[int(height * 0.7):height, :]
            lower_black, upper_black = self.color_ranges['black']
            mask_line = cv2.inRange(line_roi, np.array(lower_black), np.array(upper_black))
            M_line = cv2.moments(mask_line)

            if M_line['m00'] > 0:
                cx_line = int(M_line['m10'] / M_line['m00'])
                error = cx_line - width // 2
                self.twist.linear.x = 0.15
                self.twist.angular.z = -float(error) / 200
            else:
                # If line is lost, slow down and search
                self.twist.linear.x = 0.05
                self.twist.angular.z = 0.3

        # Second priority: Object detection
        if not self.is_handling_object:
            for color in ['red', 'green']:
                lower, upper = self.color_ranges[color]
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    # Find the largest contour that meets our size criteria
                    valid_contours = [cnt for cnt in contours 
                                    if self.min_object_area < cv2.contourArea(cnt) < self.max_object_area]
                    
                    if valid_contours:
                        biggest_contour = max(valid_contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(biggest_contour)
                        obj_center_x = x + w // 2
                        offset = obj_center_x - width // 2

                        # Check if object is centered enough
                        if abs(offset) < self.object_center_threshold:
                            self.object_detected = True
                            if color == self.target_color:
                                rospy.loginfo(f"🎯 Target {color.upper()} object detected! Picking up...")
                                self.handle_target_object()
                            else:
                                rospy.loginfo(f"🚫 Non-target {color.upper()} object detected! Pushing away...")
                                self.handle_non_target_object()

        # Publish movement command
        self.cmd_pub.publish(self.twist)

    def handle_target_object(self):
        """Handles picking up the target colored object."""
        self.is_handling_object = True
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.cmd_pub.publish(self.twist)
        
        # Close gripper
        self.gripper_pub.publish(1.0)
        rospy.sleep(2)
        
        # Move to deposit zone
        self.twist.linear.x = 0.3
        self.cmd_pub.publish(self.twist)
        rospy.sleep(3)
        
        # Release object
        self.gripper_pub.publish(0.0)
        rospy.sleep(2)
        
        self.is_handling_object = False
        self.object_detected = False

    def handle_non_target_object(self):
        """Handles pushing away non-target colored objects."""
        self.is_handling_object = True
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.cmd_pub.publish(self.twist)
        
        # Push object away
        self.twist.linear.x = 0.1
        self.cmd_pub.publish(self.twist)
        rospy.sleep(2)
        
        # Back up slightly
        self.twist.linear.x = -0.1
        self.cmd_pub.publish(self.twist)
        rospy.sleep(1)
        
        self.is_handling_object = False
        self.object_detected = False

    def lidar_callback(self, msg):
        """Handles obstacle detection using LiDAR."""
        if self.is_handling_object:
            return  # Ignore LiDAR during object handling
            
        min_distance = min(msg.ranges)
        if min_distance < self.safe_distance:
            rospy.logwarn("🚧 Obstacle detected! Adjusting path...")
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.5  # Turn away
            self.cmd_pub.publish(self.twist)

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
