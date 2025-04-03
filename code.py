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
        rospy.init_node('robot_controller', anonymous=True)

        # Publishers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.gripper_pub = rospy.Publisher('/gripper_servo_controller/command', Float64, queue_size=10)
        self.twist = Twist()
        self.bridge = CvBridge()

        # Subscribers
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        rospy.Subscriber('/detected_object', String, self.object_callback)

        # TF Buffer (for potential transforms)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Parameters and color thresholds
        self.safe_distance = 0.5  # meters for LiDAR obstacle avoidance
        self.target_color = rospy.get_param('~target_color', 'red')  # target object color
        self.color_ranges = {
            'black': ([0, 0, 0], [180, 255, 50]),   # Used for line detection
            'red': ([0, 100, 100], [10, 255, 255]),
            'green': ([40, 100, 100], [70, 255, 255])
        }

        # State variables for behavior control
        self.state = 'line_following'
        self.last_state_change = rospy.Time.now()
        self.line_detected = False
        self.line_error = 0
        self.push_offset = 0

        # Timer to manage state-based actions (10 Hz)
        rospy.Timer(rospy.Duration(0.1), self.state_manager)

    def image_callback(self, data):
        """Processes camera feed for line following and colored object detection."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            rospy.logerr("CV Bridge error: {}".format(e))
            return

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        height, width, _ = cv_image.shape

        # --- Line Detection ---
        # Crop to the lower portion of the image to minimize interference.
        cropped_hsv = hsv[int(height * 0.6):height, :]
        lower_black, upper_black = self.color_ranges['black']
        mask_line = cv2.inRange(cropped_hsv, np.array(lower_black), np.array(upper_black))
        M_line = cv2.moments(mask_line)

        if M_line['m00'] > 0:
            cx_line = int(M_line['m10'] / M_line['m00'])
            self.line_detected = True
            self.line_error = cx_line - (width // 2)
        else:
            self.line_detected = False
            self.line_error = 0

        # --- Colored Object Detection ---
        # Only process objects when in the default (line-following) state.
        if self.state == 'line_following':
            for color in ['red', 'green']:
                lower, upper = self.color_ranges[color]
                mask_color = cv2.inRange(hsv, np.array(lower), np.array(upper))
                contours, _ = cv2.findContours(mask_color, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    if area > 600:  # Only process significant objects
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        obj_center_x = x + w // 2
                        offset = obj_center_x - (width // 2)
                        if color == self.target_color:
                            rospy.loginfo(f"🎯 {color.upper()} object detected! Initiating pickup sequence.")
                            self.state = 'pickup_object'
                            self.last_state_change = rospy.Time.now()
                            break
                        else:
                            rospy.loginfo(f"🚫 {color.upper()} object detected! Initiating push sequence.")
                            self.push_offset = offset
                            self.state = 'push_object'
                            self.last_state_change = rospy.Time.now()
                            break

    def lidar_callback(self, msg):
        """Processes LiDAR data for obstacle avoidance."""
        # Filter out invalid readings.
        valid_ranges = [r for r in msg.ranges if msg.range_min <= r <= msg.range_max]
        if valid_ranges:
            min_distance = min(valid_ranges)
            if min_distance < self.safe_distance and self.state != 'avoid_obstacle':
                rospy.logwarn("🚧 Obstacle detected! Initiating avoidance maneuver.")
                self.previous_state = self.state  # Save current state
                self.state = 'avoid_obstacle'
                self.last_state_change = rospy.Time.now()

    def object_callback(self, msg):
        """Optional callback for externally provided object information."""
        rospy.loginfo(f"Object callback received: {msg.data}")

    def state_manager(self, event):
        """State machine handling robot behavior."""
        current_time = rospy.Time.now()

        if self.state == 'line_following':
            # Follow the line if detected; otherwise, rotate to search for it.
            if self.line_detected:
                self.twist.linear.x = 0.15
                self.twist.angular.z = -float(self.line_error) / 200.0
            else:
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.3
            self.cmd_pub.publish(self.twist)

        elif self.state == 'push_object':
            # Push the non-target object for a fixed duration.
            if (current_time - self.last_state_change) < rospy.Duration(2.0):
                self.twist.linear.x = 0.05
                self.twist.angular.z = -0.002 * self.push_offset
                self.cmd_pub.publish(self.twist)
            else:
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.0
                self.cmd_pub.publish(self.twist)
                self.state = 'line_following'
                self.last_state_change = current_time

        elif self.state == 'pickup_object':
            # Simulate picking up the target object by closing the gripper.
            if (current_time - self.last_state_change) < rospy.Duration(2.0):
                self.gripper_pub.publish(1.0)  # Close gripper
            else:
                # After pickup, transition back to line following.
                self.state = 'line_following'
                self.last_state_change = current_time

        elif self.state == 'avoid_obstacle':
            # Rotate to avoid the obstacle.
            if (current_time - self.last_state_change) < rospy.Duration(1.0):
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.5
                self.cmd_pub.publish(self.twist)
            else:
                # Return to previous state (or default to line following).
                self.state = getattr(self, 'previous_state', 'line_following')
                self.last_state_change = current_time

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        controller = RobotController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
