import rospy
import cv2
import cv_bridge
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

class Follower:
    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        # Subscriber for camera image
        self.image_sub = rospy.Subscriber('/mybot/camera1/image_raw', 
                                          Image, self.image_callback)
        # Publisher for velocity commands
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.twist = Twist()

    def image_callback(self, msg):
        # Convert the ROS image message to OpenCV image format
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # Convert image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define the HSV range for black color (you can adjust these values as needed)
        lower_black = np.array([0, 0, 0])   # Black: low saturation and value
        upper_black = np.array([180, 255, 50])  # Allow low brightness, high hue

        # Create a mask for black color
        mask = cv2.inRange(hsv, lower_black, upper_black)
        
        # Image dimensions
        h, w, d = image.shape
        # Define the region to search for the line (bottom portion of the image)
        search_top = 3 * h / 4
        search_bot = 3 * h / 4 + 20
        mask[0:search_top, 0:w] = 0
        mask[search_bot:h, 0:w] = 0
        
        # Find moments to calculate the center of mass of the line
        M = cv2.moments(mask)
        if M['m00'] > 0:
            # Calculate the center of mass of the detected line
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # Draw a circle at the center of the line (for debugging)
            # cv2.circle(image, (cx, cy), 20, (0, 0, 255), -1)

            # Calculate the error (distance from the center)
            err = cx - w / 2
            self.twist.linear.x = 0.2   # Move forward at a constant speed
            self.twist.angular.z = -float(err) / 100  # Turn to correct the error
            
            # Publish velocity command
            self.cmd_vel_pub.publish(self.twist)
        else:
            # If no line is detected, stop the robot
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.0
            self.cmd_vel_pub.publish(self.twist)

        # Display the mask and the original image with line detection
        cv2.imshow("Mask", mask)
        cv2.imshow("Output", image)
        cv2.waitKey(3)

# Initialize the ROS node
rospy.init_node('follower')
follower = Follower()
rospy.spin()
