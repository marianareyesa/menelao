#!/usr/bin/env python3
import rospy
from std_msgs.msg import UInt16, Float32

class GripperControl:
    def __init__(self):
        # Assumes rospy.init_node() was already called by the launcher or main_controller
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

    def set_position(self, angle):
        angle = max(0, min(170, angle))
        rospy.loginfo(f'Setting gripper to {angle}')
        self.servo_pub.publish(UInt16(angle))

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    # Only initialize the node when this script is run directly
    rospy.init_node('gripper_controller')
    gc = GripperControl()
    gc.run()
