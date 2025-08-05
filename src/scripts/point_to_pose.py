#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PointStamped, PoseStamped

# Global publisher
goal_pub = None

def point_callback(msg):
    """
    Takes a PointStamped message and converts it to a PoseStamped message
    with a default orientation before publishing it.
    """
    global goal_pub

    # Create a new PoseStamped message
    pose_stamped_msg = PoseStamped()

    # Copy the header and position from the incoming message
    pose_stamped_msg.header = msg.header
    pose_stamped_msg.pose.position = msg.point

    # Set a default orientation (no rotation)
    pose_stamped_msg.pose.orientation.w = 1.0

    # Publish the new PoseStamped message
    goal_pub.publish(pose_stamped_msg)

def converter_node():
    """
    Initializes the ROS node, subscriber, and publisher.
    """
    global goal_pub
    rospy.init_node('point_to_pose_converter', anonymous=True)

    # Publisher for the move_base simple goal
    goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)

    # Subscriber for the detected points from the RRT detector
    rospy.Subscriber("/detected_points", PointStamped, point_callback)

    rospy.loginfo("Point-to-Pose converter node started.")
    rospy.spin()

if __name__ == '__main__':
    try:
        converter_node()
    except rospy.ROSInterruptException:
        pass