#!/usr/bin/env python3
"""
Relay /odom  →  /odom_zero  so the very first pose becomes (0, 0, 0) in map
coordinates *and* yaw = 0 rad.

• Publishes nav_msgs/Odometry on /odom_zero
• Keeps the original child_frame_id so TF remains base_frame‑centric
"""

import copy
import math
import rospy
from nav_msgs.msg import Odometry
from tf.transformations import (
    euler_from_quaternion,
    quaternion_from_euler,
)

class OdomZero:
    def __init__(self):
        self.origin_pos  = None           # (x0, y0, z0)
        self.origin_yaw  = None           # yaw0  [rad]

        self.pub = rospy.Publisher("odom_zero", Odometry, queue_size=50)
        rospy.Subscriber("odom", Odometry, self._cb, queue_size=200)

    # ------------------------------------------------------------------ #
    def _cb(self, msg: Odometry):
        # ----------- capture the very first pose as the new origin ---- #
        if self.origin_pos is None:
            x0 = msg.pose.pose.position.x
            y0 = msg.pose.pose.position.y
            z0 = msg.pose.pose.position.z

            q0 = msg.pose.pose.orientation
            _, _, yaw0 = euler_from_quaternion([q0.x, q0.y, q0.z, q0.w])

            self.origin_pos = (x0, y0, z0)
            self.origin_yaw = yaw0

            rospy.loginfo(
                "odom_zero: origin set to (%.3f, %.3f, %.3f), yaw %.1f °",
                x0, y0, z0, math.degrees(yaw0)
            )
            # (optionally drop the first message instead of publishing it)
            return

        # ----------- build a zero‑based Odometry message -------------- #
        out = copy.deepcopy(msg)          # keep header / child_frame_id

        # --- translation deltas
        dx = msg.pose.pose.position.x - self.origin_pos[0]
        dy = msg.pose.pose.position.y - self.origin_pos[1]
        dz = msg.pose.pose.position.z - self.origin_pos[2]

        out.pose.pose.position.x = dx
        out.pose.pose.position.y = dy
        out.pose.pose.position.z = dz

        # --- yaw delta
        q  = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        dyaw = yaw - self.origin_yaw
        # wrap to (‑π, π]
        dyaw = (dyaw + math.pi) % (2 * math.pi) - math.pi
        q_rel = quaternion_from_euler(0.0, 0.0, dyaw)

        out.pose.pose.orientation.x = q_rel[0]
        out.pose.pose.orientation.y = q_rel[1]
        out.pose.pose.orientation.z = q_rel[2]
        out.pose.pose.orientation.w = q_rel[3]

        # velocities are copied unchanged (you can subtract if you wish)
        self.pub.publish(out)

# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    rospy.init_node("odom_zero")
    OdomZero()
    rospy.spin()
