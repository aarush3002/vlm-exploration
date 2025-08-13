#!/usr/bin/env python
import rospy
import math
import argparse
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

class TrajectoryReplayer:
    """
    Reads a trajectory file, applies a coordinate transformation, and publishes
    Twist commands to make a robot follow the path in a ROS-based simulation.
    """
    def __init__(self, trajectory_file, transform_coords=False, max_linear_speed=0.5, max_angular_speed=1.0, k_linear=1.5, k_angular=1.5):
        """
        Initializes the TrajectoryReplayer node.
        :param trajectory_file: Path to the trajectory file.
        :param transform_coords: If True, negates x, y, and yaw of waypoints.
        :param max_linear_speed: Maximum forward speed of the robot.
        :param max_angular_speed: Maximum turning speed of the robot.
        :param k_linear: Proportional gain for linear speed.
        :param k_angular: Proportional gain for angular speed.
        """
        rospy.init_node('trajectory_replayer')

        # --- Parameters ---
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.distance_threshold = 0.2  # How close to get to a waypoint (meters)
        self.k_linear = k_linear
        self.k_angular = k_angular

        # --- ROS Publisher & Subscriber ---
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/odom', Odometry, self._odom_callback)

        # --- State Variables ---
        self.current_pose = None
        self.waypoints = self._load_trajectory(trajectory_file, transform_coords)
        self.target_waypoint_index = 0

        rospy.loginfo(f"Loaded {len(self.waypoints)} waypoints from {trajectory_file}.")
        if transform_coords:
            rospy.loginfo("Coordinate transformation has been applied (x, y, yaw negated).")

    def _load_trajectory(self, filepath, transform):
        """
        Loads a trajectory file and optionally transforms the coordinates.
        Format: x, y, z, roll, pitch, yaw, timestamp
        """
        try:
            # We only need x, y, and yaw (columns 0, 1, and 5)
            data = np.loadtxt(filepath, usecols=(0, 1, 5))
            
            waypoints = []
            for row in data:
                x, y, yaw = row[0], row[1], row[2]
                if transform:
                    # Apply the reflection for the TARE -> Habitat coordinate systems
                    x = -x
                    y = -y
                    yaw = -yaw
                waypoints.append({"x": x, "y": y, "yaw": yaw})
            
            return waypoints
        except IOError as e:
            rospy.logerr(f"Error reading trajectory file: {e}")
            rospy.signal_shutdown("Failed to load trajectory file.")
            return []

    def _odom_callback(self, msg):
        """
        Updates the robot's current position and orientation from odometry data.
        """
        position = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        
        _, _, yaw_rad = euler_from_quaternion([
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w
        ])
        
        self.current_pose = {
            "x": position.x,
            "y": position.y,
            "yaw": yaw_rad
        }

    def run(self):
        """
        Main control loop to drive the robot along the trajectory.
        """
        rate = rospy.Rate(20) # 20 Hz control loop

        # Wait until we have an initial pose
        while not rospy.is_shutdown() and self.current_pose is None:
            rospy.loginfo("Waiting for initial odometry message...")
            rate.sleep()

        rospy.loginfo("Starting trajectory replay.")

        while not rospy.is_shutdown() and self.target_waypoint_index < len(self.waypoints):
            target = self.waypoints[self.target_waypoint_index]
            
            # Calculate distance and angle to the target waypoint
            dx = target["x"] - self.current_pose["x"]
            dy = target["y"] - self.current_pose["y"]
            distance_to_target = math.sqrt(dx**2 + dy**2)
            angle_to_target = math.atan2(dy, dx)

            # Check if we've reached the current waypoint
            if distance_to_target < self.distance_threshold:
                self.target_waypoint_index += 1
                rospy.loginfo(f"Waypoint {self.target_waypoint_index - 1} reached. Moving to next.")
                if self.target_waypoint_index >= len(self.waypoints):
                    break # End of trajectory
                continue

            # --- Proportional Controller ---
            # Calculate the error between our current yaw and the angle to the target
            angle_error = angle_to_target - self.current_pose["yaw"]
            # Normalize the angle error to be within [-pi, pi]
            while angle_error > math.pi:
                angle_error -= 2 * math.pi
            while angle_error < -math.pi:
                angle_error += 2 * math.pi

            # Calculate desired velocities
            linear_vel = self.k_linear * distance_to_target
            angular_vel = self.k_angular * angle_error

            # Clamp velocities to their maximum values
            linear_vel = min(linear_vel, self.max_linear_speed)
            angular_vel = max(min(angular_vel, self.max_angular_speed), -self.max_angular_speed)

            # If the angle error is large, reduce forward speed to prioritize turning.
            # This creates smoother "arc" turns instead of stopping completely.
            if abs(angle_error) > math.radians(45): # 45 degrees
                linear_vel = linear_vel * 0.2 # Reduce speed significantly while turning

            # Create and publish the Twist message
            twist_msg = Twist()
            twist_msg.linear.x = linear_vel
            twist_msg.angular.z = angular_vel
            self.cmd_pub.publish(twist_msg)

            rate.sleep()

        # Stop the robot at the end of the trajectory
        self.cmd_pub.publish(Twist())
        rospy.loginfo("Trajectory replay finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Replay a TARE planner trajectory in a ROS simulation.")
    parser.add_argument(
        "trajectory_file",
        type=str,
        help="Path to the trajectory_xxx.txt file."
    )
    parser.add_argument(
        "--transform_coords",
        action="store_true",
        help="Negate x, y, and yaw from the trajectory file to match Habitat's coordinate system."
    )
    parser.add_argument("--max_linear_speed", type=float, default=2.4, help="Maximum linear speed (m/s).")
    parser.add_argument("--max_angular_speed", type=float, default=1.5, help="Maximum angular speed (rad/s).")
    parser.add_argument("--k_linear", type=float, default=1.5, help="Proportional gain for linear velocity.")
    parser.add_argument("--k_angular", type=float, default=2.0, help="Proportional gain for angular velocity.")
    
    args = parser.parse_args()

    try:
        replayer = TrajectoryReplayer(
            trajectory_file=args.trajectory_file, 
            transform_coords=args.transform_coords,
            max_linear_speed=args.max_linear_speed,
            max_angular_speed=args.max_angular_speed,
            k_linear=args.k_linear,
            k_angular=args.k_angular
        )
        replayer.run()
    except rospy.ROSInterruptException:
        pass