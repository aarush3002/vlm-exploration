#!/usr/bin/env python

import rospy
import math
import cv2
import numpy as np
from pathlib import Path

from nav_msgs.msg import Odometry, OccupancyGrid
from move_base_msgs.msg import MoveBaseActionGoal
from geometry_msgs.msg import Point, Pose
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import MarkerArray

class ExplorationTracker:
    """
    A ROS node to track and analyze an exploration run performed by explore_lite.

    It tracks:
    1. The sequence of goal points chosen by the explorer.
    2. The full trajectory of the robot.
    3. Overlays the trajectory and goal graph onto the map at each new goal.
    """
    def __init__(self):
        """Initializes the node, subscribers, and tracking variables."""
        rospy.init_node('exploration_tracker', anonymous=True)

        # --- Parameters ---
        self.completion_timeout = rospy.get_param('~completion_timeout', 35.0)
        self.output_folder = rospy.get_param('~output_folder', 'exploration_maps')
        self.map_topic = rospy.get_param('~map_topic', '/grid_map')

        # --- State and Data Variables ---
        self.goal_points = []
        self.trajectory_points = [] # Will store geometry_msgs/Pose
        self.exploration_finished = False
        self.map_msg = None
        self.goal_counter = 0 # Counter for naming map overlay files
        self.save_counter = 0
        self.save_interval = rospy.get_param('~save_interval', 30.0)
        self.last_frontier_time = None

        # --- Subscribers ---
        rospy.Subscriber('/move_base/goal', MoveBaseActionGoal, self.goal_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_callback)
        rospy.Subscriber('/explore/frontiers', MarkerArray, self.frontier_callback)

        # --- Timer to check for completion ---
        rospy.Timer(rospy.Duration(1.0), self.check_completion_callback)
        rospy.Timer(rospy.Duration(self.save_interval), self.timed_map_save_callback)

        rospy.loginfo("Exploration Tracker node started.")
        rospy.loginfo(f"Listening for map on topic: {self.map_topic}")
        rospy.loginfo(f"Output maps will be saved to: {Path(self.output_folder).resolve()}")

    def frontier_callback(self, msg):
        if self.exploration_finished: return
        if msg.markers:
            self.last_frontier_time = rospy.Time.now()

    def map_callback(self, msg):
        """Stores the latest map message."""
        if not self.exploration_finished:
            self.map_msg = msg

    def goal_callback(self, msg):
        """
        Callback for when a new goal is published. Logs the goal.
        """
        if self.exploration_finished:
            return

        self.last_goal_time = rospy.Time.now()
        self.goal_counter += 1

        goal_pos = msg.goal.target_pose.pose.position
        self.goal_points.append((goal_pos.x, goal_pos.y))

        rospy.loginfo(f"Goal {self.goal_counter} received: ({goal_pos.x:.2f}, {goal_pos.y:.2f})")

    def timed_map_save_callback(self, event):
        """Periodically saves a map overlay image based on a timer."""
        if self.exploration_finished:
            return

        self.save_counter += 1
        filename = f"timed_save_{self.save_counter:03d}.png"
        rospy.loginfo(f"Periodic save triggered, creating {filename}")
        self.generate_map_overlay(filename)

    def odom_callback(self, msg):
        """Callback for odometry updates. Tracks trajectory."""
        if self.exploration_finished:
            return

        current_pose = msg.pose.pose
        self.trajectory_points.append(current_pose)

    def check_completion_callback(self, event):
        """Periodically checks if the exploration has completed by monitoring frontier topic."""
        if self.exploration_finished or self.last_frontier_time is None:
            return
        time_since_last_frontier = (rospy.Time.now() - self.last_frontier_time).to_sec()
        if time_since_last_frontier > self.completion_timeout:
            self.exploration_finished = True
            self.print_and_save_summary()
            self.generate_map_overlay("exploration_summary_final.png")
            rospy.signal_shutdown("Exploration complete: No new frontiers found.")

    def generate_map_overlay(self, filename):
        """Generates and saves an image of the map with trajectory and goals."""
        if self.map_msg is None:
            rospy.logwarn("Cannot generate map overlay: No map message received yet.")
            return

        # --- Extract map data ---
        grid = np.array(self.map_msg.data, dtype=np.int8).reshape(
            (self.map_msg.info.height, self.map_msg.info.width))
        res = self.map_msg.info.resolution
        ox, oy = self.map_msg.info.origin.position.x, self.map_msg.info.origin.position.y

        # --- Create base image from map ---
        gray_map = np.where(grid == -1, 127, np.where(grid > 50, 0, 255)).astype(np.uint8)
        color_map = cv2.cvtColor(gray_map, cv2.COLOR_GRAY2BGR)

        # --- Helper to convert world coordinates to pixel coordinates ---
        def world_to_pixel(wx, wy):
            px = int((wx - ox) / res)
            py = int((wy - oy) / res)
            return (px, py)

        # --- Draw the full robot trajectory (blue line) ---
        if len(self.trajectory_points) > 1:
            pixel_points = [world_to_pixel(p.position.x, p.position.y) for p in self.trajectory_points]
            pts = np.array(pixel_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(color_map, [pts], isClosed=False, color=(200, 0, 0), thickness=2)

        # --- Draw current robot pose ---
        if self.trajectory_points:
            last_pose = self.trajectory_points[-1]
            rpx, rpy = world_to_pixel(last_pose.position.x, last_pose.position.y)
            q = last_pose.orientation
            _, _, yaw_rad = euler_from_quaternion([q.x, q.y, q.z, q.w])
            cv2.circle(color_map, (rpx, rpy), radius=5, color=(0, 0, 255), thickness=-1)
            arrow_len_px = 0.5 / res
            dx_px = arrow_len_px * math.cos(yaw_rad)
            dy_px = arrow_len_px * math.sin(yaw_rad)
            cv2.arrowedLine(color_map, (rpx, rpy), (int(rpx + dx_px), int(rpy + dy_px)),
                            (0, 165, 255), 2, tipLength=0.3)

        # --- Save the final image ---
        Path(self.output_folder).mkdir(exist_ok=True)
        output_path = Path(self.output_folder) / filename
        cv2.imwrite(str(output_path), cv2.flip(color_map, 1))
        rospy.loginfo(f"Saved map overlay to: {output_path}")

    def save_raw_map(self, filename="final_occupancy_map.png"):
        """Saves the final, raw occupancy grid as an image."""
        if self.map_msg is None:
            rospy.logwarn("Cannot save raw map: No map message received yet.")
            return

        grid = np.array(self.map_msg.data, dtype=np.int8).reshape(
            (self.map_msg.info.height, self.map_msg.info.width))
        
        gray_map = np.where(grid == -1, 127, np.where(grid > 50, 0, 255)).astype(np.uint8)

        output_path = Path(self.output_folder) / filename
        cv2.imwrite(str(output_path), cv2.flip(gray_map, 1))
        rospy.loginfo(f"Saved raw occupancy map to: {output_path}")

    def print_and_save_summary(self):
        """Prints a summary, saves metrics to files, and saves final maps."""
        self.generate_map_overlay("exploration_summary_final.png")
        self.save_raw_map("final_occupancy_map.png")

        rospy.loginfo("="*40)
        rospy.loginfo("EXPLORATION RUN SUMMARY")
        rospy.loginfo("="*40)
        
        rospy.loginfo(f"Number of Goal Points: {len(self.goal_points)}")
        rospy.loginfo("--- Goal Points (x, y) ---")
        for i, point in enumerate(self.goal_points):
            rospy.loginfo(f"  {i+1}: ({point[0]:.2f}, {point[1]:.2f})")
        rospy.loginfo("="*40)

if __name__ == '__main__':
    try:
        ExplorationTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Exploration tracker node shut down.")