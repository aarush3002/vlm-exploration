#!/usr/bin/env python3

import rospy
import math
import cv2
import numpy as np
from pathlib import Path

# --- MODIFICATION: Import actionlib and MoveBaseAction ---
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseActionGoal

from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Pose, Point
from tf.transformations import euler_from_quaternion

class ExplorationTracker:
    """
    A ROS node to track and analyze an exploration run.

    It tracks:
    1. The sequence of goal points chosen by the assigner.
    2. The full trajectory of the robot.
    3. The total distance traveled and time taken.
    4. Overlays the trajectory onto the map at regular intervals and at the end.
    """
    def __init__(self):
        """Initializes the node, subscribers, and tracking variables."""
        rospy.init_node('exploration_tracker', anonymous=True)

        # --- Parameters ---
        self.completion_timeout = rospy.get_param('~completion_timeout', 70.0)
        self.output_folder = rospy.get_param('~output_folder', 'exploration_maps')
        self.map_topic = rospy.get_param('~map_topic', '/grid_map')
        self.save_interval = rospy.get_param('~save_interval', 30.0)
        self.movement_threshold = rospy.get_param('~movement_threshold', 0.02) # meters

        # --- State and Data Variables ---
        self.goal_points = []
        self.trajectory_points = []
        self.total_distance = 0.0
        self.start_time = None
        self.exploration_finished = False
        self.map_msg = None
        self.save_counter = 0
        self.last_pose = None
        self.last_movement_time = None

        # --- MODIFICATION: Add a client to control move_base ---
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Tracker connecting to move_base action server...")
        if not self.move_base_client.wait_for_server(rospy.Duration(10.0)):
            rospy.logerr("Could not connect to move_base server. Goal canceling will not work.")
        else:
            rospy.loginfo("Tracker connected to move_base server.")


        # --- Subscribers ---
        rospy.Subscriber('/move_base/goal', MoveBaseActionGoal, self.goal_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_callback)

        # --- Timers ---
        rospy.Timer(rospy.Duration(1.0), self.check_completion_callback)
        rospy.Timer(rospy.Duration(self.save_interval), self.timed_map_save_callback)

        rospy.loginfo("Exploration Tracker node started (Stuck-Robot-Detector version).")
        rospy.loginfo(f"Will terminate if robot is stationary for {self.completion_timeout} seconds.")
        rospy.loginfo(f"Output maps will be saved to: {Path(self.output_folder).resolve()}")


    def map_callback(self, msg):
        """Stores the latest map message."""
        if not self.exploration_finished:
            self.map_msg = msg

    def goal_callback(self, msg):
        """Callback for when a new goal is published by the assigner."""
        if self.exploration_finished:
            return

        goal_pos = msg.goal.target_pose.pose.position
        self.goal_points.append((goal_pos.x, goal_pos.y))
        rospy.loginfo(f"Goal {len(self.goal_points)} received: ({goal_pos.x:.2f}, {goal_pos.y:.2f})")

    def timed_map_save_callback(self, event):
        """Periodically saves a map overlay image based on a timer."""
        if self.exploration_finished or self.start_time is None:
            return

        self.save_counter += 1
        filename = f"timed_save_{self.save_counter:03d}.png"
        rospy.loginfo(f"Periodic save triggered, creating {filename}")
        self.generate_map_overlay(filename)

    def odom_callback(self, msg):
        """Callback for odometry. Tracks trajectory and updates last_movement_time."""
        if self.exploration_finished:
            return

        if self.start_time is None:
            self.start_time = rospy.Time.now()
            rospy.loginfo("First odom message received. Starting exploration timer.")
            self.last_movement_time = rospy.Time.now()

        current_pose = msg.pose.pose
        self.trajectory_points.append(current_pose)

        if self.last_pose is not None:
            dx = current_pose.position.x - self.last_pose.position.x
            dy = current_pose.position.y - self.last_pose.position.y
            distance_moved = math.sqrt(dx**2 + dy**2)

            self.total_distance += distance_moved

            if distance_moved > self.movement_threshold:
                self.last_movement_time = rospy.Time.now()

        self.last_pose = current_pose


    def check_completion_callback(self, event):
        """
        Periodically checks for exploration completion.
        If completed, continually cancels all move_base goals.
        """
        # --- MODIFICATION: New logic to continually cancel goals after completion ---
        if self.exploration_finished:
            self.move_base_client.cancel_all_goals()
            rospy.loginfo_throttle(5, "Exploration finished. Continually canceling any active goals.")
            return

        # --- Original completion check logic ---
        if self.last_movement_time is None:
            return

        time_since_last_move = (rospy.Time.now() - self.last_movement_time).to_sec()
        rospy.loginfo_throttle(10, f"Time since last movement: {time_since_last_move:.1f}s")

        if time_since_last_move > self.completion_timeout:
            rospy.loginfo(f"Completion timeout of {self.completion_timeout}s reached. Robot has not moved.")
            self.exploration_finished = True
            self.print_and_save_summary()
            # --- MODIFICATION: Do not shut down the node, let it keep running to cancel goals ---
            # rospy.signal_shutdown("Exploration complete: Robot stationary for timeout period.")

    def generate_map_overlay(self, filename):
        """Generates and saves an image of the map with trajectory and goals."""
        if self.map_msg is None:
            rospy.logwarn("Cannot generate map overlay: No map message received yet.")
            return

        grid = np.array(self.map_msg.data, dtype=np.int8).reshape(
            (self.map_msg.info.height, self.map_msg.info.width))
        res = self.map_msg.info.resolution
        ox, oy = self.map_msg.info.origin.position.x, self.map_msg.info.origin.position.y

        gray_map = np.where(grid == -1, 127, np.where(grid > 50, 0, 255)).astype(np.uint8)
        color_map = cv2.cvtColor(gray_map, cv2.COLOR_GRAY2BGR)

        def world_to_pixel(wx, wy):
            px = int((wx - ox) / res)
            py = int((wy - oy) / res)
            return (px, py)

        # Draw the full robot trajectory (blue line)
        if len(self.trajectory_points) > 1:
            pixel_points = [world_to_pixel(p.position.x, p.position.y) for p in self.trajectory_points]
            pts = np.array(pixel_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(color_map, [pts], isClosed=False, color=(200, 0, 0), thickness=2, lineType=cv2.LINE_AA)

        # Draw the traversal graph (goals as green circles)
        for gx, gy in self.goal_points:
            px, py = world_to_pixel(gx, gy)
            cv2.circle(color_map, (px, py), radius=4, color=(0, 255, 0), thickness=-1)

        # Draw current robot pose (red circle with orange arrow for orientation)
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

        # Save the final image (flipped to match RViz orientation)
        Path(self.output_folder).mkdir(exist_ok=True)
        output_path = Path(self.output_folder) / filename
        cv2.imwrite(str(output_path), cv2.flip(color_map, 0))
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
        cv2.imwrite(str(output_path), cv2.flip(gray_map, 0))
        rospy.loginfo(f"Saved raw occupancy map to: {output_path}")

    def print_and_save_summary(self):
        """Prints a summary, saves metrics to files, and saves final maps."""
        self.generate_map_overlay("exploration_summary_final.png")
        self.save_raw_map("final_occupancy_map.png")

        rospy.loginfo("="*40)
        rospy.loginfo("EXPLORATION RUN SUMMARY")
        rospy.loginfo("="*40)
        
        total_time = (rospy.Time.now() - self.start_time).to_sec() if self.start_time else 0.0
        rospy.loginfo(f"Total Exploration Time: {total_time:.2f} seconds")
        rospy.loginfo(f"Total Distance Traveled: {self.total_distance:.2f} meters")
        rospy.loginfo(f"Number of Goal Points: {len(self.goal_points)}")
        rospy.loginfo("--- Goal Points (x, y) ---")
        for i, point in enumerate(self.goal_points):
            rospy.loginfo(f"  {i+1}: ({point[0]:.2f}, {point[1]:.2f})")
        rospy.loginfo("="*40)

        try:
            output_dir = Path(self.output_folder)
            output_dir.mkdir(exist_ok=True)
            (output_dir / "total_time.txt").write_text(f"{total_time:.2f}\n")
            (output_dir / "total_distance.txt").write_text(f"{self.total_distance:.2f}\n")
            rospy.loginfo(f"Summary metrics saved to {output_dir}")
        except Exception as e:
            rospy.logerr(f"Failed to save summary files: {e}")

if __name__ == '__main__':
    try:
        ExplorationTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Exploration tracker node shut down.")