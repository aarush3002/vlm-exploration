#!/usr/bin/env python
import argparse
import os
import math
import time
from threading import Lock

import numpy as np
import rospy
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

from habitat.config.default import get_config
from habitat.core.simulator import Observations
from src.envs.habitat_eval_rlenv import HabitatEvalRLEnv
from src.utils import utils_logging
from src.utils.utils_visualization import create_clean_map_image
from src.measures.top_down_map_for_roam import add_top_down_map_for_roam_to_config

from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class HabitatTareMapperNode:
    r"""
    A ROS node that listens to odometry from a primary simulation (like TARE Planner)
    and uses Habitat to generate a top-down map and log exploration data.
    This node does NOT run its own active simulation. It is a passive mapper.
    """

    def __init__(
        self,
        node_name: str,
        config_paths: str,
        scene_id: str,
    ):
        r"""
        Initializes the passive mapping node.
        :param node_name: Name of the ROS node.
        :param config_paths: Path to the Habitat environment configuration file.
        :param scene_id: Path to the scene .glb file for Habitat to load.
        """
        # --- ROS Node Initialization ---
        self.node_name = node_name
        rospy.init_node(self.node_name)
        rospy.on_shutdown(self.on_exit_handler)
        self.logger = utils_logging.setup_logger(self.node_name)

        # --- Habitat Configuration ---
        self.config = get_config(config_paths)
        self.config.defrost()
        # Override the scene file in the config with the one provided
        self.config.SIMULATOR.SCENE = scene_id
        # Add the custom top-down map measurement to the config
        self.config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        self.config.freeze()
        add_top_down_map_for_roam_to_config(self.config)

        # --- Habitat Environment Setup ---
        # The environment is instantiated only to load the scene geometry and use its mapping utilities.
        # We will not be calling env.step().
        self.env = HabitatEvalRLEnv(config=self.config, enable_physics=False)
        self.observations = self.env.reset()
        self.logger.info(f"Habitat environment loaded with scene: {scene_id}")

        # --- Data Logging and Plotting Setup ---
        self.log_lock = Lock()
        self.last_position = None
        self.total_distance = 0.0
        self.log_distance_interval = 0.5  # Log data every 0.5 meters
        self.next_log_distance = 0.0
        
        # Define the directory for saving results based on the scene name
        scene_name = os.path.splitext(os.path.basename(scene_id))[0]
        self.home_dir = os.path.join(os.path.expanduser("~"), "tare_planner_data", scene_name)
        self.data_log_filepath = os.path.join(self.home_dir, "distance_vs_exploration.csv")
        self.start_time = rospy.Time.now()

        try:
            os.makedirs(self.home_dir, exist_ok=True)
            with open(self.data_log_filepath, 'w') as f:
                f.write("Distance(m),Exploration(%)\n")
            self.logger.info(f"Logging data to: {self.data_log_filepath}")
        except OSError as e:
            rospy.logerr(f"Could not create output directory or file: {e}")

        # --- ROS Subscriber ---
        # This is the core of the node. It listens to the odometry from the main simulation.
        self.odom_subscriber = rospy.Subscriber(
            "/state_estimation", Odometry, self.state_estimation_callback, queue_size=10
        )
        
        self.logger.info("Mapper node initialized. Waiting for odometry messages on /state_estimation...")

    def state_estimation_callback(self, msg: Odometry):
        """
        Callback function for the /state_estimation topic.
        Updates the agent's position in the Habitat sim and triggers map/data updates.
        """
        with self.log_lock:
            # --- Coordinate Transformation (ROS to Habitat) ---
            # This logic is crucial for correctly placing the agent in the Habitat scene.
            # ROS Frame (FLU): +X Forward, +Y Left, +Z Up
            # Habitat Frame (RUF): +X Right, +Y Up, +Z Forward
            # Transformation: hab_x=ros_y, hab_y=ros_z, hab_z=-ros_x
            # However, the TARE sim seems to use a different convention that matches habitat_online.py
            # TARE (x,y,z) -> Habitat (x, z, -y)
            
            agent_state = self.env._env._sim.get_agent(0).get_state()
            
            # Update position
            ros_position = msg.pose.pose.position
            agent_state.position = np.array([ros_position.x, ros_position.z, -ros_position.y])
            
            # Update rotation
            ros_orientation = msg.pose.pose.orientation
            # The exact rotation transform can be tricky. We adapt from habitat_online_v0.2.1.py
            # which suggests a 90-degree rotation is needed.
            roll, pitch, yaw = euler_from_quaternion([ros_orientation.x, ros_orientation.y, ros_orientation.z, ros_orientation.w])
            
            # This quaternion represents the transformation from the ROS body frame to the Habitat body frame
            # This might require tuning depending on the exact setup.
            q_ros_to_hab = quaternion_from_euler(0, math.radians(90), 0)
            q_ros = [ros_orientation.x, ros_orientation.y, ros_orientation.z, ros_orientation.w]
            
            # We want to set the agent's world orientation, so we just use the raw ROS orientation
            # and let the map measure handle the agent's internal heading calculation.
            agent_state.rotation = np.quaternion(ros_orientation.w, ros_orientation.x, ros_orientation.y, ros_orientation.z)

            # Set the agent's state in the simulator. This "teleports" the agent.
            self.env._env._sim.get_agent(0).set_state(agent_state, infer_sensor_states=False)

            # --- Update Map and Log Data ---
            # Manually trigger the top-down map measurement update
            self.env._env._task.measurements.update_measures(episode=None, action=None)

            # Update total distance traveled
            current_position = np.array([ros_position.x, ros_position.y, ros_position.z])
            if self.last_position is not None:
                distance_increment = np.linalg.norm(current_position - self.last_position)
                self.total_distance += distance_increment

            # Log data if the distance threshold is crossed
            if self.total_distance >= self.next_log_distance:
                self._log_data()
                self.next_log_distance += self.log_distance_interval

            self.last_position = current_position

    def _log_data(self):
        """Logs the current distance and exploration percentage to a file."""
        exploration_percentage = self.get_exploration_percentage()
        if exploration_percentage is None:
            return
        try:
            with open(self.data_log_filepath, 'a') as f:
                f.write(f"{self.total_distance:.4f},{exploration_percentage:.4f}\n")
            rospy.loginfo(f"Logged: Distance={self.total_distance:.2f}m, Exploration={exploration_percentage:.2f}%")
        except IOError as e:
            rospy.logerr(f"Could not write to data log file: {e}")

    def get_exploration_percentage(self):
        """Calculates the exploration percentage from the top-down map."""
        try:
            top_down_map_measure = self.env._env._task.measurements.measures.get("top_down_map_for_roam")
            if top_down_map_measure and hasattr(top_down_map_measure, 'get_clean_map'):
                clean_map_data = top_down_map_measure.get_clean_map()
                if clean_map_data is not None:
                    # Create an image from the map data to analyze coverage
                    clean_map_image = create_clean_map_image(clean_map_data)
                    temp_map_path = os.path.join(self.home_dir, "temp_map_for_analysis.png")
                    cv2.imwrite(temp_map_path, cv2.cvtColor(clean_map_image, cv2.COLOR_RGB2BGR))
                    return self.analyze_map_coverage(temp_map_path)
            return None
        except Exception as e:
            rospy.logerr(f"Error getting exploration percentage: {e}")
            return None

    def analyze_map_coverage(self, image_path):
        """Analyzes a map image to determine the percentage of explored area."""
        if not os.path.exists(image_path):
            rospy.logerr(f"Error: Image file not found at '{image_path}'")
            return None
        map_image = cv2.imread(image_path)
        if map_image is None:
            rospy.logerr(f"Error: Could not read the image file at '{image_path}'.")
            return None
        
        # These color values are defined in habitat.utils.visualizations.maps
        # Explored area is gray, unexplored is dark gray.
        SEEN_COLOR_BGR = np.array([150, 150, 150])
        UNSEEN_COLOR_BGR = np.array([75, 75, 75])
        
        mask_seen = np.all(map_image == SEEN_COLOR_BGR, axis=-1)
        mask_unseen = np.all(map_image == UNSEEN_COLOR_BGR, axis=-1)
        
        seen_pixels = np.count_nonzero(mask_seen)
        unseen_pixels = np.count_nonzero(mask_unseen)
        
        total_explorable_pixels = seen_pixels + unseen_pixels
        return (seen_pixels / total_explorable_pixels) * 100 if total_explorable_pixels > 0 else 0.0

    def on_exit_handler(self):
        """Handles node shutdown, saving all final data and plots."""
        self.logger.info("Executing shutdown sequence...")
        self._save_summary_files()
        self.save_final_map_image()
        self.generate_plot()
        self.logger.info("Shutdown sequence complete.")

    def _save_summary_files(self):
        """Saves final summary files for time and distance."""
        self.logger.info("Saving final summary files...")
        try:
            # 1. Save Total Time
            total_time = (rospy.Time.now() - self.start_time).to_sec()
            time_filepath = os.path.join(self.home_dir, "total_time.txt")
            with open(time_filepath, "w") as f:
                f.write(f"{total_time:.2f}\n")
            self.logger.info(f"Total time ({total_time:.2f}s) saved to {time_filepath}")

            # 2. Save Total Distance
            distance_filepath = os.path.join(self.home_dir, "total_distance.txt")
            with open(distance_filepath, "w") as f:
                f.write(f"{self.total_distance:.4f}\n")
            self.logger.info(f"Total distance ({self.total_distance:.4f}m) saved to {distance_filepath}")

        except Exception as e:
            rospy.logerr(f"Failed to save summary files: {e}")

    def save_final_map_image(self):
        """Saves the final generated top-down map as a PNG image."""
        self.logger.info("Saving final exploration map image...")
        try:
            top_down_map_measure = self.env._env._task.measurements.measures.get("top_down_map_for_roam")
            if top_down_map_measure and hasattr(top_down_map_measure, 'get_clean_map'):
                clean_map_data = top_down_map_measure.get_clean_map()
                if clean_map_data is not None:
                    clean_map_image = create_clean_map_image(clean_map_data)
                    filename = "final_exploration_map.png"
                    file_path = os.path.join(self.home_dir, filename)
                    cv2.imwrite(file_path, cv2.cvtColor(clean_map_image, cv2.COLOR_RGB2BGR))
                    self.logger.info(f"Final clean map saved to {file_path}")
        except Exception as e:
            rospy.logerr(f"Error saving final map image: {e}")

    def generate_plot(self):
        """Generates and saves a plot of exploration percentage vs. distance traveled."""
        self.logger.info("Generating exploration vs. distance plot...")
        try:
            if not os.path.exists(self.data_log_filepath):
                rospy.logwarn(f"Log file not found at {self.data_log_filepath}, skipping plot generation.")
                return
                
            distances, percentages = [], []
            with open(self.data_log_filepath, 'r') as f:
                next(f)  # Skip header
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 2:
                        distances.append(float(parts[0]))
                        percentages.append(float(parts[1]))

            if not distances or not percentages:
                rospy.logwarn("No data to plot.")
                return

            plt.figure(figsize=(10, 6))
            plt.plot(distances, percentages, marker='.', linestyle='-')
            plt.title('Exploration Progress vs. Distance Traveled')
            plt.xlabel('Distance Traveled (m)')
            plt.ylabel('Exploration Percentage (%)')
            plt.grid(True)
            plt.ylim(0, 100)
            plt.xlim(left=0)
            
            plot_filepath = os.path.join(self.home_dir, "exploration_vs_distance_plot.png")
            plt.savefig(plot_filepath)
            rospy.loginfo(f"Plot saved to {plot_filepath}")
            plt.close()

        except Exception as e:
            rospy.logerr(f"Failed to generate plot: {e}")

    def run(self):
        """Keeps the node alive to listen for messages."""
        rospy.spin()


def main():
    parser = argparse.ArgumentParser(description="Passive Habitat mapper for TARE Planner.")
    parser.add_argument("--node-name", type=str, default="habitat_tare_mapper")
    parser.add_argument(
        "--task-config", 
        type=str, 
        required=True,
        help="Path to the Habitat task config file (e.g., pointnav_rgbd_roam.yaml)."
    )
    parser.add_argument(
        "--scene-id", 
        type=str, 
        required=True,
        help="Path to the Matterport3D scene file (.glb)."
    )
    args = parser.parse_args()

    mapper_node = HabitatTareMapperNode(
        node_name=args.node_name,
        config_paths=args.task_config,
        scene_id=args.scene_id,
    )
    mapper_node.run()


if __name__ == "__main__":
    main()
