#!/usr/bin/env python
import argparse
from threading import Condition, Lock

import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from habitat.config.default import get_config
from habitat.core.simulator import Observations
from ros_x_habitat.msg import PointGoalWithGPSCompass, DepthImage
from ros_x_habitat.srv import EvalEpisode, ResetAgent, GetAgentTime, Roam
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header, Int16
from src.constants.constants import (
    EvalEpisodeSpecialIDs,
    NumericalMetrics,
    PACKAGE_NAME,
    ServiceNames,
)
from src.envs.habitat_eval_rlenv import HabitatEvalRLEnv
from src.evaluators.habitat_sim_evaluator import HabitatSimEvaluator
import time
from src.utils import utils_logging
from src.utils.utils_visualization import generate_video, observations_to_image_for_roam, create_clean_map_image
from src.measures.top_down_map_for_roam import (
    TopDownMapForRoam,
    add_top_down_map_for_roam_to_config,
)

from nav_msgs.msg import Odometry
import tf2_ros
import geometry_msgs.msg

from rosgraph_msgs.msg import Clock

from tf.transformations import (
    quaternion_from_euler,
    quaternion_multiply,
    euler_from_quaternion,
    quaternion_from_matrix,
    quaternion_matrix
)

import math
from sensor_msgs.msg import Imu 

import os
import cv2

from habitat.sims.habitat_simulator.habitat_simulator import AgentState
import quaternion

# START: Matplotlib backend fix
# Force matplotlib to use a non-GUI backend to prevent Qt errors in headless environments.
# This must be done before importing pyplot.
import matplotlib
matplotlib.use('Agg')
# END: Matplotlib backend fix
import matplotlib.pyplot as plt


class HabitatEnvNode:
    r"""
    A class to represent a ROS node with a Habitat simulator inside.
    The node subscribes to agent command topics, and publishes sensor
    readings to sensor topics.
    """

    def __init__(
        self,
        node_name: str,
        config_paths: str = None,
        enable_physics_sim: bool = False,
        use_continuous_agent: bool = False,
        pub_rate: float = 5.0,
    ):
        r"""
        Instantiates a node incapsulating a Habitat sim environment.
        :param node_name: name of the node
        :param config_paths: path to Habitat env config file
        :param enable_physics_sim: if true, turn on dynamic simulation
            with Bullet
        :param use_continuous_agent: if true, the agent would be one
            that produces continuous velocities. Must be false if using
            discrete simulator
        :pub_rate: the rate at which the node publishes sensor readings
        """
        # precondition check
        if use_continuous_agent:
            assert enable_physics_sim

        # initialize node
        self.node_name = node_name
        rospy.init_node(self.node_name)

        rospy.on_shutdown(self.on_exit_generate_video)

        # set up environment config
        self.config = get_config(config_paths)
        # embed top-down map in config
        self.config.defrost()
        self.config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        self.config.freeze()
        add_top_down_map_for_roam_to_config(self.config)

        # instantiate environment
        self.enable_physics_sim = enable_physics_sim
        self.use_continuous_agent = use_continuous_agent

        # Initialize variables to store commanded velocities from /cmd_vel
        self.current_cmd_vx = 0.0
        self.current_cmd_vy = 0.0
        self.current_cmd_vz = 0.0
        self.current_cmd_wx = 0.0
        self.current_cmd_wy = 0.0
        self.current_cmd_wz = 0.0

        # overwrite env config if physics enabled
        if self.enable_physics_sim:
            HabitatSimEvaluator.overwrite_simulator_config(self.config)
        # define environment
        self.env = HabitatEvalRLEnv(
            config=self.config, enable_physics=self.enable_physics_sim
        )

        # shutdown is set to true by eval_episode() to indicate the
        # evaluator wants the node to shutdown
        self.shutdown_lock = Lock()
        with self.shutdown_lock:
            self.shutdown = False

        # enable_eval is set to true by eval_episode() to allow
        # publish_sensor_observations() and step() to run
        self.all_episodes_evaluated = False
        self.enable_eval = False
        self.enable_eval_cv = Condition()

        # enable_reset is set to true by eval_episode() or roam() to allow
        # reset() to run
        self.enable_reset_cv = Condition()
        with self.enable_reset_cv:
            self.enable_reset = False
            self.enable_roam = False
            self.episode_id_last = None
            self.scene_id_last = None

        # agent velocities/action and variables to keep things synchronized
        self.command_cv = Condition()
        with self.command_cv:
            if self.use_continuous_agent:
                self.linear_vel = None
                self.angular_vel = None
            else:
                self.action = None
            self.count_steps = None
            self.new_command_published = False

        self.observations = None

        # timing variables and guarding lock
        self.timing_lock = Lock()
        with self.timing_lock:
            self.t_reset_elapsed = None
            self.t_sim_elapsed = None
            self.start_time = None # For total wall-clock time

        # video production variables
        self.make_video = False
        self.observations_per_episode = []
        self.video_frame_counter = 0
        self.video_frame_period = 1  # NOTE: frame rate defined as x steps/frame

        # set up logger
        self.logger = utils_logging.setup_logger(self.node_name)

        # establish evaluation service server
        self.eval_service = rospy.Service(
            f"{PACKAGE_NAME}/{node_name}/{ServiceNames.EVAL_EPISODE}",
            EvalEpisode,
            self.eval_episode,
        )

        # establish roam service server
        self.roam_service = rospy.Service(
            f"{PACKAGE_NAME}/{node_name}/{ServiceNames.ROAM}", Roam, self.roam
        )

        # define the max rate at which we publish sensor readings
        self.pub_rate = float(pub_rate)

        # environment publish and subscribe queue size
        self.sub_queue_size = 10
        self.pub_queue_size = 10

        # publish to sensor topics
        if "RGB_SENSOR" in self.config.SIMULATOR.AGENT_0.SENSORS:
            self.pub_rgb = rospy.Publisher("rgb", Image, queue_size=self.pub_queue_size)
        if "DEPTH_SENSOR" in self.config.SIMULATOR.AGENT_0.SENSORS:
            if self.use_continuous_agent:
                self.pub_depth = rospy.Publisher(
                    "depth", Image, queue_size=self.pub_queue_size
                )
                self.pub_camera_info = rospy.Publisher(
                    "camera_info", CameraInfo, queue_size=self.pub_queue_size
                )
            else:
                self.pub_depth = rospy.Publisher(
                    "depth", DepthImage, queue_size=self.pub_queue_size
                )
        if "POINTGOAL_WITH_GPS_COMPASS_SENSOR" in self.config.TASK.SENSORS:
            self.pub_pointgoal_with_gps_compass = rospy.Publisher(
                "pointgoal_with_gps_compass",
                PointGoalWithGPSCompass,
                queue_size=self.pub_queue_size
            )
        
        self.pub_odom   = rospy.Publisher("odom", Odometry, queue_size=self.pub_queue_size)
        self.pub_imu  = rospy.Publisher("imu/data", Imu,  queue_size=self.pub_queue_size)
        self.clock_pub = rospy.Publisher('/clock', Clock, queue_size=10)
        self.tf_br      = tf2_ros.TransformBroadcaster()   

        # subscribe from command topics
        if self.use_continuous_agent:
            self.sub = rospy.Subscriber(
                "cmd_vel", Twist, self.callback, queue_size=self.sub_queue_size
            )
        else:
            self.sub = rospy.Subscriber(
                "action", Int16, self.callback, queue_size=self.sub_queue_size
            )
        
        # Distance and logging variables
        self.last_position = None
        self.total_distance = 0.0
        self.log_distance_interval = 0.5
        self.next_log_distance = 0.0
        
        experiment_name = "JmbYfDe2QKZ"
        #self.home_dir = f"/home/aarush/final_gemini_results/{experiment_name}"
        # self.home_dir = f"/home/aarush/final_explore_lite_results/{experiment_name}"
        #self.home_dir = f"/home/aarush/final_opencv_results/{experiment_name}"
        #self.home_dir = f"/home/aarush/final_tare_results/{experiment_name}"
        self.home_dir = f"/home/aarush/final_dsv_results/{experiment_name}"
        self.data_log_filepath = os.path.join(self.home_dir, "distance_vs_exploration.txt")

        # Ensure output directory exists and create the log file
        try:
            os.makedirs(self.home_dir, exist_ok=True)
            with open(self.data_log_filepath, 'w') as f:
                f.write("# Distance(m), Exploration(%)\n")
        except OSError as e:
            rospy.logerr(f"Could not create output directory or file: {e}")

        # wait until connections with the agent is established
        self.logger.info("env making sure agent is subscribed to sensor topics...")
        while (
            self.pub_rgb.get_num_connections() == 0
            or self.pub_depth.get_num_connections() == 0
            or self.pub_pointgoal_with_gps_compass.get_num_connections() == 0
        ):
            pass

        self.logger.info("env initialized")

    def reset(self):
        r"""
        Resets the agent and the simulator.
        """
        with self.enable_reset_cv:
            while self.enable_reset is False:
                self.enable_reset_cv.wait()

            self.enable_reset = False

            with self.shutdown_lock:
                if self.shutdown:
                    return

            if self.episode_id_last != EvalEpisodeSpecialIDs.REQUEST_NEXT:
                last_ep_found = False
                while not last_ep_found:
                    try:
                        self.env.reset()
                        e = self.env._env.current_episode
                        if (str(e.episode_id) == str(self.episode_id_last)) and (
                            e.scene_id == self.scene_id_last
                        ):
                            self.logger.info(
                                f"Last episode found: episode-id={self.episode_id_last}, scene-id={self.scene_id_last}"
                            )
                            last_ep_found = True
                    except StopIteration:
                        self.logger.info("Last episode not found!")
                        raise StopIteration
            else:
                pass

            with self.timing_lock:
                self.t_reset_elapsed = 0.0
                self.t_sim_elapsed = 0.0
                self.start_time = None # Reset wall-clock time

            t_reset_start = time.clock()
            self.observations = self.env.reset()
            t_reset_end = time.clock()
            with self.timing_lock:
                self.t_reset_elapsed += t_reset_end - t_reset_start

            # ================================================================== #
            # === START: ADD THIS CODE TO FORCE A (0,0,0) STARTING POSE ======== #
            # ================================================================== #

            # Define the desired starting position [x, y, z] and orientation (as a quaternion)
            # NOTE: In Habitat's coordinate system, Y is the vertical axis.
            # start_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            # start_rotation = quaternion.from_euler_angles(np.array([0.0, -math.pi / 2, 0.0])) # 0 Yaw

            # # Get the agent and create a new state object
            # agent = self.env._env._sim.get_agent(0)
            # new_state = AgentState(position=start_position, rotation=start_rotation)

            # # Set the agent's state to the new starting pose
            # agent.set_state(new_state)

            # # Refresh observations from the new pose
            # self.observations = self.env._env._sim.get_sensor_observations()

            # self.logger.info("Agent start pose has been manually set to (0, 0, 0) with 0 yaw.")

            # ================================================================== #
            # === END: ADDED CODE ============================================== #
            # ================================================================== #

            with self.command_cv:
                self.count_steps = 0
            
            # Reset distance tracking and log initial state
            self.total_distance = 0.0
            self.last_position = None
            self.next_log_distance = 0.0
            self._log_data()
            self.next_log_distance += self.log_distance_interval

    def _enable_reset(self, request, enable_roam):
        with self.enable_reset_cv:
            self.episode_id_last = str(request.episode_id_last)
            self.scene_id_last = str(request.scene_id_last)
            assert self.enable_reset is False
            self.enable_reset = True
            self.enable_roam = enable_roam
            self.enable_reset_cv.notify()

    def _enable_evaluation(self):
        with self.enable_eval_cv:
            assert self.enable_eval is False
            self.enable_eval = True
            self.enable_eval_cv.notify()

    def eval_episode(self, request):
        resp = {
            "episode_id": EvalEpisodeSpecialIDs.RESPONSE_NO_MORE_EPISODES,
            "scene_id": "",
            NumericalMetrics.DISTANCE_TO_GOAL: 0.0,
            NumericalMetrics.SUCCESS: 0.0,
            NumericalMetrics.SPL: 0.0,
            NumericalMetrics.NUM_STEPS: 0,
            NumericalMetrics.SIM_TIME: 0.0,
            NumericalMetrics.RESET_TIME: 0.0,
        }

        if str(request.episode_id_last) == EvalEpisodeSpecialIDs.REQUEST_SHUTDOWN:
            with self.shutdown_lock:
                self.shutdown = True
            with self.enable_reset_cv:
                self.enable_reset = True
                self.enable_reset_cv.notify()
            return resp
        else:
            self._enable_reset(request=request, enable_roam=False)
            self._enable_evaluation()
            with self.enable_eval_cv:
                while self.enable_eval is True:
                    self.enable_eval_cv.wait()

                if self.all_episodes_evaluated is False:
                    resp = {
                        "episode_id": str(self.env._env.current_episode.episode_id),
                        "scene_id": str(self.env._env.current_episode.scene_id),
                    }
                    metrics = self.env._env.get_metrics()
                    metrics_dic = {
                        k: metrics[k]
                        for k in [
                            NumericalMetrics.DISTANCE_TO_GOAL,
                            NumericalMetrics.SUCCESS,
                            NumericalMetrics.SPL,
                        ]
                    }
                    with self.timing_lock:
                        with self.command_cv:
                            metrics_dic[NumericalMetrics.NUM_STEPS] = self.count_steps
                            metrics_dic[NumericalMetrics.SIM_TIME] = (
                                self.t_sim_elapsed / self.count_steps if self.count_steps > 0 else 0
                            )
                            metrics_dic[
                                NumericalMetrics.RESET_TIME
                            ] = self.t_reset_elapsed
                    resp.update(metrics_dic)
                else:
                    self.all_episodes_evaluated = False
                return resp

    def roam(self, request):
        self._enable_reset(request=request, enable_roam=True)
        self.make_video = request.make_video
        self.video_frame_period = request.video_frame_period
        self._enable_evaluation()
        return True

    def cv2_to_depthmsg(self, depth_img: np.ndarray):
        if self.use_continuous_agent:
            assert self.config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH is False
            # depth_img_in_m = np.squeeze(depth_img, axis=2)
            if depth_img.ndim == 3:
                depth_img_in_m = np.squeeze(depth_img, axis=2)
            else:
                # If it's already 2D (H, W), use it directly
                depth_img_in_m = depth_img
            depth_msg = CvBridge().cv2_to_imgmsg(
                depth_img_in_m.astype(np.float32), encoding="passthrough"
            )
        else:
            depth_msg = DepthImage()
            depth_msg.height, depth_msg.width, _ = depth_img.shape
            depth_msg.step = depth_msg.width
            depth_msg.data = np.ravel(depth_img)
        return depth_msg

    def obs_to_msgs(self, observations_hab: Observations):
        observations_ros = {}
        t_curr = rospy.Time.now()

        for sensor_uuid, sensor_data in observations_hab.items():
            if sensor_uuid == "rgb":
                # sensor_msg = CvBridge().cv2_to_imgmsg(
                #     sensor_data.astype(np.uint8), encoding="rgb8"
                # )
                rgb_image = cv2.cvtColor(sensor_data, cv2.COLOR_RGBA2RGB)
                sensor_msg = CvBridge().cv2_to_imgmsg(
                    rgb_image.astype(np.uint8), encoding="rgb8"
                )
            elif sensor_uuid == "depth":
                sensor_msg = self.cv2_to_depthmsg(sensor_data)
            elif sensor_uuid == "pointgoal_with_gps_compass":
                sensor_msg = PointGoalWithGPSCompass()
                sensor_msg.distance_to_goal = sensor_data[0]
                sensor_msg.angle_to_goal = sensor_data[1]
            else:
                continue

            h = Header()
            h.stamp = t_curr
            h.frame_id = "laser" 
            sensor_msg.header = h
            observations_ros[sensor_uuid] = sensor_msg
        return observations_ros
    
    def _publish_gt_odom(self):
        st = self.env._env._sim.get_agent_state()
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_frame"

        clock_msg = Clock()
        clock_msg.clock = odom.header.stamp
        self.clock_pub.publish(clock_msg)

        odom.pose.pose.position.x = -st.position[0]
        odom.pose.pose.position.y = st.position[2]
        odom.pose.pose.position.z = st.position[1]

        q_hab_to_ros_basis = quaternion_from_euler(math.radians(-90), 0.0, math.radians(90), axes='sxyz')
        q_h = [st.rotation.x, st.rotation.y, st.rotation.z, st.rotation.w]
        q_ros = quaternion_multiply(q_hab_to_ros_basis, q_h)
        _, _, yaw = euler_from_quaternion(q_ros, axes="sxyz")
        q_flat = quaternion_from_euler(0.0, 0.0, -yaw)
        
        odom.pose.pose.orientation.x = q_flat[0]
        odom.pose.pose.orientation.y = q_flat[1]
        odom.pose.pose.orientation.z = q_flat[2]
        odom.pose.pose.orientation.w = q_flat[3]

        R_hab_to_ros_basis = quaternion_matrix(q_hab_to_ros_basis)[:3,:3]
        v_h = np.array(st.velocity)
        v_ros = R_hab_to_ros_basis @ v_h
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = v_ros

        w_h = np.array(st.angular_velocity)
        w_ros = R_hab_to_ros_basis @ w_h
        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = -w_ros[2]

        self.pub_odom.publish(odom)

        tf = geometry_msgs.msg.TransformStamped()
        tf.header, tf.child_frame_id = odom.header, odom.child_frame_id
        tf.transform.translation = odom.pose.pose.position
        tf.transform.rotation = odom.pose.pose.orientation
        self.tf_br.sendTransform(tf)

        imu = Imu()
        imu.header = odom.header
        imu.orientation, imu.angular_velocity = odom.pose.pose.orientation, odom.twist.twist.angular
        imu.linear_acceleration_covariance[0] = -1.0
        for i in (0, 4, 8):
            imu.orientation_covariance[i] = 1e-3
            imu.angular_velocity_covariance[i] = 1e-3
        self.pub_imu.publish(imu)

        # Update distance and log based on distance interval
        current_position = odom.pose.pose.position
        if self.last_position is None:
            self.last_position = current_position
            
        dx = current_position.x - self.last_position.x
        dy = current_position.y - self.last_position.y
        distance_increment = math.sqrt(dx**2 + dy**2)
        self.total_distance += distance_increment

        if self.total_distance >= self.next_log_distance:
            self._log_data()
            self.next_log_distance += self.log_distance_interval

        self.last_position = current_position

    def publish_sensor_observations(self):
        observations_ros = self.obs_to_msgs(self.observations)
        for sensor_uuid, sensor_msg in observations_ros.items():
            if sensor_uuid == "rgb":
                self.pub_rgb.publish(sensor_msg)
            elif sensor_uuid == "depth":
                self.pub_depth.publish(sensor_msg)
                if self.use_continuous_agent:
                    self.pub_camera_info.publish(
                        self.make_depth_camera_info_msg(
                            sensor_msg.header, sensor_msg.height, sensor_msg.width
                        )
                    )
            elif sensor_uuid == "pointgoal_with_gps_compass":
                self.pub_pointgoal_with_gps_compass.publish(sensor_msg)
        self._publish_gt_odom()

    def make_depth_camera_info_msg(self, header, height, width):
        camera_info_msg = CameraInfo()
        camera_info_msg.header = header
        fx, fy = width / 2.0, height / 2.0
        cx, cy = width / 2.0, height / 2.0
        camera_info_msg.width, camera_info_msg.height = width, height
        camera_info_msg.distortion_model = "plumb_bob"
        camera_info_msg.K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        camera_info_msg.D = [0, 0, 0, 0, 0]
        camera_info_msg.P = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]
        return camera_info_msg

    def step(self):
        with self.command_cv:
            self.new_command_published = False
            t_sim_start = time.clock()
            if self.use_continuous_agent:
                self.env.set_agent_velocities(self.linear_vel, self.angular_vel)
                (self.observations, _, _, info) = self.env.step()
            else:
                (self.observations, _, _, info) = self.env.step(self.action)
            t_sim_end = time.clock()
            with self.timing_lock:
                self.t_sim_elapsed += t_sim_end - t_sim_start

        if self.make_video and self.video_frame_counter % self.video_frame_period == 0:
            out_im_per_action = observations_to_image_for_roam(
                self.observations, info, self.config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
            )
            self.observations_per_episode.append(out_im_per_action)
        self.video_frame_counter += 1
        with self.command_cv:
            self.count_steps += 1

    def publish_and_step_for_eval(self):
        r = rospy.Rate(self.pub_rate)
        with self.enable_eval_cv:
            while self.enable_eval is False:
                self.enable_eval_cv.wait()
            while not self.env._env.episode_over:
                self.publish_sensor_observations()
                self.step()
                r.sleep()
            self.enable_eval = False
            self.enable_eval_cv.notify()

    def publish_and_step_for_roam(self):
        r = rospy.Rate(self.pub_rate)
        with self.enable_eval_cv:
            while self.enable_eval is False:
                self.enable_eval_cv.wait()
            
            with self.timing_lock:
                if self.start_time is None:
                    self.start_time = rospy.Time.now()

            while True:
                with self.shutdown_lock:
                    if self.shutdown:
                        break
                self.publish_sensor_observations()
                self.step()
                r.sleep()
            self.enable_eval = False
    
    def _log_data(self):
        exploration_percentage = self.get_exploration_percentage()
        if exploration_percentage is None:
            return
        try:
            with open(self.data_log_filepath, 'a') as f:
                f.write(f"{self.total_distance:.4f}, {exploration_percentage:.4f}\n")
            rospy.loginfo(f"Logged: Distance={self.total_distance:.2f}m, Exploration={exploration_percentage:.2f}%")
        except IOError as e:
            rospy.logerr(f"Could not write to data log file: {e}")

    def get_exploration_percentage(self):
        try:
            top_down_map_measure = self.env._env._task.measurements.measures.get("top_down_map_for_roam")
            if top_down_map_measure and hasattr(top_down_map_measure, 'get_clean_map'):
                clean_map_data = top_down_map_measure.get_clean_map()
                if clean_map_data is not None:
                    clean_map_image = create_clean_map_image(clean_map_data)
                    temp_map_path = os.path.join(self.home_dir, "temp_map_for_analysis.png")
                    cv2.imwrite(temp_map_path, cv2.cvtColor(clean_map_image, cv2.COLOR_RGB2BGR))
                    return self.analyze_map_coverage(temp_map_path)
            return None
        except Exception as e:
            rospy.logerr(f"Error getting exploration percentage: {e}")
            return None

    def callback(self, cmd_msg):
        if self.use_continuous_agent:
            if isinstance(cmd_msg, Twist):
                self.linear_vel = np.array([cmd_msg.linear.y, 0.0, -cmd_msg.linear.x])
                self.angular_vel = np.array([0.0, cmd_msg.angular.z, 0.0])
            else:
                rospy.logwarn("Expected Twist message for continuous agent.")
        else:
            self.action = cmd_msg.data

        with self.command_cv:
            self.new_command_published = True
            self.command_cv.notify()

    def simulate(self):
        while True:
            try:
                self.reset()
                with self.shutdown_lock:
                    if self.shutdown:
                        rospy.signal_shutdown("received request to shut down")
                        break
                with self.enable_reset_cv:
                    if self.enable_roam:
                        self.publish_and_step_for_roam()
                    else:
                        self.publish_and_step_for_eval()
            except StopIteration:
                with self.enable_reset_cv:
                    self.enable_reset = False
                with self.enable_eval_cv:
                    self.all_episodes_evaluated = True
                    self.env.reset_episode_iterator()
                    self.enable_eval = False
                    self.enable_eval_cv.notify()

    def analyze_map_coverage(self, image_path):
        if not os.path.exists(image_path):
            rospy.logerr(f"Error: Image file not found at '{image_path}'")
            return None
        map_image = cv2.imread(image_path)
        if map_image is None:
            rospy.logerr(f"Error: Could not read the image file at '{image_path}'.")
            return None
        SEEN_COLOR_BGR = np.array([150, 150, 150])
        UNSEEN_COLOR_BGR = np.array([75, 75, 75])
        mask_seen = np.all(map_image == SEEN_COLOR_BGR, axis=-1)
        mask_unseen = np.all(map_image == UNSEEN_COLOR_BGR, axis=-1)
        seen_pixels = np.count_nonzero(mask_seen)
        unseen_pixels = np.count_nonzero(mask_unseen)
        total_explorable_pixels = seen_pixels + unseen_pixels
        return (seen_pixels / total_explorable_pixels) * 100 if total_explorable_pixels > 0 else 0.0

    def _save_summary_files(self):
        """Saves the final summary files for time, distance, and exploration."""
        rospy.loginfo("Saving final summary files...")
        try:
            # 1. Save Total Time
            with self.timing_lock:
                if self.start_time:
                    total_time = (rospy.Time.now() - self.start_time).to_sec()
                    time_filepath = os.path.join(self.home_dir, "total_time_hab.txt")
                    with open(time_filepath, "w") as f:
                        f.write(f"{total_time:.2f}\n")
                    rospy.loginfo(f"Total time saved to {time_filepath}")

            # 2. Save Total Distance
            distance_filepath = os.path.join(self.home_dir, "total_distance.txt")
            with open(distance_filepath, "w") as f:
                f.write(f"{self.total_distance:.4f}\n")
            rospy.loginfo(f"Total distance saved to {distance_filepath}")

            # 3. Save Final Exploration Percentage
            final_percentage = self.get_exploration_percentage()
            if final_percentage is not None:
                percent_filepath = os.path.join(self.home_dir, "total_seen_percent.txt")
                with open(percent_filepath, "w") as f:
                    f.write(f"{final_percentage:.4f}\n")
                rospy.loginfo(f"Final exploration percentage saved to {percent_filepath}")

        except Exception as e:
            rospy.logerr(f"Failed to save summary files: {e}")

    def on_exit_generate_video(self):
        rospy.loginfo("Executing shutdown sequence...")
        self.save_final_maps()
        self.generate_plot()
        self._save_summary_files() # Dump summary files
        if self.make_video:
            generate_video(
                video_option=self.config.VIDEO_OPTION,
                video_dir=self.config.VIDEO_DIR,
                images=self.observations_per_episode,
                episode_id="roam_episode",
                scene_id=self.scene_id_last if self.scene_id_last else "unknown",
                agent_seed=0,
                checkpoint_idx=0,
                metrics={},
                tb_writer=None,
            )
        rospy.loginfo("Shutdown sequence complete.")

    def save_final_maps(self):
        rospy.loginfo("Saving final exploration map images...")
        try:
            if self.observations_per_episode:
                final_frame = self.observations_per_episode[-1]
                filename = f"final_rendered_map_{int(time.time())}.png"
                file_path = os.path.join(self.home_dir, filename)
                cv2.imwrite(file_path, cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR))
                rospy.loginfo(f"Final rendered map saved to {file_path}")

            top_down_map_measure = self.env._env._task.measurements.measures.get("top_down_map_for_roam")
            if top_down_map_measure and hasattr(top_down_map_measure, 'get_clean_map'):
                clean_map_data = top_down_map_measure.get_clean_map()
                if clean_map_data is not None:
                    clean_map_image = create_clean_map_image(clean_map_data)
                    filename = f"final_clean_map_{int(time.time())}.png"
                    file_path = os.path.join(self.home_dir, filename)
                    cv2.imwrite(file_path, cv2.cvtColor(clean_map_image, cv2.COLOR_RGB2BGR))
                    rospy.loginfo(f"Final clean map saved to {file_path}")
        except Exception as e:
            rospy.logerr(f"Error saving final map images: {e}")
            
    def generate_plot(self):
        rospy.loginfo("Generating exploration vs. distance plot...")
        try:
            if not os.path.exists(self.data_log_filepath):
                rospy.logwarn(f"Log file not found at {self.data_log_filepath}, skipping plot generation.")
                return
                
            distances, percentages = [], []
            with open(self.data_log_filepath, 'r') as f:
                next(f)
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
            plt.title('Exploration Progress')
            plt.xlabel('Distance Traveled (m)')
            plt.ylabel('Exploration Percentage (%)')
            plt.grid(True)
            plt.ylim(0, 100)
            plt.xlim(left=0)
            
            plot_filename = f"exploration_vs_distance_{int(time.time())}.png"
            plot_filepath = os.path.join(self.home_dir, plot_filename)
            plt.savefig(plot_filepath)
            rospy.loginfo(f"Plot saved to {plot_filepath}")
            plt.close()

        except Exception as e:
            rospy.logerr(f"Failed to generate plot: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-name", type=str, default="env_node")
    parser.add_argument("--task-config", type=str, default="configs/pointnav_d_orignal.yaml")
    parser.add_argument("--enable-physics-sim", default=False, action="store_true")
    parser.add_argument("--use-continuous-agent", default=False, action="store_true")
    parser.add_argument("--sensor-pub-rate", type=float, default=20.0)
    args = parser.parse_args()

    env_node = HabitatEnvNode(
        node_name=args.node_name,
        config_paths=args.task_config,
        enable_physics_sim=args.enable_physics_sim,
        use_continuous_agent=args.use_continuous_agent,
        pub_rate=args.sensor_pub_rate,
    )
    env_node.simulate()


if __name__ == "__main__":
    main()