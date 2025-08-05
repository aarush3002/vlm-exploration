import os
import math
import time
from pathlib import Path
from threading import Event, Lock

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from PIL import Image as PILImage

from tf.transformations import euler_from_quaternion
import scipy

bridge = CvBridge()

def euclidean_dist(x1, y1, x2, y2):
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5
# --------------------------------------------------------------------------- #
# helper: small wait-for-message with timeout                                 #
# --------------------------------------------------------------------------- #
def wait_for(topic, msg_type, timeout=5.0):
    try:
        return rospy.wait_for_message(topic, msg_type, timeout)
    except rospy.ROSException:
        rospy.logwarn(f"Timeout while waiting for {topic}")
        return None


# --------------------------------------------------------------------------- #
# class Robot – minimal continuous agent                                      #
# --------------------------------------------------------------------------- #
class Robot:
    def __init__(self, picture_interval=40, degrees_per_rotate=10,
                 lin_speed=2.4, ang_speed_deg=60.0):

        self.cmd_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        rospy.Subscriber("odom", Odometry, self._odom_cb)

        rospy.Subscriber("rgb",   Image, self._rgb_cb, queue_size=1)
        rospy.Subscriber("depth", Image, self._depth_cb, queue_size=1)

        self.odom_lock = Lock()
        self.pose = None                      # (x, y, yaw_deg)

        self.rgb_event   = Event()
        self.depth_event = Event()
        self.last_rgb    = None
        self.last_depth  = None

        self.ang_speed            = math.radians(ang_speed_deg)
        self.lin_speed            = lin_speed

        self.graph_origin = None

        self.map_lock = Lock()
        self.map_msg  = None        # will hold the latest OccupancyGrid
        self.trajectory = []

        # --- NEW: Trajectory Downsampling Parameters ---
        self.last_trajectory_pose = None
        self.last_trajectory_time = rospy.Time(0)
        self.min_dist_between_points = 0.1 # meters
        self.min_time_between_points = rospy.Duration(0.5) # seconds
        # --- End of New Parameters ---
        # rospy.Subscriber("/map", OccupancyGrid, self._map_cb, queue_size=1)
        rospy.Subscriber("/grid_map", OccupancyGrid, self._map_cb, queue_size=1)

        self.costmap_lock = Lock()
        self.costmap_msg = None
        self.costmap_arr = None
        rospy.Subscriber(
            "/move_base/global_costmap/costmap",
            OccupancyGrid,
            self._costmap_cb,
            queue_size=1
        )

        self.front_range   = float('inf')        # latest min range [m]

        rospy.Subscriber("/scan", LaserScan, self._laser_cb, queue_size=1)

        Path("captures").mkdir(exist_ok=True)

    # --- callbacks --------------------------------------------------------- #

    def _odom_cb(self, msg):
        with self.odom_lock:
            q = msg.pose.pose.orientation            # geometry_msgs/Quaternion


            # ---- quaternion -> yaw (REP‑103, Z‑axis) --------------------------
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            yaw_rad   = math.atan2(siny_cosp, cosy_cosp)
            yaw_deg   = (math.degrees(yaw_rad) + 360) % 360   # 0 … 360

            # ---- cache pose ---------------------------------------------------
            self.pose = (
                {"x": msg.pose.pose.position.x,
                "y": msg.pose.pose.position.y},
                yaw_deg
            )

            # Check if enough time or distance has passed to record a new point
        
            # First point is always recorded
            if self.last_trajectory_pose is None:
                should_record = True
            else:
                # Check time
                time_since_last = msg.header.stamp - self.last_trajectory_time
                # Check distance
                dist_since_last = euclidean_dist(
                    self.pose[0]['x'], self.pose[0]['y'],
                    self.last_trajectory_pose[0]['x'], self.last_trajectory_pose[0]['y']
                )
                
                should_record = (dist_since_last > self.min_dist_between_points or
                                time_since_last > self.min_time_between_points)

            if should_record:
                self.trajectory.append(self.pose)
                self.last_trajectory_pose = self.pose
                self.last_trajectory_time = msg.header.stamp

    def _map_cb(self, msg: OccupancyGrid):
        """Cache the most-recent /map message and its useful meta-data."""
        with self.map_lock:
            self.map_msg   = msg
            self.map_arr   = np.array(msg.data, dtype=np.int8).reshape(
                                (msg.info.height, msg.info.width))
            self.map_res   = msg.info.resolution

            # print("Grid map origin", msg.info.origin.position.x, msg.info.origin.position.y)
            self.map_origin= (msg.info.origin.position.x,
                            msg.info.origin.position.y)

    def _rgb_cb(self, msg):
        self.last_rgb = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.rgb_event.set()

    def _depth_cb(self, msg):
        # depth comes in metres, encoding passthrough
        self.last_depth = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.depth_event.set()

    def _laser_cb(self, msg: LaserScan):
        """Cache the minimum range in ±1 ° directly ahead."""
        n = len(msg.ranges)
        # beams are counter‑clockwise; 0 = straight ahead
        span = int(math.radians(5) / msg.angle_increment)
        centre = 0                                   # front beam
        i0 = (centre - span) % n
        i1 = (centre + span) % n
        if i0 < i1:
            rng = [r for r in msg.ranges[i0:i1+1] if math.isfinite(r)]
        else:                                         # wrap‑around
            rng = [r for r in (msg.ranges[i0:] + msg.ranges[:i1+1])
                if math.isfinite(r)]
        self.front_range = min(rng) if rng else float('inf')

    def _costmap_cb(self, msg: OccupancyGrid):
        """Cache the most-recent global costmap message."""
        with self.costmap_lock:
            self.costmap_msg = msg
            self.costmap_arr = np.array(msg.data, dtype=np.int8).reshape(
                                (msg.info.height, msg.info.width))

    # --- motion primitives -------------------------------------------------- #
    def _publish_twist(self, vx=0.0, wz=0.0, duration=0.0):
        tw = Twist()
        tw.linear.x  = vx
        tw.angular.z = wz
        end = rospy.Time.now() + rospy.Duration.from_sec(duration)
        rate = rospy.Rate(20)
        while rospy.Time.now() < end and not rospy.is_shutdown():
            self.cmd_pub.publish(tw)
            rate.sleep()
        self.cmd_pub.publish(Twist())     # stop

    def rotate_deg(self, angle_deg, yaw_tol_deg=0.5):
        """
        Rotate the robot by `angle_deg` degrees (CCW = positive) using the yaw
        reported on /odom.  The routine always takes the shortest path and never
        blocks on new odom messages.
        """
        # ------------------------------------------------ wait for first pose
        while self.pose is None and not rospy.is_shutdown():
            rospy.sleep(0.02)

        with self.odom_lock:
            start_yaw = self.pose[1]            # degrees in (‑180, +180]
        
        #print("Start yaw", self.pose[1])

        goal_yaw = start_yaw + angle_deg        # keep it in the same domain

        #print("Goal yaw", goal_yaw)

        # --------- helpers -----------------------------------------------------
        def ang_err(target, current):
            """
            Smallest signed difference target‑current in degrees, result ∈ (‑180,180].
            """
            diff = target - current
            while diff > 180.0:
                diff -= 360.0
            while diff <= -180.0:
                diff += 360.0
            return diff

        tw   = Twist()
        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            with self.odom_lock:
                cur_yaw = self.pose[1]

            #print(self.pose)

            #print("current yaw", cur_yaw)

            err = ang_err(goal_yaw, cur_yaw)

            #print("angle error", err)

            if abs(err) <= yaw_tol_deg:
                break

            # direction & speed
            tw.angular.z = math.copysign(self.ang_speed, err)

            # slow down for the last 10 °
            if abs(err) < 10.0:
                tw.angular.z *= 0.4

            self.cmd_pub.publish(tw)
            rate.sleep()

        # stop
        self.cmd_pub.publish(Twist())

    def set_ang_vel(self, angle_deg, rate_hz=30):
        """
        Rotate the robot by `angle_deg` degrees **open‑loop**:

        * No odometry feedback, no safety checks, no slowdown.
        * Positive `angle_deg`  →  counter‑clockwise (REP‑103)
          Negative `angle_deg`  →  clockwise.
        """
        # constant angular speed already stored in self.ang_speed  (rad/s)
        duration = abs(math.radians(angle_deg)) / self.ang_speed
        twist    = Twist()
        twist.angular.z = math.copysign(self.ang_speed, angle_deg)

        end  = rospy.Time.now() + rospy.Duration.from_sec(duration)
        rate = rospy.Rate(rate_hz)

        while rospy.Time.now() < end and not rospy.is_shutdown():
            self.cmd_pub.publish(twist)
            print("cur yaw", self.pose[1])
            rate.sleep()

        self.cmd_pub.publish(Twist())          # tidy stop

    def drive_forward(self,
                  distance_m,
                  pos_tol=0.1,
                  slow_down_ratio=0.0, #0.25
                  safety_stop=0.15,        # < this → full stop
                  safety_slow=0.0,        # < this → crawl, 0.5
                  crawl_ratio=0.20):       # % of lin_speed when crawling
        """
        Drive straight ahead for `distance_m` m with odometry feedback.

        The robot also monitors /scan (or depth) and
        * **stops** if an obstacle is closer than `safety_stop`;
        * **crawls** if it is between `safety_stop` and `safety_slow`.

        Parameters
        ----------
        distance_m : float
            Desired travel distance.
        safety_stop : float
            If the closest obstacle in front is < ` safety_stop` [m] we abort.
        safety_slow : float
            Below this distance we reduce speed to `crawl_ratio × lin_speed`.
        """
        # ------------- wait for odom & remember starting pose ----------
        while self.pose is None and not rospy.is_shutdown():
            rospy.sleep(0.01)

        with self.odom_lock:
            start_x = self.pose[0]["x"]
            start_y = self.pose[0]["y"]

        tw   = Twist()
        rate = rospy.Rate(30)                              # control loop

        while not rospy.is_shutdown():
            # ---- distance progress -----------------------------------
            with self.odom_lock:
                cur_x = self.pose[0]["x"]
                cur_y = self.pose[0]["y"]
            travelled = math.hypot(cur_x - start_x, cur_y - start_y)
            remaining = distance_m - travelled

            if remaining <= pos_tol:
                break                                      # reached goal

            # ---- safety check (front laser) --------------------------
            d = self.front_range                           # cached by _laser_cb
            print(self.front_range)
            if d < safety_stop:
                rospy.logwarn("Obstacle %.2f m ahead – aborting drive_forward",
                            d)
                break                                      # emergency stop

            # ---- speed profile --------------------------------------
            if d < safety_slow:
                # close to obstacle ⇒ crawl
                speed = self.lin_speed * crawl_ratio
            elif remaining < slow_down_ratio * distance_m:
                speed = self.lin_speed * 0.4               # final approach
            else:
                speed = self.lin_speed                    # cruise

            tw.linear.x = speed
            self.cmd_pub.publish(tw)
            rate.sleep()

        # ------------- stop cleanly -----------------------------------
        self.cmd_pub.publish(Twist())
    
    def publish_empty_twist(self):
        stop_msg = Twist()
        self.cmd_pub.publish(stop_msg)
        rospy.loginfo("Published stop command to /cmd_vel.")

    def publish_unstuck_twist(self):
        unstuck_msg = Twist()
        unstuck_msg.linear.x = 5
        self.cmd_pub.publish(unstuck_msg)
        rospy.sleep(0.5)
        self.publish_empty_twist()

    # --- perception --------------------------------------------------------- #
    def capture_rgbd(self):
        """return (rgb, depth) blocking until both images arrive"""
        self.rgb_event.clear()
        self.depth_event.clear()
        if not self.rgb_event.wait(1.0) or not self.depth_event.wait(1.0):
            raise RuntimeError("Camera images not arriving")
        return self.last_rgb.copy(), self.last_depth.copy()
    

    def find_frontiers(self, occupancy_grid, min_length_pixels=15):
        """
        Finds frontiers in an occupancy grid.

        :param occupancy_grid: A 2D numpy array where -1 is unknown, 0 is free.
        :param min_length_pixels: The minimum number of pixels for a contour to be considered a frontier.
        :return: A list of frontier contours.
        """
        # Create a binary image where potential frontier cells are white
        # A cell is a potential frontier if it is free (0)
        is_free = occupancy_grid == 0

        # A cell has an unknown neighbor if any of its neighbors are unknown (-1)
        is_unknown = occupancy_grid == -1
        from scipy.ndimage import binary_dilation
        has_unknown_neighbor = binary_dilation(is_unknown, structure=np.ones((3,3)))

        # A frontier cell is a free cell that has an unknown neighbor
        frontier_mask = np.where(is_free & has_unknown_neighbor, 255, 0).astype(np.uint8)

        # Find, filter, and return the contours
        contours, _ = cv2.findContours(frontier_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Filter contours by length
        frontiers = [c for c in contours if len(c) > min_length_pixels]

        return frontiers
    
    def p2w(self, px, py):
        """Convert pixel coordinates to world coordinates."""
        if self.map_msg is None:
            return None, None

        res = self.map_res
        ox, oy = self.map_origin
        h, w = self.map_arr.shape

        # Un-mirror the X coordinate
        #px = w - 1 - px

        # Convert to world coordinates
        wx = ox + px * res
        wy = oy + py * res

        return wx, wy

    def overlay_graph_on_occupancy_map(
        self,
        graph,
        robot_position,
        robot_yaw_deg,
        frontiers,  # Pass in the list of frontiers
        out_folder="map_graphs",
        out_filename="map_with_graph.png",
        robot_color=(0, 0, 255),         # Red
        arrow_color=(0, 165, 255),      # Orange
        vertex_color=(0, 255, 0),        # Green
        trajectory_color=(200, 0, 0),   # Bright Blue for actual path
        graph_edge_color=(100, 100, 100), # Muted Gray for graph edges
        frontier_outline_color=(0, 255, 255), # Yellow
        frontier_label_color=(255, 0, 255), # Magenta
        vertex_radius=3,
        robot_radius=3,
        edge_thickness=1,
        arrow_len_m=0.5,
        crop_margin_px=10):
        """
        Draws trajectory, graph, robot pose, costmap, and numerically labeled frontiers.
        """
        # --- Create Base Map Image ---
        with self.map_lock:
            if self.map_msg is None: raise RuntimeError("No /map message received yet.")
            grid, res, (ox, oy), (h, w) = self.map_arr.copy(), self.map_res, self.map_origin, self.map_arr.shape

        gray = np.where(grid == -1, 127, np.where(grid > 50, 0, 255)).astype(np.uint8)
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # --- Find and Draw Frontier Outlines ---
        all_frontiers = self.find_frontiers(grid, min_length_pixels=20)
        color = cv2.flip(color, 1)  # Mirror the entire canvas once
        for frontier in all_frontiers:
            frontier[:, :, 0] = w - 1 - frontier[:, :, 0] # Flip the contour points to match
        cv2.drawContours(color, all_frontiers, -1, frontier_outline_color, 1)

        # --- Helper function for ALL coordinate transformations ---
        def world_to_flipped_pixel(wx, wy):
            # 1. Convert world coordinate to original pixel coordinate
            col = int(round((wx - ox) / res))
            row = int(round((wy - oy) / res))
            # 2. Apply the horizontal flip to match the canvas
            col = w - 1 - col
            return (col, row)

        # --- Label the Filtered Frontiers ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        if frontiers:
            for idx, (dist, midpoint, contour) in enumerate(frontiers):
                px, py = world_to_flipped_pixel(midpoint[0], midpoint[1])
                cv2.putText(color, str(idx), (px-2, py+2), font, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(color, str(idx), (px-2, py+2), font, 0.4, frontier_label_color, 1, cv2.LINE_AA)

        # --- Draw Actual Robot Trajectory ---
        if len(self.trajectory) > 1:
            pixel_points = [world_to_flipped_pixel(pos['x'], pos['y']) for pos, yaw in self.trajectory]
            pts = np.array(pixel_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(color, [pts], isClosed=False, color=trajectory_color, thickness=2)

        # --- Draw Graph Vertices (Unlabeled) ---
        for v_node in graph.adj_list:
            vx, vy = world_to_flipped_pixel(v_node.x, v_node.y)
            cv2.circle(color, (vx, vy), vertex_radius, vertex_color, -1)
        
        # --- Draw Robot Pose ---
        if not isinstance(robot_position, dict):
            robot_position = {"x": robot_position[0], "y": robot_position[1]}
        rpx, rpy = world_to_flipped_pixel(robot_position["x"], robot_position["y"])
        cv2.circle(color, (rpx, rpy), robot_radius, robot_color, -1)
        yaw_rad = math.radians(robot_yaw_deg)
        arrow_px = arrow_len_m / res
        # The arrow direction also needs to be flipped on the x-axis
        dx_px = -arrow_px * math.cos(yaw_rad)
        dy_px = arrow_px * math.sin(yaw_rad)
        cv2.arrowedLine(color, (rpx, rpy), (int(rpx + dx_px), int(rpy + dy_px)),
                        arrow_color, 2, tipLength=0.3)
        
        # Crop, add footer/scale bar, and save
        mask = np.any(color != 127, axis=2)
        ys, xs = np.where(mask)
        if xs.size and ys.size:
            m = crop_margin_px
            y0, y1 = max(int(ys.min()) - m, 0), min(int(ys.max()) + m, color.shape[0] - 1)
            x0, x1 = max(int(xs.min()) - m, 0), min(int(xs.max()) + m, color.shape[1] - 1)
            color = color[y0:y1 + 1, x0:x1 + 1]

        footer_h = 20
        color = cv2.copyMakeBorder(color, 0, footer_h, 0, 0, cv2.BORDER_CONSTANT, value=(127, 127, 127))

        inc_px = max(int(0.5 / res), 1)
        left_margin, suffix_margin = 10, 12
        avail_px = color.shape[1] - left_margin - suffix_margin
        num_inc = max(2, (avail_px // inc_px))
        num_inc -= num_inc % 1
        bar_len_px = num_inc * inc_px
        x0, y0 = left_margin, color.shape[0] - footer_h // 2
        x1 = x0 + bar_len_px
        cv2.line(color, (x0, y0), (x1, y0), (0, 0, 255), 2)
        for i in range(num_inc + 1):
            xi = x0 + i * inc_px
            cv2.line(color, (xi, y0 - 4), (xi, y0 + 4), (0, 0, 255), 2)
            if i % 2 == 0:
                label = f"{i*0.5:g}"
                (tw, _), _ = cv2.getTextSize(label, font, 0.4, 1)
                cv2.putText(color, label, (xi - tw // 2, y0 - 6), font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(color, "m", (x1 + 4, y0 + 2), font, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        Path(out_folder).mkdir(exist_ok=True)
        out_path = os.path.join(out_folder, out_filename)
        PILImage.fromarray(color[..., ::-1]).save(out_path)
        rospy.loginfo(f"Graph+robot overlay saved to {out_path}")
        return out_path

    def save_raw_map(self, out_folder="map_graphs", out_filename="final_occupancy_map.png"):
        """Saves the current raw occupancy grid to a PNG file."""
        with self.map_lock:
            if self.map_msg is None: raise RuntimeError("No /map message received yet.")
            grid, res, (ox, oy), (h, w) = self.map_arr.copy(), self.map_res, self.map_origin, self.map_arr.shape

        gray = np.where(grid == -1, 127, np.where(grid > 50, 0, 255)).astype(np.uint8)
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        color = cv2.flip(color, 1)

        Path(out_folder).mkdir(exist_ok=True)
        out_path = os.path.join(out_folder, out_filename)
        PILImage.fromarray(color[..., ::-1]).save(out_path)




