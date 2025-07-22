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

bridge = CvBridge()


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
                 lin_speed=0.8, ang_speed_deg=60.0):

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
        # rospy.Subscriber("/map", OccupancyGrid, self._map_cb, queue_size=1)
        rospy.Subscriber("/map", OccupancyGrid, self._map_cb, queue_size=1)

        self.front_range   = float('inf')        # latest min range [m]

        rospy.Subscriber("/scan", LaserScan, self._laser_cb, queue_size=1)

        Path("captures").mkdir(exist_ok=True)

    # --- callbacks --------------------------------------------------------- #
    # def _odom_cb(self, msg):
    #     with self.odom_lock:
    #         print(msg.pose.pose.orientation)
    #         q = msg.pose.pose.orientation
            
    #         x = q.x
    #         y = q.z          # swap
    #         z = -q.y         # and flip sign
    #         w = q.w
    #         # quaternion → yaw
    #         siny_cosp = 2 * (w * z + x * y)
    #         cosy_cosp = 1 - 2 * (y * y + z * z)
    #         yaw_deg   = math.degrees(math.atan2(siny_cosp, cosy_cosp))

    #         yaw_deg = (-yaw_deg) % 360.0

    #         self.pose = ({"x": msg.pose.pose.position.x, "y": msg.pose.pose.position.y}, yaw_deg)
    #         # self.pose = (msg.pose.pose.position.x,
    #         #              msg.pose.pose.position.y,
    #         #              yaw)

    def _odom_cb(self, msg):
        with self.odom_lock:
            q = msg.pose.pose.orientation            # geometry_msgs/Quaternion

            # # roll, pitch, yaw in radians
            # roll_r, pitch_r, yaw_r = euler_from_quaternion(
            #     [q.x, q.y, q.z, q.w], axes="sxyz")

            # # convert to degrees for easy reading
            # roll_deg  = math.degrees(roll_r)
            # pitch_deg = math.degrees(pitch_r)
            # yaw_deg   = (math.degrees(yaw_r) + 360) % 360   # 0 … 360

            # print(f"RPY(deg)  roll:{roll_deg:7.2f}  pitch:{pitch_deg:7.2f}  yaw:{yaw_deg:7.2f}")

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
        span = int(math.radians(45) / msg.angle_increment)
        centre = 0                                   # front beam
        i0 = (centre - span) % n
        i1 = (centre + span) % n
        if i0 < i1:
            rng = [r for r in msg.ranges[i0:i1+1] if math.isfinite(r)]
        else:                                         # wrap‑around
            rng = [r for r in (msg.ranges[i0:] + msg.ranges[:i1+1])
                if math.isfinite(r)]
        self.front_range = min(rng) if rng else float('inf')

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

    # def rotate_deg(self, angle_deg, angle_tol = 1.0):
    #     with self.odom_lock:
    #         if self.pose is None:
    #             rospy.logwarn("rotate_deg(): odometry not ready yet")
    #             return
    #         start_yaw = self.pose[1]                     # degrees

    #     # desired absolute yaw in [0,360)
    #     target_yaw = (start_yaw + angle_deg) % 360.0
    #     sign       =  1.0 if angle_deg > 0 else -1.0     # CCW (+) or CW (−)

    #     rate = rospy.Rate(50)       # 50 Hz command stream
    #     tw   = Twist()
    #     tw.linear.x  = 0.0
    #     tw.angular.z = sign * self.ang_speed             # rad/s (y-axis)

    #     while not rospy.is_shutdown():
    #         # --- publish max-rate angular twist -------------------------------- #
    #         self.cmd_pub.publish(tw)

    #         # --- current yaw & remaining error --------------------------------- #
    #         with self.odom_lock:
    #             cur_yaw = self.pose[1]                   # degrees

    #         # shortest signed angle from cur → target
    #         err = ((target_yaw - cur_yaw + 540) % 360) - 180

    #         if abs(err) <= angle_tol:
    #             break                                    # done

    #         # optional: slow down for the last 10°
    #         if abs(err) < 10.0:
    #             tw.angular.z = sign * self.ang_speed * 0.4

    #         rate.sleep()

    #     # --- stop -------------------------------------------------------------- #
    #     self.cmd_pub.publish(Twist())

    # def rotate_deg(self, angle_deg, yaw_tol_deg=1.0):
    #     """
    #     Rotate the robot by `angle_deg` (positive = CCW) using the yaw coming
    #     from the /odom subscriber.  No blocking waits on /odom messages.
    #     """
    #     # --- wait until we have at least one odom reading ---------------------
    #     while self.pose is None and not rospy.is_shutdown():
    #         rospy.sleep(0.02)          # 20 ms tick

    #     with self.odom_lock:
    #         start_yaw = self.pose[1]   # degrees

    #     goal_yaw = (start_yaw + angle_deg) % 360.0
    #     sign     =  1.0 if angle_deg > 0 else -1.0

    #     tw = Twist()
    #     tw.angular.z = sign * self.ang_speed         # rad/s
    #     rate = rospy.Rate(30)                        # control loop

    #     while not rospy.is_shutdown():
    #         # publish constant twist
    #         self.cmd_pub.publish(tw)

    #         # current yaw (cached, no waiting)
    #         with self.odom_lock:
    #             cur_yaw = self.pose[1]               # degrees

    #         #print(cur_yaw)

    #         # shortest signed angle error
    #         err = ((goal_yaw - cur_yaw + 540) % 360) - 180

    #         if abs(err) < yaw_tol_deg:
    #             break                                # done

    #         # optional final slowdown
    #         if abs(err) < 10.0:
    #             tw.angular.z = sign * self.ang_speed * 0.4

    #         rate.sleep()

    #     self.cmd_pub.publish(Twist())                # stop

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

    # def drive_forward(self,
    #               distance_m,
    #               pos_tol=0.2,           # acceptable residual (m)
    #               slow_down_ratio=0.25):  # slow for the last 25 %
    #     """
    #     Drive straight ahead for `distance_m` metres using odometry feedback.

    #     It continuously monitors /odom instead of relying on a timed burst,
    #     so it is robust to simulation slow‑downs or small modelling errors.

    #     Parameters
    #     ----------
    #     distance_m : float
    #         How far to travel, in metres.
    #     pos_tol : float, optional
    #         We stop when the remaining distance is ≤ `pos_tol`.
    #     slow_down_ratio : float, optional
    #         When the remaining distance is below this fraction of the goal,
    #         linear speed is reduced to improve accuracy.
    #     """
    #     # ------------------------------------------------ wait for odom
    #     while self.pose is None and not rospy.is_shutdown():
    #         rospy.sleep(0.01)

    #     with self.odom_lock:
    #         start_x = self.pose[0]["x"]
    #         start_y = self.pose[0]["y"]

    #     # ------------------------------------------------ command packet
    #     tw          = Twist()
    #     tw.angular.z = 0.0
    #     rate        = rospy.Rate(30)           # control loop

    #     while not rospy.is_shutdown():
    #         # ---- current travelled distance ----------------------------------
    #         with self.odom_lock:
    #             cur_x = self.pose[0]["x"]
    #             cur_y = self.pose[0]["y"]
    #         travelled = math.hypot(cur_x - start_x, cur_y - start_y)
    #         remaining = distance_m - travelled

    #         if remaining <= pos_tol:
    #             break        # goal reached

    #         # ---- speed profile ----------------------------------------------
    #         if remaining < slow_down_ratio * distance_m:
    #             tw.linear.x = self.lin_speed * 0.4   # final approach
    #         else:
    #             tw.linear.x = self.lin_speed         # cruise

    #         self.cmd_pub.publish(tw)
    #         rate.sleep()

    #     # ------------------------------------------------ stop cleanly
    #     self.cmd_pub.publish(Twist())

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
                  pos_tol=0.2,
                  slow_down_ratio=0.25,
                  safety_stop=0.2,        # < this → full stop
                  safety_slow=0.5,        # < this → crawl
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

    # --- perception --------------------------------------------------------- #
    def capture_rgbd(self):
        """return (rgb, depth) blocking until both images arrive"""
        self.rgb_event.clear()
        self.depth_event.clear()
        if not self.rgb_event.wait(1.0) or not self.depth_event.wait(1.0):
            raise RuntimeError("Camera images not arriving")
        return self.last_rgb.copy(), self.last_depth.copy()
    
    # def overlay_graph_on_occupancy_map(self, graph,
    #                                robot_position, robot_yaw_deg,
    #                                out_folder="map_graphs",
    #                                out_filename="map_with_graph.png",
    #                                robot_color=(0,0,255),
    #                                vertex_color=(0,255,0),
    #                                edge_color=(255,0,0),
    #                                vertex_radius=2, robot_radius=2,
    #                                edge_thickness=1, arrow_len_m=0.5,
    #                                crop_margin_px=10):
    #     """
    #     Draw vertices (green), edges (blue) and the robot pose (red) on the most
    #     recent /map and save a PNG.  Requires that _map_cb has already received
    #     at least one OccupancyGrid.
    #     """
    #     with self.map_lock:
    #         if self.map_msg is None:
    #             raise RuntimeError("No /map message received yet.")
    #         grid      = self.map_arr.copy()            # 2‑D int8
    #         res       = self.map_res
    #         ox, oy    = self.map_origin
    #         h, w      = grid.shape

    #     # --- occupancy -> BGR image -------------------------------------------
    #     gray  = np.where(grid == -1, 127,
    #             np.where(grid > 50, 0, 255)).astype(np.uint8)
    #     color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    #     # ----------------------------------------------------------------------
    #     def w2p(wx, wy):
    #         """
    #         World metres -> (col,row) in the occupancy grid.
    #         col grows East (+x), row grows South (‑y, top‑left origin).
    #         """
    #         col = int((wx - ox) / res)
    #         row = h - 1 - int((wy - oy) / res)
    #         return col, row

    #     # --- robot ------------------------------------------------------------
    #     rpx = w2p(*robot_position)
    #     cv2.circle(color, rpx, robot_radius, robot_color, -1)
    #     yaw = math.radians(robot_yaw_deg)
    #     dx  =  int((arrow_len_m/res) * math.cos(yaw))
    #     dy  = -int((arrow_len_m/res) * math.sin(yaw))   # screen Y is down
    #     cv2.arrowedLine(color, rpx, (rpx[0]+dx, rpx[1]+dy),
    #                     robot_color, 2, tipLength=0.3)

    #     # --- edges ------------------------------------------------------------
    #     seen = set()
    #     for v in graph.adj_list:
    #         vpx = w2p(v.x, v.y)
    #         for nb, _ in graph.adj_list[v]:
    #             key = tuple(sorted([(v.x, v.y), (nb.x, nb.y)]))
    #             if key in seen:
    #                 continue
    #             seen.add(key)
    #             cv2.line(color, vpx, w2p(nb.x, nb.y), edge_color, edge_thickness)

    #     # --- vertices & labels ------------------------------------------------
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     for idx in sorted(graph.vertex_labels):
    #         v  = graph.vertex_labels[idx]
    #         vx, vy = w2p(v.x, v.y)
    #         cv2.circle(color, (vx, vy), vertex_radius, vertex_color, -1)
    #         cv2.putText(color, str(idx), (vx+3, vy-3),
    #                     font, 0.3, (0,0,0), 4, cv2.LINE_AA)
    #         cv2.putText(color, str(idx), (vx+3, vy-3),
    #                     font, 0.3, (0,255,255), 1, cv2.LINE_AA)

    #     # --- auto‑crop around explored area -----------------------------------
    #     mask = np.any(color != 127, axis=2)
    #     ys, xs = np.where(mask)
    #     if xs.size and ys.size:
    #         m   = crop_margin_px
    #         y0  = max(int(ys.min()) - m, 0)
    #         y1  = min(int(ys.max()) + m, color.shape[0] - 1)
    #         x0  = max(int(xs.min()) - m, 0)
    #         x1  = min(int(xs.max()) + m, color.shape[1] - 1)
    #         color = color[y0:y1+1, x0:x1+1]

    #     # --- save -------------------------------------------------------------
    #     Path(out_folder).mkdir(exist_ok=True)
    #     out_path = os.path.join(out_folder, out_filename)
    #     cv2.imwrite(out_path, color)
    #     rospy.loginfo(f"Graph+robot overlay saved to {out_path}")
    #     return out_path

    # def overlay_graph_on_occupancy_map(
    #     self,
    #     graph,
    #     robot_position, robot_yaw_deg,
    #     out_folder="map_graphs",
    #     out_filename="map_with_graph.png",
    #     robot_color   =(0, 0, 255),      # red dot
    #     arrow_color   =(0,165,255),      # orange heading
    #     vertex_color  =(0,255,0),
    #     edge_color    =(255,0,0),
    #     vertex_radius =2,
    #     robot_radius  =2,
    #     edge_thickness=1,
    #     arrow_len_m   =0.5,
    #     crop_margin_px=10):
    #     """
    #     Draw vertices, edges and robot pose on the latest /map.

    #     **NOTE**  
    #     `self.graph_origin` must already hold
    #         ({"x": x₀,"y": y₀}, yaw₀_deg) from the very first /odom.
    #     """
    #     if not hasattr(self, "graph_origin") or self.graph_origin is None:
    #         raise RuntimeError("self.graph_origin not initialised!")

    #     pos0, yaw0_deg = self.graph_origin
    #     if not isinstance(robot_position, dict):
    #         robot_position = {"x": robot_position[0], "y": robot_position[1]}

    #     # ---------------------------- map data
    #     with self.map_lock:
    #         if self.map_msg is None:
    #             raise RuntimeError("No /map message received yet.")
    #         grid   = self.map_arr.copy()
    #         res    = self.map_res
    #         ox, oy = self.map_origin
    #         h, w   = grid.shape

    #     gray  = np.where(grid == -1, 127,
    #             np.where(grid > 50, 0, 255)).astype(np.uint8)
    #     color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    #     # ---------------------------- helpers
    #     c0, s0 = math.cos(math.radians(yaw0_deg)), \
    #             math.sin(math.radians(yaw0_deg))

    #     def to_local(wx, wy):
    #         dx, dy = wx - pos0["x"], wy - pos0["y"]
    #         return  c0*dx + s0*dy,  -s0*dx + c0*dy

    #     def w2p(wx, wy):
    #         """world → pixel (col,row) with a *horizontal mirror*"""
    #         x_, y_ = to_local(wx, wy)
    #         col = w - 1 - int((x_ - ox) / res)      # ← HERE: mirror L↔R
    #         row = h - 1 - int((y_ - oy) / res)
    #         return col, row

    #     # ---------------------------- robot
    #     rpx = w2p(robot_position["x"], robot_position["y"])
    #     cv2.circle(color, rpx, robot_radius, robot_color, -1)

    #     yaw_local = math.radians(robot_yaw_deg - yaw0_deg)
    #     dx_px =  int((arrow_len_m / res) * math.cos(yaw_local))
    #     dy_px = -int((arrow_len_m / res) * math.sin(yaw_local))
    #     cv2.arrowedLine(color, rpx, (rpx[0] + dx_px, rpx[1] + dy_px),
    #                     arrow_color, 2, tipLength=0.3)

    #     # ---------------------------- edges
    #     seen = set()
    #     for v in graph.adj_list:
    #         vpx = w2p(v.x, v.y)
    #         for nb, _ in graph.adj_list[v]:
    #             key = tuple(sorted([(v.x, v.y), (nb.x, nb.y)]))
    #             if key in seen:
    #                 continue
    #             seen.add(key)
    #             cv2.line(color, vpx, w2p(nb.x, nb.y),
    #                     edge_color, edge_thickness)

    #     # ---------------------------- vertices + labels
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     for idx in sorted(graph.vertex_labels):
    #         v  = graph.vertex_labels[idx]
    #         vx, vy = w2p(v.x, v.y)
    #         cv2.circle(color, (vx, vy), vertex_radius, vertex_color, -1)
    #         cv2.putText(color, str(idx), (vx + 3, vy - 3),
    #                     font, 0.3, (0,0,0),   4, cv2.LINE_AA)
    #         cv2.putText(color, str(idx), (vx + 3, vy - 3),
    #                     font, 0.3, (0,255,255), 1, cv2.LINE_AA)

    #     # ---------------------------- tight crop
    #     mask = np.any(color != 127, axis=2)
    #     ys, xs = np.where(mask)
    #     if xs.size and ys.size:
    #         m  = crop_margin_px
    #         y0 = max(int(ys.min()) - m, 0)
    #         y1 = min(int(ys.max()) + m, color.shape[0] - 1)
    #         x0 = max(int(xs.min()) - m, 0)
    #         x1 = min(int(xs.max()) + m, color.shape[1] - 1)
    #         color = color[y0:y1 + 1, x0:x1 + 1]

    #     # ---------------------------- save
    #     Path(out_folder).mkdir(exist_ok=True)
    #     out_path = os.path.join(out_folder, out_filename)
    #     cv2.imwrite(out_path, cv2.flip(color, 1))
    #     rospy.loginfo(f"Graph+robot overlay saved to {out_path}")
    #     return out_path

    def overlay_graph_on_occupancy_map(
        self,
        graph,
        robot_position, robot_yaw_deg,
        out_folder="map_graphs",
        out_filename="map_with_graph.png",
        robot_color   =(  0,   0, 255),   # red
        arrow_color   =(  0, 165, 255),   # orange
        vertex_color  =(  0, 255,   0),   # green
        edge_color    =(255,   0,   0),   # blue
        vertex_radius =2,
        robot_radius  =2,
        edge_thickness=1,
        arrow_len_m   =0.5,
        crop_margin_px=10):
        """
        Draw vertices, edges and robot pose on the latest /map.

        • Map image is **mirrored horizontally** so it matches the joystick view.  
        • World → local transform does: (i) translate by the first /odom pose,
        **then** (ii) rotate by –yaw₀ (so everything is expressed as if the
        robot had started at (0, 0) facing +x).  
        • Labels remain readable after the mirror.
        """
        # ------------------------------------------------ sanity
        if not hasattr(self, "graph_origin") or self.graph_origin is None:
            raise RuntimeError("self.graph_origin not initialised!")
        pos0, yaw0_deg = self.graph_origin          # translation + yaw₀

        print(pos0, yaw0_deg)

        if not isinstance(robot_position, dict):
            robot_position = {"x": robot_position[0], "y": robot_position[1]}

        # ------------------------------------------------ map → mirrored canvas
        with self.map_lock:
            if self.map_msg is None:
                raise RuntimeError("No /map message received yet.")
            grid   = self.map_arr.copy()
            res    = self.map_res
            ox, oy = self.map_origin
            h, w   = grid.shape

            # yaw of the map (deg) from occupancy‐grid metadata
            q = self.map_msg.info.origin.orientation        # geometry_msgs/Quaternion
            siny = 2*(q.w*q.z + q.x*q.y)
            cosy = 1 - 2*(q.y*q.y + q.z*q.z)
            map_yaw_deg = math.degrees(math.atan2(siny, cosy))
            print("Map yaw deg", map_yaw_deg)

        gray   = np.where(grid == -1, 127,
                np.where(grid > 50, 0, 255)).astype(np.uint8)
        color  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        color  = cv2.flip(color, 1)                 # *** horizontal mirror ***

        # ------------------------------------------------ helpers
        theta = math.radians(-2 * (yaw0_deg + 3.5))          # +yaw₀  (CCW)
        c0, s0 = math.cos(theta), math.sin(theta)

        def to_local(wx, wy):
            """
            1. translate world point by –pos0
            2. rotate CCW by +yaw₀        (align graph with map)
            3. flip 180 °                 (same extra inversion as before)
            """
            dx, dy = wx - pos0["x"], wy - pos0["y"]

            # correct CCW rotation (+yaw₀)
            x_r =  c0*dx - s0*dy
            y_r =  s0*dx + c0*dy

            return -x_r, -y_r              # 180° flip

        def w2p(wx, wy):
            """World metres → pixel (col,row) on the mirrored image."""
            x_l, y_l = to_local(wx, wy)
            col_orig = int((x_l - ox) / res)         # before mirror
            row      = h - 1 - int((y_l - oy) / res)
            col      = w - 1 - col_orig              # mirror L↔R
            return col, row

        # ------------------------------------------------ robot marker + arrow
        rpx = w2p(robot_position["x"], robot_position["y"])
        cv2.circle(color, rpx, robot_radius, robot_color, -1)

        # arrow uses the same (+yaw₀) convention
        yaw_local = math.radians(robot_yaw_deg - yaw0_deg)      # already global
        dx_px = -(arrow_len_m / res) * math.cos(yaw_local)
        dy_px = (arrow_len_m / res) * math.sin(yaw_local)
        cv2.arrowedLine(color, rpx,
                        (int(rpx[0] + dx_px), int(rpx[1] + dy_px)),
                        arrow_color, 2, tipLength=0.3)

        # ------------------------------------------------ edges
        seen=set()
        for v in graph.adj_list:
            vpx = w2p(v.x, v.y)
            for nb,_ in graph.adj_list[v]:
                key = tuple(sorted([(v.x,v.y),(nb.x,nb.y)]))
                if key in seen: continue
                seen.add(key)
                cv2.line(color, vpx, w2p(nb.x, nb.y),
                        edge_color, edge_thickness)

        # ------------------------------------------------ vertices + labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        for idx in sorted(graph.vertex_labels):
            v  = graph.vertex_labels[idx]
            vx, vy = w2p(v.x, v.y)
            cv2.circle(color, (vx,vy), vertex_radius, vertex_color, -1)
            cv2.putText(color, str(idx), (vx+3, vy-3),
                        font, 0.3, (0,0,0),   4, cv2.LINE_AA)
            cv2.putText(color, str(idx), (vx+3, vy-3),
                        font, 0.3, (0,255,255), 1, cv2.LINE_AA)

        # ------------------------------------------------ tight crop
        mask = np.any(color != 127, axis=2)
        ys, xs = np.where(mask)
        if xs.size and ys.size:
            m  = crop_margin_px
            y0 = max(int(ys.min()) - m, 0)
            y1 = min(int(ys.max()) + m, color.shape[0] - 1)
            x0 = max(int(xs.min()) - m, 0)
            x1 = min(int(xs.max()) + m, color.shape[1] - 1)
            color = color[y0:y1 + 1, x0:x1 + 1]

        # ------------------------------------------------ SCALE BAR  (0.5 m increments)
        # pixel length of 0.5 m
        inc_px = max(int(0.5 / res), 1)
        # make bar as long as will fit (leave 10 px margin on each side)
        max_inc = (color.shape[1] - 20) // inc_px
        max_inc -= max_inc % 1                        # ensure whole increments
        num_inc = max_inc if max_inc else 4           # fall back to 4 inc (2 m)
        bar_len_px = num_inc * inc_px
        bar_len_m  = num_inc * 0.5

        print("max_inc", max_inc, "num_inc", num_inc, "bar_len_px", bar_len_px, "bar_len_m", bar_len_m)

        x0, y0 = 10, color.shape[0] - 10              # start point (BL corner)
        x1     = x0 + bar_len_px

        cv2.line(color, (x0, y0), (x1, y0), (0, 0, 255), 2)      # main bar

        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(num_inc + 1):
            xi = x0 + i * inc_px
            # tick mark
            cv2.line(color, (xi, y0 - 4), (xi, y0 + 4), (0, 0, 255), 2)
            # label every full metre (i × 0.5 is integer when i even)
            if i % 2 == 0:
                metres = i * 0.5
                label  = f"{metres:g}"
                (tw, th), _ = cv2.getTextSize(label, font, 0.4, 1)
                cv2.putText(color, label, (xi - tw // 2, y0 - 6), font,
                            0.4, (0, 0, 255), 1, cv2.LINE_AA)
        # trailing “m”
        cv2.putText(color, "m", (x1 + 4, y0 + 2), font,
                    0.45, (0, 0, 255), 1, cv2.LINE_AA)

        # ------------------------------------------------ save
        Path(out_folder).mkdir(exist_ok=True)
        out_path = os.path.join(out_folder, out_filename)
        # cv2.imwrite(out_path, color)                # no extra flip
        PILImage.fromarray(color[..., ::-1]).save(out_path) 
        rospy.loginfo(f"Graph+robot overlay saved to {out_path}")
        return out_path



