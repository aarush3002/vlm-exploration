"""
Autonomous roaming runner – Gemini edition
-----------------------------------------
Replaces the OpenAI single‑turn helpers with a persistent Gemini
multimodal chat.  The rest of the robotics stack (ROS, RealSense,
SAM, etc.) is untouched.

Prerequisites
-------------
    pip install --upgrade google-generativeai
    export GEMINI_API_KEY="<your‑key>"

"""

import math
import os
import sys
import time
import signal
from pathlib import Path
from typing import List, Union, Dict, Any

import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
from PIL import Image, ImageDraw
import rospy

import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import quaternion_from_euler

# ---- Google Gemini ---------------------------------------------------------
import google.generativeai as genai
from google.generativeai import types as gtypes 

gen_error = genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

VISION_MODEL = "gemini-2.5-pro"        # or flash‑vision for lower cost
MAX_TOKENS   = 20000


def make_image_part(path: str) -> Dict[str, Any]:
    """Return {"mime_type": ..., "data": ...} for an image file."""
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as f:
        return {"mime_type": mime, "data": f.read()}

# Add this new function to aarush_gemini.py

def send_test_goal(client):
    """Sends a single, hardcoded goal to move_base for testing."""
    rospy.loginfo("--- Sending a hardcoded test goal ---")
    goal = MoveBaseGoal()

    # --- Goal Target Pose ---
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()

    # Position from your message
    goal.target_pose.pose.position.x = -2.45
    goal.target_pose.pose.position.y = 2.42
    goal.target_pose.pose.position.z = 0.0

    # Orientation from your message
    goal.target_pose.pose.orientation.x = 0.0
    goal.target_pose.pose.orientation.y = 0.0
    goal.target_pose.pose.orientation.z = 0.9967875335468651
    goal.target_pose.pose.orientation.w = 0.08009127896067898
    # --------------------------

    client.send_goal(goal)
    rospy.loginfo("Test goal sent. Waiting for result...")
    
    # Wait for the server to finish the action
    client.wait_for_result()
    
    result_status = client.get_state()
    if result_status == actionlib.GoalStatus.SUCCEEDED:
        rospy.loginfo("Test goal reached successfully!")
    else:
        rospy.logwarn(f"Failed to reach test goal. Final status: {result_status}")

# def safe_send(chat, parts, *, max_retries=5, wait_s=60):   # ← drop types.Part
#     for attempt in range(1, max_retries + 1):
#         try:
#             resp = chat.send_message(
#                 parts,
#                 generation_config={"max_output_tokens": MAX_TOKENS},
#             )
#             return resp.text
#         except Exception as e:
#             if attempt == max_retries:
#                 raise
#             print(f"[{attempt}/{max_retries}] {e!r} – retrying in {wait_s}s")
#             time.sleep(wait_s)

# ---------------------------------------------------------------------------
# Robot + utility imports (unchanged from original runner)
# ---------------------------------------------------------------------------
from aarush_sam_local import run_sam_inference_and_download_masks, create_annotated_image
from aarush_robot_observer import Robot
from aarush_graph_stuff import Graph, Vertex, euclidean_dist

rospy.init_node("autonomous_roamer")
robot = Robot()

time.sleep(2)

# Motion parameters ----------------------------------------------------------
degrees_per_rotate = 10
picture_interval   = 40       # take a frame every 40° during 360° sweep
robot_speed        = 0.28     # m/s

# ---------------------------------------------------------------------------
# Misc helpers (unchanged – only trimmed for brevity)
# ---------------------------------------------------------------------------

def signal_handler(sig, frame):
    print("Ctrl+C pressed, shutting down.")
    sys.exit(0)


def pil_safe_open(path, max_wait=2):
    t0 = time.time()
    while True:
        try:
            with Image.open(path) as im:
                return im.convert("L")
        except OSError:
            if time.time() - t0 > max_wait:
                raise
            time.sleep(0.1)

# ... (rotate_x_degrees, move_forward_x_meters, save_rgb, etc. – unchanged) ...
# Full definitions omitted for brevity; keep the ones from the original runner.
def save_rgb(color_image, color_image_folder, color_image_name):
    dst_dir = os.path.abspath(color_image_folder)
    dst_path = os.path.join(dst_dir, f"{color_image_name}.png")
    cv2.imwrite(dst_path, color_image)
    return dst_path

# def save_depth():
#     depth_path = "/home/pi/rtabmap_docker/maps/depth.png"
#     depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # uint16
#     depth_m  = depth_raw.astype(np.float32) * 0.001  
#     # depth_m = depth_raw.astype(np.float32) * 0.001
#     return depth_m

def save_depth(depth_m: np.ndarray, folder: str, name: str):
    """
    Save both the **raw** 16‑bit depth (millimetres) and an 8‑bit
    visualisation so you can inspect the frame in an image viewer.

      •  <name>.png      – uint16, depth in **millimetres**
      •  <name>_vis.png  – uint8, depth normalised 0–255
    """
    dst_dir = os.path.abspath(folder)
    os.makedirs(dst_dir, exist_ok=True)

    # ---------- raw 16‑bit PNG ----------
    depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    raw_path = os.path.join(dst_dir, f"{name}.png")
    cv2.imwrite(raw_path, depth_mm)

    # ---------- 8‑bit preview ----------
    valid = depth_m[depth_m > 0]
    if valid.size:
        d_min, d_max = valid.min(), valid.max()
        depth_vis = ((depth_m - d_min) / (d_max - d_min + 1e-6) * 255).astype(np.uint8)
    else:
        depth_vis = np.zeros_like(depth_m, dtype=np.uint8)
    vis_path = os.path.join(dst_dir, f"{name}_vis.png")
    cv2.imwrite(vis_path, depth_vis)

    return raw_path, vis_path

# ---------------------------------------------------------------------------
# Gemini conversation setup
# ---------------------------------------------------------------------------
vision_model = genai.GenerativeModel(VISION_MODEL)

def new_chat():
    return vision_model.start_chat(history=[])

chat = new_chat()
#chat         = vision_model.start_chat(history=[])  # persists over turns

def safe_send(parts: List[Union[str, Dict[str, Any]]],
             *, max_retries: int = 5, wait_s: int = 60) -> str:
    """
    Send a multimodal request; rebuild the chat if the response is empty or
    blocked.  Returns the reply text.
    """
    global chat
    print(chat)
    for attempt in range(1, max_retries + 1):
        try:
            ########################################################################
            # 🔍  DEBUG BLOCK – prints one‑line stats about the outgoing request   #
            ########################################################################
            if os.getenv("GEMINI_DEBUG"):                # enable by:  export GEMINI_DEBUG=1
                size_k  = 0
                part_kinds = []
                for p in parts:
                    if isinstance(p, str):
                        size_k += len(p.encode()) / 1024
                        part_kinds.append("txt")
                    else:                                # {"mime_type": ..., "data": ...}
                        size_k += len(p["data"]) / 1024
                        part_kinds.append(p["mime_type"].split("/")[-1])
                print(f"[safe_send]  {len(parts):2d} parts  "
                    f"{size_k:,.0f} kB  "
                    f"{'+'.join(part_kinds)}")
            ########################################################################
            resp = chat.send_message(
                parts,
                generation_config={"max_output_tokens": MAX_TOKENS},
            )

            cand = resp.candidates[0]
            # finish_reason == 2  → blocked or truncated → treat as failure
            if cand.finish_reason == 2 or not resp.parts:
                raise RuntimeError(
                    f"empty / blocked response (finish_reason={cand.finish_reason})"
                )
            return resp.text

        except Exception as e:
            if attempt == max_retries:
                raise
            print(f"[{attempt}/{max_retries}] {e!r} – resetting chat, retrying in {wait_s}s")
            chat = new_chat()
            time.sleep(wait_s)

def angle_to_rotate(start_x, start_y, start_yaw, end_vertex):
    """Compute the angle (in degrees) the robot must rotate to face end_vertex."""
    dx = end_vertex.x - start_x
    dy = end_vertex.y - start_y
    target_angle_deg = math.degrees(math.atan2(dy, dx))
    delta = target_angle_deg - start_yaw
    # Normalize delta to [-180, 180]
    delta = (delta + 180) % 360 - 180
    #delta = (delta + 360) % 360
    return delta

# def navigate_to_last_dp(path, initial_x, initial_y, initial_yaw):
#     """Follow the given path (a list of Vertex instances) to navigate toward the destination."""
#     curr_x = initial_x
#     curr_y = initial_y
#     # curr_yaw = (initial_yaw + 180) % 360 - 180
#     curr_yaw = (initial_yaw + 360) % 360
#     for next_vertex in path:
#         curr_pos, curr_yaw = robot.pose
#         curr_x = curr_pos["x"]
#         curr_y = curr_pos["y"]

#         angle = angle_to_rotate(curr_x, curr_y, curr_yaw, next_vertex)
#         #rotate_x_degrees(angle)
#         # rotate_to_desired_yaw(curr_yaw + angle, curr_yaw)
#         robot.rotate_deg(angle)
#         distance = euclidean_dist(curr_x, curr_y, next_vertex.x, next_vertex.y)
#         #time_to_move = distance / robot_speed
#         # move_forward_x_seconds(time_to_move)
#         #move_forward_x_meters(distance)
#         robot.drive_forward(distance)

def navigate_to_last_dp(move_base_client, path):
    """
    Follows a path (a list of Vertex instances) by sending each
    vertex as a sequential goal to move_base.
    """
    #rospy.loginfo(f"Starting navigation to backtrack along a path of {len(path)} waypoints.")
    
    rospy.loginfo(f"Starting navigation to last decision point {path[-1].x} {path[-1].y}.")

    current_pos, current_yaw_deg = robot.pose
    current_yaw_rad = math.radians(current_yaw_deg)

    # The orientation for this goal is to face the waypoint
    next_vertex = path[-1]
    dx = next_vertex.x - current_pos['x']
    dy = next_vertex.y - current_pos['y']
    goal_yaw_rad = math.atan2(dy, dx)

    success = navigate_with_move_base(move_base_client, robot, next_vertex.x, next_vertex.y, goal_yaw_rad)

    if not success:
        rospy.logerr("Failed to navigate to a waypoint in the path. Aborting backtrack.")
        return False
    else:
        rospy.loginfo("Successfully navigated the backtrack path.")
        return True
    # Loop through each vertex in the path
    for i, next_vertex in enumerate(path):
        rospy.loginfo(f"Navigating to waypoint {i+1}/{len(path)}...")
        
        # Get the robot's current pose to calculate the orientation for the next goal
        current_pos, current_yaw_deg = robot.pose
        current_yaw_rad = math.radians(current_yaw_deg)

        # The orientation for this goal is to face the waypoint
        dx = next_vertex.x - current_pos['x']
        dy = next_vertex.y - current_pos['y']
        goal_yaw_rad = math.atan2(dy, dx)
        
        # Call the helper function to navigate to the current waypoint
        success = navigate_with_move_base(move_base_client, robot, next_vertex.x, next_vertex.y, goal_yaw_rad)

        # If any waypoint fails, abort the entire backtracking sequence
        if not success:
            rospy.logerr("Failed to navigate to a waypoint in the path. Aborting backtrack.")
            return False
            
    rospy.loginfo("Successfully navigated the backtrack path.")
    return True

def graph_to_string(graph) -> str:
    """
    Returns a human-readable multi-line string.

    Each line shows
        label : (x, y)  ->  label₁(dist₁), label₂(dist₂), ...
    using the chronological labels stored in graph.vertex_labels.
    """
    if not graph.adj_list:
        return "<empty graph>"

    # Build a dict so we can look up a vertex’s label quickly
    label_of = {v: lbl for lbl, v in graph.vertex_labels.items()}

    lines = []
    for lbl in sorted(graph.vertex_labels):          # chronological order
        v = graph.vertex_labels[lbl]
        neighbours = []
        for nb, dist in graph.adj_list.get(v, []):
            n_lbl = label_of.get(nb, "?")
            neighbours.append(f"{n_lbl}({dist:.2f})")
        line = f"{lbl:>3} : ({v.x:.2f}, {v.y:.2f})  ->  " + ", ".join(neighbours)
        lines.append(line)

    return "\n".join(lines)

def calculate_dynamic_offset(distance, min_dist=0.5, max_dist=4.0, min_offset=0.2, max_offset=0.6):
    """
    Calculates a safety offset that scales linearly with distance.
    """
    if distance <= min_dist:
        return min_offset
    if distance >= max_dist:
        return max_offset
    
    # Linear interpolation
    scale = (distance - min_dist) / (max_dist - min_dist)
    offset = min_offset + scale * (max_offset - min_offset)
    
    return offset

def is_goal_in_costmap(goal_x, goal_y, robot):
    """Checks if a goal coordinate is within the costmap bounds and is not an obstacle."""
    # Wait for the costmap to be available
    if robot.costmap_msg is None:
        rospy.logwarn("Costmap not available yet for goal checking.")
        return False

    # Get costmap properties
    origin_x = robot.costmap_msg.info.origin.position.x
    origin_y = robot.costmap_msg.info.origin.position.y
    resolution = robot.costmap_msg.info.resolution
    width = robot.costmap_msg.info.width
    height = robot.costmap_msg.info.height

    # Convert world coordinates to map coordinates
    map_x = int((goal_x - origin_x) / resolution)
    map_y = int((goal_y - origin_y) / resolution)

    # Check if the goal is within the map boundaries
    if map_x < 0 or map_x >= width or map_y < 0 or map_y >= height:
        rospy.logwarn("Goal is outside of the costmap boundaries.")
        return False

    # Check the cost of the goal cell
    # Cost values: -1=unknown, 0=free, 1-99=inflated, 100=lethal obstacle
    cost = robot.costmap_arr[map_y, map_x]

    if cost == -1 or cost == 100:
        rospy.logwarn(f"Goal is in an unknown or lethal obstacle area (cost: {cost}).")
        return False

    return True

def is_at_goal(current_x, current_y, goal_x, goal_y, tolerance=0.1):
    """Check if the robot is within a tolerance radius of the goal."""
    distance = math.sqrt((goal_x - current_x)**2 + (goal_y - current_y)**2)
    return distance <= tolerance

# def navigate_with_move_base(client, goal_x, goal_y, goal_yaw_rad, timeout_seconds=120.0):
#     """
#     Creates a MoveBaseGoal, sends it to the server, and waits for a
#     specified amount of time for it to complete.
#     """
#     goal = MoveBaseGoal()
#     goal.target_pose.header.frame_id = "map"
#     goal.target_pose.header.stamp = rospy.Time.now()

#     goal.target_pose.pose.position.x = goal_x
#     goal.target_pose.pose.position.y = goal_y

#     q = quaternion_from_euler(0, 0, goal_yaw_rad)
#     goal.target_pose.pose.orientation.x = q[0]
#     goal.target_pose.pose.orientation.y = q[1]
#     goal.target_pose.pose.orientation.z = q[2]
#     goal.target_pose.pose.orientation.w = q[3]

#     rospy.loginfo(f"Sending goal to move_base: ({goal_x:.2f}, {goal_y:.2f})")
#     client.send_goal(goal)

#     rospy.loginfo(f"Waiting for result with a {timeout_seconds} second timeout...")
    
#     # --- THIS IS THE KEY CHANGE ---
#     # Wait for the result, but only for a specific duration.
#     finished_in_time = client.wait_for_result(rospy.Duration.from_sec(timeout_seconds))

#     if not finished_in_time:
#         # If the timer ran out, cancel the goal and stop the robot
#         client.cancel_goal()
#         rospy.logwarn(f"Timed out after {timeout_seconds} seconds. Goal canceled.")
#         robot.publish_empty_twist()
#         return False
#     else:
#         # If it finished in time, check the final status
#         result_status = client.get_state()
#         if result_status == actionlib.GoalStatus.SUCCEEDED:
#             rospy.loginfo("Goal reached successfully.")
#             return True
#         else:
#             rospy.logwarn(f"Failed to reach goal. Status code: {result_status}")
#             return False
def get_frontier_midpoint(contour, robot):
    """Calculates the world coordinates of the midpoint of a frontier contour."""
    if contour is None or len(contour) == 0:
        return None
    
    # Calculate the moments of the contour to find the centroid (midpoint)
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    
    # Midpoint in pixel coordinates
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # Convert pixel midpoint to world coordinates
    wx, wy = robot.p2w(cx, cy)
    return (wx, wy)

def find_and_filter_frontiers(robot, frontier_blacklist, max_frontiers=5, max_radius_m=3.0):
    """
    Finds all frontiers, filters by distance and blacklist, and returns the
    closest N frontiers.
    """
    if robot.map_arr is None:
        rospy.logwarn("Map not available for frontier detection.")
        return []

    # 1. Find all frontier contours in the occupancy grid
    all_frontiers = robot.find_frontiers(robot.map_arr, min_length_pixels=20)

    if not all_frontiers:
        return []

    # 2. Calculate midpoint and distance for each frontier
    robot_pos, _ = robot.pose
    frontiers_with_dist = []
    for contour in all_frontiers:
        midpoint_world = get_frontier_midpoint(contour, robot)
        if midpoint_world:
            dist = euclidean_dist(robot_pos['x'], robot_pos['y'], midpoint_world[0], midpoint_world[1])
            if dist <= max_radius_m:
                # Store distance as the primary element for sorting
                frontiers_with_dist.append((dist, midpoint_world, contour))

    # 3. Sort frontiers by distance (closest first)
    frontiers_with_dist.sort(key=lambda x: x[0])
    
    # 4. Filter out blacklisted frontiers
    unblacklisted_frontiers = []
    for frontier_data in frontiers_with_dist:
        midpoint = frontier_data[1]  # The (wx, wy) tuple
        if round_midpoint(midpoint) not in frontier_blacklist:
            unblacklisted_frontiers.append(frontier_data)
    
    if len(unblacklisted_frontiers) < len(frontiers_with_dist):
        rospy.loginfo(f"Ignoring {len(frontiers_with_dist) - len(unblacklisted_frontiers)} blacklisted frontiers.")

    # 5. Return the top N closest, unblacklisted frontiers
    return unblacklisted_frontiers[:max_frontiers]

def get_frontier_orientation(contour, midpoint_world, robot):
    """
    Calculates a goal orientation that is perpendicular to the frontier line,
    pointing into unknown space.
    """
    # Use cv2.fitLine to find the principal axis of the frontier points
    # This gives us a normalized vector (vx, vy) representing the line's direction
    [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)

    # The angle of the frontier line itself
    frontier_angle = math.atan2(vy, vx)

    # The two possible perpendicular angles (90 degrees offset)
    perp_angle_1 = frontier_angle + math.pi / 2.0
    perp_angle_2 = frontier_angle - math.pi / 2.0

    # We need to figure out which perpendicular angle points into unknown space.
    # We do this by checking a test point slightly away from the midpoint.
    test_dist = 0.2 # 20 cm

    # Test point for the first perpendicular angle
    test_x1 = midpoint_world[0] + test_dist * math.cos(perp_angle_1)
    test_y1 = midpoint_world[1] + test_dist * math.sin(perp_angle_1)

    # Check if this test point is in "unknown" space on the raw map
    if not is_goal_in_occupancy_grid(test_x1, test_y1, robot):
        # This direction points towards unknown or occupied, so it's the correct one
        return perp_angle_1
    else:
        # Otherwise, the other perpendicular direction must be correct
        return perp_angle_2

def is_goal_in_open_space(goal_x, goal_y, robot_obj, radius=0.2):
    """
    Checks if a goal and the area in a radius around it are in free space
    on the occupancy grid.
    """
    if robot_obj.map_msg is None:
        rospy.logwarn("Occupancy grid not available for goal checking.")
        return False

    # Get occupancy grid properties
    origin_x = robot_obj.map_msg.info.origin.position.x
    origin_y = robot_obj.map_msg.info.origin.position.y
    resolution = robot_obj.map_msg.info.resolution
    width = robot_obj.map_msg.info.width
    height = robot_obj.map_msg.info.height

    # Convert goal world coordinates to map coordinates
    center_x = int((goal_x - origin_x) / resolution)
    center_y = int((goal_y - origin_y) / resolution)

    # Convert radius in meters to pixels
    radius_px = int(radius / resolution)

    # Check all pixels within a square bounding box around the circle
    for dx in range(-radius_px, radius_px + 1):
        for dy in range(-radius_px, radius_px + 1):
            # Check if the point is within the circle
            if dx**2 + dy**2 <= radius_px**2:
                px = center_x + dx
                py = center_y + dy

                # Check if the pixel is within the map boundaries
                if px < 0 or px >= width or py < 0 or py >= height:
                    return False # Part of the circle is off the map

                # Check the value of the cell
                # OccupancyGrid values: -1=unknown, 0=free, 1-100=occupied
                value = robot_obj.map_arr[py, px]
                if value != 0:
                    return False # Part of the circle is not free space

    # If we checked all pixels in the circle and all were free, the goal is valid
    return True

def is_goal_in_occupancy_grid(goal_x, goal_y, robot_obj):
    """
    Checks if a goal coordinate is within the SLAM map bounds and in free space.
    """
    if robot_obj.map_msg is None:
        rospy.logwarn("Occupancy grid not available yet for goal checking.")
        return False

    # Get occupancy grid properties from the robot object
    origin_x = robot_obj.map_msg.info.origin.position.x
    origin_y = robot_obj.map_msg.info.origin.position.y
    resolution = robot_obj.map_msg.info.resolution
    width = robot_obj.map_msg.info.width
    height = robot_obj.map_msg.info.height

    # Convert world coordinates to map coordinates
    map_x = int((goal_x - origin_x) / resolution)
    map_y = int((goal_y - origin_y) / resolution)

    # Check if the goal is within the map boundaries
    if map_x < 0 or map_x >= width or map_y < 0 or map_y >= height:
        return False

    # Check the value of the goal cell
    # OccupancyGrid values: -1=unknown, 0=free, 1-100=occupied
    value = robot_obj.map_arr[map_y, map_x]

    # A valid point must be in a known "free" cell
    return value == 0

def find_best_intermediate_goal(start_x, start_y, goal_x, goal_y, robot_obj):
    """
    Traces a line from an out-of-bounds goal back towards the robot to find
    the furthest point that is in a reasonably open area.
    """
    rospy.loginfo("Finding best intermediate goal in open space...")
    direction_x = goal_x - start_x
    direction_y = goal_y - start_y
    magnitude = math.sqrt(direction_x**2 + direction_y**2)

    if magnitude == 0:
        return None

    step = robot_obj.map_res if robot_obj.map_res is not None else 0.05

    # Start from the original goal and step backwards towards the robot
    for i in range(int(magnitude / step), 0, -1):
        dist = i * step
        test_x = start_x + (direction_x / magnitude) * dist
        test_y = start_y + (direction_y / magnitude) * dist

        # THIS IS THE KEY CHANGE: Call the new, smarter checker function
        if is_goal_in_open_space(test_x, test_y, robot_obj):
            rospy.loginfo(f"Found valid intermediate goal at ({test_x:.2f}, {test_y:.2f})")
            return (test_x, test_y)

    rospy.logwarn("Could not find any valid intermediate goal on the path.")
    return None

def navigate_with_move_base(client, robot_obj, goal_x, goal_y, goal_yaw_rad,
                            total_timeout_s=120.0,
                            stuck_time_s=60.0,
                            stuck_dist_m=0.1):
    """
    Sends a goal to move_base and actively monitors for progress. If the robot
    does not move a minimum distance in a set amount of time, it cancels the goal.
    """
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = goal_x
    goal.target_pose.pose.position.y = goal_y
    q = quaternion_from_euler(0, 0, goal_yaw_rad)
    goal.target_pose.pose.orientation.x = q[0]
    goal.target_pose.pose.orientation.y = q[1]
    goal.target_pose.pose.orientation.z = q[2]
    goal.target_pose.pose.orientation.w = q[3]

    rospy.loginfo(f"Sending goal to move_base: ({goal_x:.2f}, {goal_y:.2f})")
    client.send_goal(goal)

    # --- NEW PROGRESS MONITORING LOGIC ---
    start_time = rospy.Time.now()
    last_check_time = rospy.Time.now()
    last_check_pos, _ = robot_obj.pose

    rate = rospy.Rate(2) # Check roughly twice per second
    while not rospy.is_shutdown():
    #while not rospy.is_shutdown() and (rospy.Time.now() - start_time).to_sec() < total_timeout_s:
        # Check if the goal has already completed
        goal_status = client.get_state()
        if goal_status in [actionlib.GoalStatus.SUCCEEDED, actionlib.GoalStatus.ABORTED,
                           actionlib.GoalStatus.REJECTED, actionlib.GoalStatus.PREEMPTED]:
            break

        # Check if enough time has passed to check for progress
        if (rospy.Time.now() - last_check_time).to_sec() > stuck_time_s:
            current_pos, _ = robot_obj.pose
            distance_moved = euclidean_dist(current_pos['x'], current_pos['y'],
                                            last_check_pos['x'], last_check_pos['y'])
            
            # If the robot hasn't moved enough, consider it stuck
            if distance_moved < stuck_dist_m:
                rospy.logwarn(f"Robot has not moved >{stuck_dist_m}m in {stuck_time_s}s. Canceling goal.")
                client.cancel_goal()
                robot_obj.publish_empty_twist()
                return False # Return failure
            
            # If it has moved, reset the progress checker
            last_check_time = rospy.Time.now()
            last_check_pos = current_pos

        rate.sleep()
    # --- END OF NEW LOGIC ---

    final_status = client.get_state()
    if final_status == actionlib.GoalStatus.SUCCEEDED:
        rospy.loginfo("Goal reached successfully.")
        return True
    else:
        # Handle cases where the goal failed within the timeout, or the main timeout was reached
        if (rospy.Time.now() - start_time).to_sec() >= total_timeout_s:
             rospy.logwarn(f"Total navigation timeout of {total_timeout_s}s reached. Canceling goal.")
             client.cancel_goal()
             robot.publish_empty_twist()

        rospy.logwarn(f"Failed to reach goal. Final status code: {final_status}")
        return False

def round_midpoint(midpoint_tuple):
    """Rounds the x and y coordinates of a midpoint to one decimal place."""
    return (round(midpoint_tuple[0], 1), round(midpoint_tuple[1], 1))

# ---------------------------------------------------------------------------
# Main control loop
# ---------------------------------------------------------------------------

def main():

    start_time = time.time()
    graph = Graph()
    decision_point_stack = []
    frontier_blacklist = set()
    prev_vertex          = None
    traversal_vertex     = None

    counter = 0
    prev_traveled_yaw = None

    move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    rospy.loginfo("Waiting for move_base action server...")
    # Wait up to 60 seconds for the server to become available
    if not move_base_client.wait_for_server(rospy.Duration(60)):
        rospy.logerr("Could not connect to move_base server. Is it running?")
        return # Exit if the server is not available
    rospy.loginfo("Connected to move_base action server.")

    # send_test_goal(move_base_client)
    # return
    while True:
        # 1. Current pose ----------------------------------------------------
        starting_position, starting_yaw = robot.pose
        if counter == 0:
            robot.graph_origin = robot.pose

        starting_vertex = traversal_vertex or Vertex(starting_position["x"], starting_position["y"])
        if prev_vertex is not None and traversal_vertex is None:
            graph.create_edge(prev_vertex, starting_vertex)

        print(f"Pos: {starting_position} | Yaw: {starting_yaw:.1f}°")

        closest_frontiers = find_and_filter_frontiers(robot, frontier_blacklist, max_frontiers=5, max_radius_m=3.0)

        occ_map_path = robot.overlay_graph_on_occupancy_map(
            graph, (starting_vertex.x, starting_vertex.y), starting_yaw,
            closest_frontiers,
            out_folder="aarush_map_graphs",
            out_filename=f"map_graph_-1.png")

        # 2. 360° sweep ------------------------------------------------------
        # images_dir     = f"aarush_VLM_{counter}_color"
        # os.makedirs(images_dir, exist_ok=True)
        multi = 360 // picture_interval

        # starting_map = {}
        # angles       = []

        for i in range(multi):
            pos_dict, raw_yaw = robot.pose
            print(f"Pos: {pos_dict} | Yaw: {raw_yaw:.1f}°")
            # rgb, _   = robot.capture_rgbd()
            # img_path = save_rgb(rgb, images_dir, i)

            # heading  = (raw_yaw + 360) % 360
            # angles.append(heading)
            # starting_map[i] = [img_path, heading]

            robot.rotate_deg(-picture_interval)

        closest_frontiers = find_and_filter_frontiers(robot, frontier_blacklist, max_frontiers=5, max_radius_m=3.0)
        #print(closest_frontiers)

        if not closest_frontiers:
            rospy.logwarn("No valid frontiers found. Attempting to backtrack.")
            # (Your backtracking logic would go here, using the decision_point_stack)
            if len(decision_point_stack) > 0:
                traversal_vertex = decision_point_stack.pop()
                shortest_path = graph.dijkstra(starting_vertex, traversal_vertex)
                curr_pose, curr_yaw = robot.pose
                # navigate_to_last_dp(shortest_path, curr_pose["x"], curr_pose["y"], curr_yaw)
                occ_map_path = robot.overlay_graph_on_occupancy_map(
                    graph, (starting_vertex.x, starting_vertex.y), starting_yaw,
                    closest_frontiers,
                    out_folder=f"aarush_gemini_run_{counter}",
                    out_filename=f"map_graph_{counter}.png")
                navigate_to_last_dp(move_base_client, shortest_path)
                robot.save_raw_map(out_folder="map_graphs", out_filename=f"map_graphs_{counter}.png")
                counter +=1
                continue
            else:
                #Searching over 60m radius for any loose frontiers
                closest_frontiers = find_and_filter_frontiers(robot, frontier_blacklist, max_frontiers=5, max_radius_m=60.0)
                if not closest_frontiers:
                    rospy.loginfo("No backtrack path. Exploration complete.")
                    break

        if len(closest_frontiers) > 1:
            # (Your decision point stack logic would be added here)
            decision_point_stack.append(starting_vertex)
            rospy.loginfo(f"Decision point found with {len(closest_frontiers)} options.")


        images_dir = f"aarush_gemini_run_{counter}"
        # 3. Save overlay map ------------------------------------------------
        occ_map_path = robot.overlay_graph_on_occupancy_map(
            graph, (starting_vertex.x, starting_vertex.y), starting_yaw,
            closest_frontiers, # <-- ADD THIS ARGUMENT
            out_folder=images_dir, out_filename=f"map_graph_{counter}.png"
        )

        os.makedirs(images_dir, exist_ok=True)
        frontier_views = {}
        rospy.loginfo(f"Capturing images of {len(closest_frontiers)} frontiers...")

        for i, (dist, midpoint, contour) in enumerate(closest_frontiers):
            # Calculate angle to the frontier's midpoint
            curr_position, curr_yaw = robot.pose

            dx = midpoint[0] - curr_position['x']
            dy = midpoint[1] - curr_position['y']

            target_yaw_rad = math.atan2(dy, dx)
            target_yaw_deg = (math.degrees(target_yaw_rad) + 360) % 360
            
            # Rotate to face the frontier
            robot.rotate_deg(target_yaw_deg - curr_yaw)
            time.sleep(1.0) # Pause to ensure robot is stable

            # Capture and save the image
            rgb, _ = robot.capture_rgbd()
            img_path = save_rgb(rgb, images_dir, f"frontier_{i}")
            
            frontier_views[i] = {
                "path": img_path,
                "heading": target_yaw_deg,
                "midpoint": midpoint,
                "distance": dist
            }


        # 4. Gemini: choose best direction ----------------------------------
        prompt = """You are an autonomous exploration robot. Your goal is to explore the environment as quickly as possible.
                    You are given a top-down occupancy map and a series of images. Each image corresponds to a "frontier," which is a boundary between known and unknown space.
                    Based on the images and the map, choose the single best frontier to navigate towards to maximize new exploration. Avoid images/frontiers that are 
                    very close to completely pitch-black areas as the black areas are non-traversable and will end exploration immediately if traversed to.
                    Your answer must be only the number of the chosen frontier (e.g., "0", "1", etc.) with absolutely nothing else. On the next line after this,
                    explain your reasoning as to how you chose the frontier."""
        
        parts: List[types.Part | str] = [
            prompt
        ]

        parts = [prompt, make_image_part(occ_map_path), "Current occupancy map."]
        for i, view_data in frontier_views.items():
            parts.append(make_image_part(view_data["path"]))
            parts.append(f"Image for Frontier {i}.")

        #full_response = safe_send(chat, parts)
        rospy.loginfo("Asking VLM to choose the best frontier...")
        full_response = safe_send(parts)
        rospy.loginfo(f"VLM response: '{full_response}'")
        first_resp_path = Path(f"{images_dir}/first_response.txt")
        first_resp_path.write_text(full_response)

        try:
            # --- Step 10: VLM returns a choice ---
            chosen_idx = int(full_response.split("\n")[0].strip())
            chosen_frontier = frontier_views[chosen_idx]

            desired_midpoint = chosen_frontier['midpoint']

            # --- NEW LOGIC: Determine the single best reachable goal AND orientation ---
            current_pos, _ = robot.pose

            # # Find the furthest reachable point on the map towards the desired midpoint
            # navigable_goal_coords = find_best_intermediate_goal(
            #     current_pos['x'], current_pos['y'],
            #     desired_midpoint[0], desired_midpoint[1],
            #     robot
            # )
            
            goal_x, goal_y = desired_midpoint[0], desired_midpoint[1]

            frontier_contour = closest_frontiers[chosen_idx][2]
            goal_yaw_rad = get_frontier_orientation(frontier_contour, (goal_x, goal_y), robot)

            # Call the navigation function with the new, smarter orientation
            success = navigate_with_move_base(move_base_client, robot, goal_x, goal_y, goal_yaw_rad)
            rounded_midpoint = round_midpoint(desired_midpoint)
            frontier_blacklist.add(rounded_midpoint)
            if not success:
                rospy.logwarn(f"Navigation to {rounded_midpoint} failed.")

        except (ValueError, KeyError) as e:
            rospy.logerr(f"Could not parse VLM response or find chosen frontier: {e}. Skipping this step.")

        robot.save_raw_map(out_folder="map_graphs", out_filename=f"map_graphs_{counter}.png")
        prev_vertex = starting_vertex
        traversal_vertex = None
        counter += 1


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    main()
