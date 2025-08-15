#!/usr/bin/env python
"""
Autonomous roaming runner - Gemini edition
-----------------------------------------
An autonomous exploration script using a multimodal VLM (Gemini) for
high-level decision making and the ROS navigation stack (move_base) for
robust, low-level motion control.
"""

import math
import os
import sys
import time
import signal
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Twist
from tf.transformations import quaternion_from_euler
import google.generativeai as genai

from aarush_robot_observer import Robot
from aarush_graph_stuff import Graph, Vertex, euclidean_dist

# =========================================================================== #
# Gemini Configuration and Helpers
# =========================================================================== #

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
VISION_MODEL = "gemini-2.5-pro"
chat = None
experiment_name = "2n8kARJN3HM"

def new_chat():
    """Creates a new Gemini chat session."""
    global chat
    model = genai.GenerativeModel(VISION_MODEL)
    chat = model.start_chat(history=[])
    rospy.loginfo("New Gemini chat session started.")

def make_image_part(path: str) -> Dict[str, Any]:
    """Loads an image file for the VLM prompt."""
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as f:
        return {"mime_type": mime, "data": f.read()}

def safe_send(parts: List[Any], max_retries: int = 3, wait_s: int = 5) -> str:
    """Sends a request to the Gemini API with error handling and retries."""
    global chat
    if chat is None: new_chat()
    for attempt in range(1, max_retries + 1):
        try:
            resp = chat.send_message(parts, stream=False)
            if not resp.text:
                 raise RuntimeError(f"VLM returned an empty or blocked response. Finish reason: {resp.candidates[0].finish_reason}")
            return resp.text
        except Exception as e:
            if attempt == max_retries: raise
            rospy.logwarn(f"Gemini API error: {e}. Resetting chat and retrying in {wait_s}s.")
            new_chat()
            time.sleep(wait_s)

def save_rgb(color_image, folder, name):
    """Saves an RGB image to a file."""
    path = os.path.join(os.path.abspath(folder), f"{name}.png")
    cv2.imwrite(path, cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
    return path

# =========================================================================== #
# Frontier and Navigation Logic
# =========================================================================== #

def get_frontier_midpoint(contour, robot: Robot) -> Tuple[float, float]:
    """Calculates the world coordinates of the midpoint of a frontier contour."""
    if contour is None or len(contour) == 0: return None
    M = cv2.moments(contour)
    if M["m00"] == 0: return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return robot.p2w(cx, cy)

def is_goal_in_occupancy_grid(goal_x, goal_y, robot_obj: Robot):
    """Checks if a goal coordinate is within the SLAM map bounds and in free space."""
    if robot_obj.map_msg is None: return False
    origin_x, origin_y = robot_obj.map_origin
    resolution = robot_obj.map_res
    width, height = robot_obj.map_arr.shape[1], robot_obj.map_arr.shape[0]
    map_x = int((goal_x - origin_x) / resolution)
    map_y = int((goal_y - origin_y) / resolution)
    if not (0 <= map_x < width and 0 <= map_y < height): return False
    return robot_obj.map_arr[map_y, map_x] == 0

def get_frontier_orientation(contour, midpoint_world: Tuple[float, float], robot: Robot) -> float:
    """Calculates a goal orientation that is perpendicular to the frontier line."""
    [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    frontier_angle = math.atan2(vy, vx)
    perp_angle_1 = frontier_angle + math.pi / 2.0
    test_dist = 0.2
    test_x1 = midpoint_world[0] + test_dist * math.cos(perp_angle_1)
    test_y1 = midpoint_world[1] + test_dist * math.sin(perp_angle_1)
    if not is_goal_in_occupancy_grid(test_x1, test_y1, robot):
        return perp_angle_1
    else:
        return frontier_angle - math.pi / 2.0

def round_midpoint(midpoint_tuple):
    """Rounds the x and y coordinates of a midpoint to one decimal place."""
    # return (round(midpoint_tuple[0], 1), round(midpoint_tuple[1], 1))
    x, y = midpoint_tuple
    rounded_x = round(x / 0.15) * 0.15
    rounded_y = round(y / 0.15) * 0.15
    return (round(rounded_x, 1), round(rounded_y, 1))

# def find_and_filter_frontiers(robot: Robot, frontier_blacklist: set, max_frontiers=5, max_radius_m=3.0) -> List:
#     """Finds, filters by blacklist/distance, sorts by distance, and returns the top N frontiers."""
#     if robot.map_arr is None: return []
#     all_frontiers = robot.find_frontiers(robot.map_arr, min_length_pixels=20)
#     if not all_frontiers: return []
#     robot_pos, _ = robot.pose
#     frontiers_with_dist = []
#     for contour in all_frontiers:
#         midpoint_world = get_frontier_midpoint(contour, robot)
#         if midpoint_world:
#             dist = euclidean_dist(robot_pos['x'], robot_pos['y'], midpoint_world[0], midpoint_world[1])
#             if dist <= max_radius_m:
#                 frontiers_with_dist.append((dist, midpoint_world, contour))
    
#     frontiers_with_dist.sort(key=lambda x: x[0]) # Sort by distance (closest first)
    
#     unblacklisted_frontiers = []
#     for frontier_data in frontiers_with_dist:
#         midpoint = frontier_data[1]
#         if round_midpoint(midpoint) not in frontier_blacklist:
#             unblacklisted_frontiers.append(frontier_data)
    
#     if len(unblacklisted_frontiers) < len(frontiers_with_dist):
#         rospy.loginfo(f"Ignoring {len(frontiers_with_dist) - len(unblacklisted_frontiers)} blacklisted frontiers.")

#     return unblacklisted_frontiers[:max_frontiers]

def find_and_filter_frontiers(robot, cand_x, cand_y, frontier_blacklist: set, max_frontiers=5, max_radius_m=3.0) -> List:
    """Finds, filters by blacklist/distance, sorts by distance, and returns the top N frontiers."""
    if robot.map_arr is None: return []
    all_frontiers = robot.find_frontiers(robot.map_arr, min_length_pixels=20)
    if not all_frontiers: return []
    frontiers_with_dist = []
    for contour in all_frontiers:
        midpoint_world = get_frontier_midpoint(contour, robot)
        if midpoint_world:
            dist = euclidean_dist(cand_x, cand_y, midpoint_world[0], midpoint_world[1])
            if dist <= max_radius_m:
                frontiers_with_dist.append((dist, midpoint_world, contour))
    
    frontiers_with_dist.sort(key=lambda x: x[0]) # Sort by distance (closest first)
    
    unblacklisted_frontiers = []
    for frontier_data in frontiers_with_dist:
        midpoint = frontier_data[1]
        if round_midpoint(midpoint) not in frontier_blacklist:
            unblacklisted_frontiers.append(frontier_data)
    
    if len(unblacklisted_frontiers) < len(frontiers_with_dist):
        rospy.loginfo(f"Ignoring {len(frontiers_with_dist) - len(unblacklisted_frontiers)} blacklisted frontiers.")

    return unblacklisted_frontiers[:max_frontiers]

def perceive_and_capture(robot: Robot, closest_frontiers: List, images_dir: str) -> Dict:
    """Rotates to face each frontier and captures an image."""
    rospy.loginfo(f"Capturing images of {len(closest_frontiers)} frontiers...")
    frontier_views = {}
    _, starting_yaw = robot.pose
    for i, (dist, midpoint, contour) in enumerate(closest_frontiers):
        robot_pos, _ = robot.pose
        dx = midpoint[0] - robot_pos['x']
        dy = midpoint[1] - robot_pos['y']
        target_yaw_rad = math.atan2(dy, dx)
        target_yaw_deg = (math.degrees(target_yaw_rad) + 360) % 360
        _, current_yaw = robot.pose
        robot.rotate_deg(target_yaw_deg - current_yaw)
        time.sleep(1.0)
        rgb, _ = robot.capture_rgbd()
        img_path = save_rgb(rgb, images_dir, f"frontier_{i}")
        frontier_views[i] = {"path": img_path, "heading": target_yaw_deg, "midpoint": midpoint, "distance": dist}
    #robot.rotate_deg(starting_yaw - robot.pose[1]) # Return to original orientation
    return frontier_views

def get_vlm_choice(frontier_views: Dict, map_image_path: str) -> int:
    """Builds a prompt, sends it to Gemini, and parses the response."""
    prompt = """You are an autonomous exploration robot. Your goal is to fully explore/cover the entire environment as quickly and efficiently as possible.
    You are given the current top-down occupancy map and a series of images. Each image corresponds to a numbered "frontier" on the map.
    Based on the images, the map, and previous steps, choose the single best frontier to navigate towards to maximize exploration efficiency 
    while minimizing total distance traveled. If an area contains any frontiers, it has not been fully explored - you should select frontiers in
    the area you are currently in to fully explore it until there are no more frontiers in that area, before 
    moving onto the next area (similar to depth-first search). Avoid frontiers/images close to fully pitch-black areas, as these areas are strictly not traversable. 
    You are also unable to climb stairs. Your answer must be only the number of the chosen frontier (e.g., "0", "1", etc.) with absolutely nothing else. 
    Starting on the next line directly after this, please explain your reasoning as to how you chose the frontier."""
    
    parts = [prompt, make_image_part(map_image_path), "Current occupancy map."]
    for i, view_data in frontier_views.items():
        parts.append(make_image_part(view_data["path"]))
        parts.append(f"Image for Frontier {i}.")

    rospy.loginfo("Asking VLM to choose the best frontier...")
    response = safe_send(parts)
    rospy.loginfo(f"VLM response: '{response}'")
    Path(os.path.dirname(map_image_path)).joinpath("vlm_response.txt").write_text(response)
    try:
        return int(response.strip().split('\n')[0])
    except (ValueError, IndexError):
        rospy.logerr("Could not parse VLM response.")
        return None

def execute_navigation(move_base_client, robot: Robot, chosen_frontier_data: Tuple, frontier_blacklist: set) -> bool:
    """Calculates a safe goal from the chosen frontier and navigates there."""
    dist, midpoint, contour = chosen_frontier_data
    current_pos, _ = robot.pose
    safety_offset = 0.0
    safe_distance = dist - safety_offset
    if safe_distance <= 0:
        rospy.logwarn("Chosen frontier is too close. Blacklisting.")
        frontier_blacklist.add(round_midpoint(midpoint))
        return False
    dx_vec = midpoint[0] - current_pos['x']
    dy_vec = midpoint[1] - current_pos['y']
    vec_len = math.sqrt(dx_vec**2 + dy_vec**2)
    dx_norm, dy_norm = dx_vec / vec_len, dy_vec / vec_len
    goal_x = current_pos['x'] + dx_norm * safe_distance
    goal_y = current_pos['y'] + dy_norm * safe_distance
    goal_yaw_rad = get_frontier_orientation(contour, midpoint, robot)
    success = navigate_with_move_base(move_base_client, robot, goal_x, goal_y, goal_yaw_rad)
    frontier_blacklist.add(round_midpoint(midpoint))
    if not success:
        rospy.logwarn(f"Navigation failed. Blacklisting region around {round_midpoint(midpoint)}.")
    else:
        rospy.logwarn(f"Navigation success. Blacklisting region around {round_midpoint(midpoint)} to prevent repeats.")
    return success

def handle_dead_end(move_base_client, robot, graph, decision_point_stack, starting_vertex, frontier_blacklist) -> Vertex:
    """Handles the logic for backtracking when no frontiers are found."""
    if not decision_point_stack:
        return None # Signal to main loop that backtracking failed
    rospy.loginfo("Dead end detected. Backtracking to the last decision point...")
    pos_dict, _ = robot.pose
    curr_x, curr_y = pos_dict["x"], pos_dict["y"]
    sorted_decision_point_stack = sorted(decision_point_stack, key=lambda v: euclidean_dist(curr_x, curr_y, v.x, v.y))
    # traversal_vertex = decision_point_stack.pop()
    traversal_vertex = sorted_decision_point_stack.pop(0)
    decision_point_stack.remove(traversal_vertex)

    #We don't need to revisit decision points whose frontiers have been covered by previous traversals.
    while len(find_and_filter_frontiers(robot, traversal_vertex.x, traversal_vertex.y, frontier_blacklist, max_frontiers=5, max_radius_m=3.0)) == 0:
        if not decision_point_stack:
            return None 
        #traversal_vertex = decision_point_stack.pop()
        traversal_vertex = sorted_decision_point_stack.pop(0)
        decision_point_stack.remove(traversal_vertex)
        print("Popped updated decision point")

    shortest_path = graph.dijkstra(starting_vertex, traversal_vertex)
    success = navigate_to_last_dp(move_base_client, robot, shortest_path)
    # if not success:
    #     rospy.logwarn("Navigation to decision point failed. Retrying handle_dead_end.")
    #     handle_dead_end(move_base_client, robot, graph, decision_point_stack, starting_vertex, frontier_blacklist)
    # if not success:
    #     robot.publish_unstuck_twist()
    return traversal_vertex

def navigate_with_move_base(client, robot_obj, goal_x, goal_y, goal_yaw_rad, total_timeout_s=300.0, stuck_time_s=60.0, stuck_dist_m=0.1):
    """Sends a goal to move_base and actively monitors for progress."""
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
    client.send_goal(goal)
    start_time = rospy.Time.now()
    last_check_time = rospy.Time.now()
    last_check_pos, _ = robot_obj.pose
    rate = rospy.Rate(2)
    while not rospy.is_shutdown() and (rospy.Time.now() - start_time).to_sec() < total_timeout_s:
        goal_status = client.get_state()
        if goal_status in [actionlib.GoalStatus.SUCCEEDED, actionlib.GoalStatus.ABORTED, actionlib.GoalStatus.REJECTED, actionlib.GoalStatus.PREEMPTED]:
            break
        if (rospy.Time.now() - last_check_time).to_sec() > stuck_time_s:
            current_pos, _ = robot_obj.pose
            distance_moved = euclidean_dist(current_pos['x'], current_pos['y'], last_check_pos['x'], last_check_pos['y'])
            if distance_moved < stuck_dist_m:
                rospy.logwarn(f"Robot stuck. Canceling goal.")
                client.cancel_goal()
                robot_obj.publish_empty_twist()
                return False
            last_check_time = rospy.Time.now()
            last_check_pos = current_pos
        rate.sleep()
    final_status = client.get_state()
    if final_status == actionlib.GoalStatus.SUCCEEDED:
        rospy.loginfo("Goal reached successfully.")
        return True
    else:
        if (rospy.Time.now() - start_time).to_sec() >= total_timeout_s:
             rospy.logwarn(f"Total navigation timeout reached. Canceling goal.")
             client.cancel_goal()
             robot_obj.publish_empty_twist()
        rospy.logwarn(f"Failed to reach goal. Final status code: {final_status}")
        return False

def navigate_to_last_dp(client, robot_obj, path):
    """Navigates to the final vertex in a given path for backtracking."""
    if not path: return False
    goal_vertex = path[-1]
    if len(path) > 1:
        prev_vertex = path[-2]
        dx, dy = goal_vertex.x - prev_vertex.x, goal_vertex.y - prev_vertex.y
    else:
        current_pos, _ = robot_obj.pose
        dx, dy = goal_vertex.x - current_pos['x'], goal_vertex.y - current_pos['y']
    goal_yaw_rad = math.atan2(dy, dx)
    return navigate_with_move_base(client, robot_obj, goal_vertex.x, goal_vertex.y, goal_yaw_rad)

# =========================================================================== #
# Main Control Loop
# =========================================================================== #

def main(robot: Robot):
    """The main exploration state machine."""
    graph = Graph()
    decision_point_stack = []
    frontier_blacklist = set()
    prev_vertex, traversal_vertex = None, None
    counter = 0

    move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    rospy.loginfo("Waiting for move_base action server...")
    if not move_base_client.wait_for_server(rospy.Duration(60)):
        rospy.logerr("Could not connect to move_base server. Exiting."); return
    rospy.loginfo("Connected to move_base. Giving system a moment to warm up...")
    #rospy.sleep(5.0)

    while not rospy.is_shutdown():
        rospy.loginfo(f"\n----- Iteration {counter}: Starting new exploration cycle -----")

        for i in range(9):
            pos_dict, raw_yaw = robot.pose
            print(f"Pos: {pos_dict} | Yaw: {raw_yaw:.1f}°")
            robot.rotate_deg(40)
        
        current_pos, current_yaw = robot.pose
        starting_vertex = traversal_vertex or Vertex(current_pos["x"], current_pos["y"])
        if prev_vertex and not traversal_vertex:
            graph.create_edge(prev_vertex, starting_vertex)
        
        closest_frontiers = find_and_filter_frontiers(robot, current_pos["x"], current_pos["y"], frontier_blacklist, max_frontiers=5, max_radius_m=3.0)
        
        if not closest_frontiers:
            rospy.logwarn("No valid frontiers found in the 3m radius.")
            if not decision_point_stack:
                rospy.loginfo("Decision point stack empty, exploration complete.")
                # rospy.loginfo("Decision point stack is empty. Performing a final wide-radius search...")
                # closest_frontiers = find_and_filter_frontiers(robot, current_pos["x"], current_pos["y"], frontier_blacklist, max_frontiers=5, max_radius_m=60.0)
                # if not closest_frontiers:
                #     rospy.loginfo("Wide-radius search found no new frontiers. Exploration complete.")
                #     images_dir = f"/home/aarush/final_gemini_results/{experiment_name}/aarush_gemini_run_{counter}"
                #     map_image_path = robot.overlay_graph_on_occupancy_map(
                #         graph, (current_pos['x'], current_pos['y']), current_yaw, closest_frontiers,
                #         out_folder=images_dir, out_filename=f"map_graph_{counter}.png"
                #     )
                #     break
                # rospy.loginfo(f"Found {len(closest_frontiers)} distant frontiers. Continuing exploration.")
                break
            else:
                print("Before handle_dead_end", decision_point_stack)
                traversal_vertex = handle_dead_end(move_base_client, robot, graph, decision_point_stack, starting_vertex, frontier_blacklist)
                print("After handle_dead_end", decision_point_stack)
                if traversal_vertex is None: break
                prev_vertex, counter = starting_vertex, counter + 1
                continue

        if len(closest_frontiers) > 1:
            rospy.loginfo(f"Decision point found with {len(closest_frontiers)} options. Saving to stack.")
            decision_point_stack.append(starting_vertex)
        
        images_dir = f"/home/aarush/final_gemini_results/{experiment_name}/aarush_gemini_run_{counter}"
        os.makedirs(images_dir, exist_ok=True)
        frontier_views = perceive_and_capture(robot, closest_frontiers, images_dir)
        map_image_path = robot.overlay_graph_on_occupancy_map(
            graph, (current_pos['x'], current_pos['y']), current_yaw, closest_frontiers,
            out_folder=images_dir, out_filename=f"map_graph_{counter}.png"
        )
        
        if len(closest_frontiers) > 1:
            chosen_idx = get_vlm_choice(frontier_views, map_image_path)
        else:
            chosen_idx = 0

        if chosen_idx is not None and chosen_idx < len(closest_frontiers):
            chosen_frontier_data = closest_frontiers[chosen_idx]
            execute_navigation(move_base_client, robot, chosen_frontier_data, frontier_blacklist)
        else:
            rospy.logerr("VLM returned an invalid choice or failed. Blacklisting visible frontiers.")
            for _, midpoint, _ in closest_frontiers:
                frontier_blacklist.add(round_midpoint(midpoint))

        prev_vertex, traversal_vertex, counter = starting_vertex, None, counter + 1
        rospy.sleep(1.0)

    rospy.loginfo("--- Exploration Finished ---")

def signal_handler(sig, frame):
    """Gracefully shut down on Ctrl+C."""
    rospy.loginfo("Ctrl+C pressed, shutting down.")
    sys.exit(0)

if __name__ == "__main__":
    rospy.init_node("autonomous_roamer")
    signal.signal(signal.SIGINT, signal_handler)
    
    robot = Robot()
    rospy.loginfo("Waiting for first odometry message...")
    while robot.pose is None and not rospy.is_shutdown():
        rospy.sleep(0.1)

    start_time = time.time()
    try:
        main(robot)
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt received. Shutting down.")
    except Exception as e:
        rospy.logerr(f"An unhandled exception occurred in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        total_time = time.time() - start_time
        rospy.loginfo(f"--- Exploration Script Finished ---")
        rospy.loginfo(f"Total time elapsed: {total_time:.2f} seconds")
        robot.save_raw_map(out_folder=f"/home/aarush/final_gemini_results/{experiment_name}")
        time_path = f"/home/aarush/final_gemini_results/{experiment_name}/total_time.txt"
        with open(time_path, "w") as f:
            f.write(f"{total_time:.2f}\n")
