import math
import numpy as np
import os
import cv2
import time
import torch
import sys
import signal
import imageio.v2 as iio
import shutil
import io
import matplotlib.pyplot as plt
import yaml
from PIL import Image, ImageDraw 
from pathlib import Path
import rospy

# Import your pre-trained policy and VLM/SAM utilities
from aarush_run_combine_tools import encode_image, query_with_sequence, get_best_segment
#from aarush_sam_testing import run_sam_inference_and_download_masks, create_annotated_image
from aarush_sam_local import run_sam_inference_and_download_masks, create_annotated_image

# Import observer and graph utilities
from aarush_robot_observer import Robot
from aarush_graph_stuff import Graph, Vertex, euclidean_dist, scale_intrinsics
# from aarush_open3d import *

rospy.init_node("autonomous_roamer")
robot = Robot()
time.sleep(2)
degrees_per_rotate = 10 #10
picture_interval = 40 #40
robot_speed = 0.28

def pil_safe_open(path, max_wait=2):
    t0 = time.time()
    while True:
        try:
            with Image.open(path) as im:
                return im.convert("L")      # force 8-bit gray
        except OSError:                     # file incomplete
            if time.time() - t0 > max_wait:
                raise
            time.sleep(0.1)
            
def signal_handler(sig, frame):
    print("Ctrl+C pressed, shutting down.")
    sys.exit(0)

# Control helper functions

def rotate_x_degrees(degrees):
    """Rotate the robot by issuing a series of action commands."""
    direction = 'turn_right_middle' if degrees < 0 else 'turn_left_middle'
    multi = abs(degrees) // degrees_per_rotate
    print(f"Rotating {direction} {multi} times")
    for i in range(int(multi)):
        AGC.run_action(direction)

def rotate_to_desired_yaw(desired_yaw, current_yaw):
    print("Desired yaw", (desired_yaw + 360) % 360)
    print("Current yaw", (current_yaw + 360) % 360)

    # estimated_magnitude = (desired_yaw - current_yaw + 180) % 360 - 180
    diff = (desired_yaw - current_yaw + 360 ) % 360
    estimated_magnitude = diff if diff <= 180 else diff - 360
    direction = 'turn_right_middle' if estimated_magnitude < 0 else 'turn_left_middle'
    while abs(diff) > degrees_per_rotate:
        # estimated_magnitude = (desired_yaw - current_yaw + 180) % 360 - 180
        diff = (desired_yaw - current_yaw + 360 ) % 360
        estimated_magnitude = diff if diff <= 180 else diff - 360
        direction = 'turn_right_middle' if estimated_magnitude < 0 else 'turn_left_middle'
        AGC.run_action(direction)
        prev_yaw = current_yaw
        _, current_yaw = read_pose(prev_yaw)
        print("Current Yaw", current_yaw)


def move_forward_x_seconds(seconds):
    """Move forward by issuing the forward motion command repeatedly for a specified duration."""
    start_time = time.time()
    while time.time() < start_time + seconds:
        AGC.run_action("go_forward_middle")

def angle_to_rotate(start_x, start_y, start_yaw, end_vertex):
    """Compute the angle (in degrees) the robot must rotate to face end_vertex."""
    dx = end_vertex.x - start_x
    dy = end_vertex.y - start_y
    target_angle_deg = math.degrees(math.atan2(dy, dx))
    delta = target_angle_deg - start_yaw
    # Normalize delta to [-180, 180]
    # delta = (delta + 180) % 360 - 180
    delta = (delta + 360) % 360
    return -delta

def navigate_to_last_dp(path, initial_x, initial_y, initial_yaw, robot_speed):
    """Follow the given path (a list of Vertex instances) to navigate toward the destination."""
    curr_x = initial_x
    curr_y = initial_y
    # curr_yaw = (initial_yaw + 180) % 360 - 180
    curr_yaw = (initial_yaw + 360) % 360
    for next_vertex in path:
        angle = angle_to_rotate(curr_x, curr_y, curr_yaw, next_vertex)
        #rotate_x_degrees(angle)
        rotate_to_desired_yaw(curr_yaw + angle, curr_yaw)
        distance = euclidean_dist(curr_x, curr_y, next_vertex.x, next_vertex.y)
        #time_to_move = distance / robot_speed
        # move_forward_x_seconds(time_to_move)
        move_forward_x_meters(distance)
        curr_x = next_vertex.x
        curr_y = next_vertex.y
        prev_yaw = curr_yaw
        curr_pos, curr_yaw = read_pose(prev_yaw)

def move_forward_x_meters(distance,
                          pos_tol: float = 0.20,     # stop when ≤ 20 cm from goal
                          angle_tol: float = 10.0):  # re-align when |Δθ| > 10 °
    """
    Drive roughly `distance` m in the direction `yaw`, correcting heading on the fly.

    Args
    ----
    distance  : float
        How far to travel in meters.
    pos_tol   : float
        Acceptable positional error (meters) from the goal.
    angle_tol : float
        If the instantaneous heading error exceeds this (deg), re-orient first.
    """
    # Compute goal position in world coordinates.
    pos_dict, yaw = read_pose(None)
    prev_yaw = yaw
    x, y = pos_dict["x"], pos_dict["y"]

    goal_x = x + distance * math.cos(math.radians(yaw))
    goal_y = y + distance * math.sin(math.radians(yaw))
    start_time = time.time()
    expected_duration = distance / robot_speed

    while time.time() < start_time + expected_duration + 30:
        # Current pose
        pos_dict, cur_yaw = read_pose(prev_yaw)
        prev_yaw = cur_yaw
        cur_x, cur_y = pos_dict["x"], pos_dict["y"]

        # How far still to go?
        remaining = euclidean_dist(cur_x, cur_y, goal_x, goal_y)
        if remaining <= pos_tol:
            break  # reached the goal (within tolerance)

        # Desired bearing toward goal
        desired_bearing = math.degrees(math.atan2(goal_y - cur_y, goal_x - cur_x))
        # heading_error = (desired_bearing - cur_yaw + 180) % 360 - 180  # normalize to [-180, 180]
        diff = (desired_bearing - cur_yaw + 360) % 360
        heading_error = diff if diff <= 180 else diff - 360

        # Re-orient if we’ve drifted too much.
        if abs(heading_error) > angle_tol:
            direction = 'turn_right_middle' if heading_error < 0 else 'turn_left_middle'
            steps = max(1, int(abs(heading_error) // degrees_per_rotate))
            for _ in range(steps):
                AGC.run_action(direction)
            continue  # read pose again after turning

        # Heading is good enough → take one forward step.
        AGC.run_action("go_forward_middle")

def read_pose_file():
    fname = "/home/pi/rtabmap_docker/maps/pose_data.txt"
    while True:
        with open(fname) as f:
            parts = f.read().strip().split()
            if len(parts) == 7:
                x, y, z, qw, qx, qy, qz = map(float, parts)

                siny_cosp = 2 * (qw * qz + qx * qy)
                cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
                yaw = math.degrees(np.arctan2(siny_cosp, cosy_cosp))

                if yaw < 0:
                    yaw += 360

                position = {"x": x, "y": y}
                return position, yaw

def read_pose(prev_yaw):
    fname = "/home/pi/rtabmap_docker/maps/pose_data.txt"

    position, curr_yaw = read_pose_file()
    while curr_yaw == prev_yaw:
        position, curr_yaw = read_pose_file()
        time.sleep(0.1)
    
    return position, curr_yaw

    # with open(fname) as f:
    #     x, y, z, qw, qx, qy, qz = map(float, f.read().strip().split())

    # nw = qw
    # nx = -qz
    # ny = qx
    # nz = qy

    # position = {"x": x, "y": -z}
    # yaw = -math.atan2(2.0 * (nw*nz + nx*ny), nw*nw + nx*nx - ny*ny - nz*nz) * 180.0 / math.pi
    # return position, yaw

def save_rgb(color_image, color_image_folder, color_image_name):
    dst_dir = os.path.abspath(color_image_folder)
    dst_path = os.path.join(dst_dir, f"{color_image_name}.png")
    cv2.imwrite(dst_path, color_image)
    return dst_path

def save_depth():
    depth_path = "/home/pi/rtabmap_docker/maps/depth.png"
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # uint16
    depth_m  = depth_raw.astype(np.float32) * 0.001  
    # depth_m = depth_raw.astype(np.float32) * 0.001
    return depth_m

def pgm_to_png(pgm_path,
               out_folder,
               out_name="grid_map.png"):
    """
    Convert a PGM occupancy-grid file (8-bit) to a PNG that any
    image viewer can open.

    Args
    ----
    pgm_path   : full path to the *.pgm  (e.g. /root/data/grid_map.pgm)
    out_folder : directory where the PNG should be written
    out_name   : file name of the PNG (default: grid_map.png)

    Returns
    -------
    str – the full path of the written PNG
    """
    if not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)

    png_path = os.path.join(out_folder, out_name)

    # PGM → PIL Image → save as PNG
    img = pil_safe_open(pgm_path)   # ensure 8-bit greyscale
    img.save(png_path, format="PNG")

    return png_path

def save_graph_on_grid_map(graph, overlay_folder, overlay_filename,
                           grid_pgm="/home/pi/rtabmap_docker/maps/grid_map.pgm",
                           grid_yaml="/home/pi/rtabmap_docker/maps/grid_map.yaml",
                           robot_pose=None, yaw_rad=None):
    """
    Produces something like the first screenshot:
      • grey = unknown, white = free, black = occupied
      • green dots  = vertices in `graph`
      • blue lines  = edges in `graph`
      • red arrow   = robot (optional)

    Result is written to  overlay_folder/overlay_filename.png
    and the full path is returned.
    """
    # ------------------------------------------------------------------ 0. I/O
    if not os.path.exists(overlay_folder):
        os.makedirs(overlay_folder, exist_ok=True)
    out_path = os.path.join(overlay_folder, overlay_filename + ".png")

    # ------------------------------- 1. read map + meta-data (origin/res)
    pgm  = pil_safe_open(grid_pgm)             # 8-bit greyscale
    w, h = pgm.size
    # map yaml: origin[x,y] & resolution
    origin = (0.0, 0.0); res = 0.05                         # defaults
    try:
        with open(grid_yaml) as f:
            for line in f:
                if line.startswith("origin:"):
                    origin = tuple(map(float, line.split("[")[1].split("]")[0].split(",")))[:2]
                elif line.startswith("resolution:"):
                    res = float(line.split(":")[1])
    except FileNotFoundError:
        pass                                                # fine – keep defaults

    # palette → RGB image (grey-white-black like RTAB-Map’s GUI)
    cmap = plt.get_cmap('gray')               # 0=black, 1=white
    pgm_arr = np.array(pgm) / 255.0
    rgb_arr = (cmap(pgm_arr)[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb_arr)
    draw = ImageDraw.Draw(img)

    # ------------------------------------ 2. helper: world → pixel
    def w2p(x, y):
        """world (m) → image pixel (int).  RTAB-Map's map has axis:
           +x right, +y up ; origin is lower-left corner of image"""
        px = int((x - origin[0]) / res)
        py = h - int((y - origin[1]) / res)                 # flip y
        return px, py

    # ----------------------------------- 3. draw edges + vertices
    drawn = set()
    for v in graph.adj_list:
        vx, vy = w2p(v.x, v.y)
        # vertex
        draw.ellipse((vx-3, vy-3, vx+3, vy+3), fill="lime")  # green
        # edges
        for nb, _ in graph.adj_list[v]:
            key = tuple(sorted([(v.x,v.y),(nb.x,nb.y)]))
            if key in drawn: continue
            drawn.add(key)
            x1, y1 = w2p(nb.x, nb.y)
            draw.line((vx, vy, x1, y1), fill="blue", width=2)

    # ----------------------------------- 4. robot pose (optional)
    if robot_pose:
        rx, ry = w2p(*robot_pose)
        draw.polygon([(rx,   ry-5),
                      (rx+5, ry+5),
                      (rx-5, ry+5)], fill="red")
        if yaw_rad is not None:
            dx = 20*math.cos(yaw_rad)
            dy = -20*math.sin(yaw_rad)   # screen y inverted
            draw.line((rx, ry, rx+dx, ry+dy), fill="red", width=3)

    img.save(out_path)
    return out_path

def read_static_map_meta(
        pgm_path="/home/pi/rtabmap_docker/maps/grid_map.pgm",
        yaml_path="/home/pi/rtabmap_docker/maps/grid_map.yaml"):
    """
    Returns (width, height, resolution, origin_x, origin_y)
    For maps saved with `map_saver`.  Orientation is always identity.
    """
    # --- 1. width & height --------------------------------------------------
    with open(pgm_path, "rb") as f:
        header = f.readline()        # P5
        while True:                  # skip comment lines
            line = f.readline()
            if not line.startswith(b"#"):
                break
        w, h = map(int, line.split())   # e.g. "640 480\n"

    # --- 2. resolution & origin --------------------------------------------
    with open(yaml_path) as f:
        y = yaml.safe_load(f)
    res       = float(y["resolution"])
    origin_xy = y["origin"][:2]        # [x, y, 0]

    pgm_img  = pil_safe_open(pgm_path)
    data     = np.asarray(pgm_img, dtype=np.uint8)
    if w == None or h == None:
        h, w = data.shape

    return w, h, res, origin_xy[0], origin_xy[1], data

def save_grid_map(out_folder, out_filename):
    src = "/home/pi/rtabmap_docker/maps/grid_map.png"
    dst_dir = os.path.abspath(out_folder)
    dst_path = os.path.join(dst_dir, f"{out_filename}.png")
    shutil.copy2(src, dst_path)

    return dst_path

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


def main():
    graph = Graph()
    decision_point_stack = []
    prev_vertex = None
    traversal_vertex = None
    counter = 0

    prev_position = None
    prev_traveled_yaw = None

    prev_selected_images = []
    prev_selected_segments = []
    prev_reasonings = []
    prev_occ_maps = []

    while True:
        # Get the current position and yaw from the observer.
        starting_position, starting_yaw = robot.pose

        if counter == 0:
            robot.graph_origin = robot.pose

        if traversal_vertex is not None:
            starting_vertex = traversal_vertex
        else:
            starting_vertex = Vertex(starting_position["x"], starting_position["y"])
            if prev_vertex is not None:
                graph.create_edge(prev_vertex, starting_vertex)

        print(f"Current position: {starting_position}, yaw: {starting_yaw:.2f} deg")

        # Rotate 360° and capture images at every 40° (i.e. every 4 rotations if each step is 10°).
        magnitude = 360
        direction = 'turn_right_middle' if magnitude < 0 else 'turn_left_middle'
        multi = abs(magnitude) // (picture_interval)
        print(f"Rotating and capturing images: {direction} {multi} times")

        starting_map = {}
        angle_deltas = []
        angles = []

        os.makedirs(f"aarush_VLM_{counter}_color", exist_ok=True)

        for i in range(int(multi)):
            # Capture color and depth images, and obtain the yaw at capture time.
            #time.sleep(1)
            pos_dict, raw_yaw = robot.pose

            rgb, depth = robot.capture_rgbd()
            image_path = save_rgb(rgb, f"aarush_VLM_{counter}_color", i)

            print(f"Captured image {i}: pos={pos_dict}, yaw={(raw_yaw + 360) % 360}")
            # delta = (raw_yaw - starting_yaw + 180) % 360 - 180
            delta = (raw_yaw - starting_yaw + 360) % 360
            # angles.append((raw_yaw + 180) % 360 - 180)
            angles.append((raw_yaw + 360) % 360)
            angle_deltas.append(delta)
            starting_map[i] = [image_path, (raw_yaw + 360) % 360]

            #rotate 40 degrees left
            robot.rotate_deg(-40)

        #time.sleep(1)
        #pgm_to_png("/home/pi/rtabmap_docker/maps/grid_map.pgm", "aarush_grid_maps", f"grid_map_{counter}.png")
        current_position, current_yaw = robot.pose

        # Optionally, save the occupancy grid image and overlay the graph.
        # save_graph_on_grid_map(graph, "aarush_map_graphs", 
        #                     f"new_aarush_map_with_graph_{counter}", 
        #                     grid_pgm="/home/pi/rtabmap_docker/maps/grid_map.pgm",
        #                     grid_yaml="/home/pi/rtabmap_docker/maps/grid_map.yaml",
        #                     robot_pose=(current_position["x"], current_position["y"]), yaw_rad=math.radians(current_yaw))

        occ_map_path = robot.overlay_graph_on_occupancy_map(graph, (starting_vertex.x, starting_vertex.y), current_yaw,
            out_folder="aarush_map_graphs", out_filename=f"new_aarush_map_with_graph_{counter}.png")
        
        #save_grid_map("aarush_grid_maps", f"grid_map_{counter}")

        # For example:
        # grid_path = observer.save_grid_map_as_image(f"aarush_grid_map_{counter}", save_folder="grid_maps")
        # overlay_path = observer.overlay_graph_on_occupancy_map(
        #     graph, (starting_vertex.x, starting_vertex.y), current_yaw,
        #     out_folder="map_graphs", out_filename=f"aarush_map_with_graph_{counter}.png"
        # )

        # Prepare the list of captured image paths for the VLM query.
        img_path_list = [starting_map[k][0] for k in starting_map.keys()]

        curr_graph = graph.visualize(robot_position=(starting_position["x"], starting_position["y"]),
                                     robot_orientation=math.radians(current_yaw))
        if not os.path.exists("aarush_graphs"):
            os.mkdir("aarush_graphs")

        graph_path = f"aarush_graphs/aarush_VLM_graph_{counter}.png"
        curr_graph.save(graph_path)

        map_graph_path = f"aarush_map_graphs/new_aarush_map_with_graph_{counter}.png"

        # Query the VLM/SAM system using the images, graph overlay, and angle deltas.
        #prev_payload, full_response = query_with_sequence(img_path_list, graph_path, angle_deltas)
        prev_vertex_angle = None
        if prev_vertex is not None:
            # vector FROM prev TO current (= starting_vertex **now**)
            # dx = starting_vertex.x - prev_vertex.x
            # dy = starting_vertex.y - prev_vertex.y

            # # atan2 gives angle CCW from +x (East) in radians
            # prev_vertex_angle = math.degrees(math.atan2(dy, dx)) % 360.0
            # print("Previous vertex", prev_vertex.x, prev_vertex.y)
            # print("Starting Vertex", starting_vertex.x, starting_vertex.y)
            # print("Previous Vertex angle", prev_vertex_angle)

            prev_vertex_angle = (prev_traveled_yaw + 180) % 360

        prev_payload, full_response = query_with_sequence(current_yaw, prev_vertex_angle, img_path_list, map_graph_path, angles, prev_selected_images, prev_selected_segments, prev_reasonings, prev_occ_maps, graph_to_string(graph))
        first_response_path = Path(f"aarush_VLM_{counter}_color/first_response.txt")

        with first_response_path.open("w", encoding="utf-8") as f:
            f.write(str(full_response))

        image_rankings = full_response.split("\n")[0].split(",")
        print(f"IMAGE RANKINGS: {image_rankings}")

        num_img_to_explore = int(image_rankings[0]) #* (picture_interval // degrees_per_rotate)

        if "not decision point" in full_response.split("\n")[1]:
            is_decision_point = False
        else:
            is_decision_point = True

        if int(image_rankings[0]) == -1:
            if not decision_point_stack:
                print("Navigation complete.")
                break
            traversal_vertex = decision_point_stack.pop()
            shortest_path = graph.dijkstra(starting_vertex, traversal_vertex)
            navigate_to_last_dp(shortest_path, starting_position["x"], starting_position["y"],
                                  current_yaw, robot_speed)
            counter += 1
            continue
        else:
            if is_decision_point:
                decision_point_stack.append(starting_vertex)

            desired_yaw = starting_map[num_img_to_explore][1]
            print(f"Desired Yaw: {desired_yaw}, Current Yaw: {current_yaw}")
            # magnitude = (desired_yaw - current_yaw + 180) % 360 - 180
            magnitude = (desired_yaw - current_yaw + 360) % 360
            # rotate_x_degrees(magnitude)
            #rotate_to_desired_yaw(desired_yaw, current_yaw)
            robot.rotate_deg(desired_yaw - current_yaw)

            # Capture a new image for segmentation.
            time.sleep(1)
            pos_dict, curr_yaw = robot.pose
            rgb, depth = robot.capture_rgbd()
            selected_img_path = save_rgb(rgb, f"aarush_VLM_{counter}_color", "selected_img")

            mask_paths, combined_mask_path = run_sam_inference_and_download_masks(
                selected_img_path,
                f"aarush_VLM_{counter}_color/segments",
                ""  # Replace with your actual token
            )

            if mask_paths != []:
                segment_headings = create_annotated_image(
                    selected_img_path,
                    mask_paths,
                    f"aarush_VLM_{counter}_color/segments/annotated_image.png"
                )
                best_segments_with_reasoning = get_best_segment(
                    prev_payload,
                    full_response,
                    selected_img_path,
                    f"aarush_VLM_{counter}_color/segments/annotated_image.png",
                    mask_paths
                )

                print(best_segments_with_reasoning)
                _, depth_img = robot.capture_rgbd()

                second_response_path = Path(f"aarush_VLM_{counter}_color/second_response.txt")
            
                with second_response_path.open("w", encoding="utf-8") as f:
                    f.write(str(best_segments_with_reasoning))

                segment_rankings = best_segments_with_reasoning.split("\n")[0].split(",")
                selected_segment = int(segment_rankings[0])

                for i, curr_segment in enumerate(segment_rankings):
                    curr_segment = int(curr_segment)

                    chosen_mask = cv2.imread(f"aarush_VLM_{counter}_color/segments/mask_{curr_segment}.png", cv2.IMREAD_GRAYSCALE)

                    if not isinstance(chosen_mask, np.ndarray):
                        print(f"Error: Failed to load mask for best segment {curr_segment}. Got {chosen_mask} (type {type(chosen_mask)})")
                        continue  # You can choose to handle this error differently if needed.

                    mask_bool = chosen_mask > 128
                    seg_depth_pixels = depth_img[mask_bool]
                    seg_depth_pixels = seg_depth_pixels[seg_depth_pixels > 0]
                    if len(seg_depth_pixels > 0):
                        selected_segment = curr_segment
                        selected_segment_index = i
                        break
                    else:
                        print("Segment mask depth pixels are empty, trying next segment")
                

                angle_to_rotate_seg = segment_headings[selected_segment]
                #distance_to_segment = float(np.median(seg_depth_pixels))
                distance_to_segment = float(np.median(seg_depth_pixels)) - 0.5
                print("Distance to travel to segment", distance_to_segment)


                #time_to_move = distance_to_segment / robot_speed
                #rotate_x_degrees(angle_to_rotate_seg)
                current_position, current_yaw = robot.pose


                robot.rotate_deg(angle_to_rotate_seg)
                # move_forward_x_seconds(time_to_move)
                robot.drive_forward(distance_to_segment)
                # move_forward_x_meters(distance_to_segment)
                prev_vertex = starting_vertex
                traversal_vertex = None
                counter += 1

                prev_selected_images.append(selected_img_path)
                prev_selected_segments.append(selected_segment)

                reasoning = (
                    f"{best_segments_with_reasoning.splitlines()[i + 1]} "
                    f"Traveled {distance_to_segment} meters towards "
                    f"{current_yaw + angle_to_rotate_seg} degrees"
                )
                prev_reasonings.append(reasoning)

                prev_traveled_yaw = (current_yaw + angle_to_rotate_seg + 360) % 360

                with second_response_path.open("a", encoding="utf-8") as f:
                    f.write(reasoning)
                
                prev_occ_maps.append(occ_map_path)

        # Pause briefly between iterations.
        #time.sleep(1.5)
        #Create a vertex at the current position and add it to the Graph

        #Spin in a circle, at each picture_interval, sleep for 2 seconds save the current RGB image into curr_rgb, save current pose into curr_pose dictionary

            #Compare current pose to previous pose, if same then keep re-reading pose_data.txt until it is different

            #Save rgb image to aarush_VLM_{counter}_color folder, save image path and current yaw to dictionary

        #Query VLM, get the best image and direction

        #Rotate to the chosen yaw, take the current depth image and store in curr_depth, take current rgb image and store in selected_rgb, save rgb image to folder

        #Query SAM, save all masks/labeled segmentation, annotate segmentation image

        #Query VLM again, get best segment and best mask

        #Apply segment mask to curr_depth, calculate distance to travel by averaging the depth values

        #Travel distance





if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    main()