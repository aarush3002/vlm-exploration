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

# ---- Google Gemini ---------------------------------------------------------
import google.generativeai as genai
from google.generativeai import types as gtypes 

gen_error = genai

genai.configure(api_key=os.getenv("AIzaSyBbvCyGbZOXTLHfx4itJ1WSWGUHfmuCzcY"))

VISION_MODEL = "gemini-2.5-pro"        # or flash‑vision for lower cost
MAX_TOKENS   = 20000


def make_image_part(path: str) -> Dict[str, Any]:
    """Return {"mime_type": ..., "data": ...} for an image file."""
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as f:
        return {"mime_type": mime, "data": f.read()}



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
    # delta = (delta + 180) % 360 - 180
    delta = (delta + 360) % 360
    return -delta

def navigate_to_last_dp(path, initial_x, initial_y, initial_yaw):
    """Follow the given path (a list of Vertex instances) to navigate toward the destination."""
    curr_x = initial_x
    curr_y = initial_y
    # curr_yaw = (initial_yaw + 180) % 360 - 180
    curr_yaw = (initial_yaw + 360) % 360
    for next_vertex in path:
        angle = angle_to_rotate(curr_x, curr_y, curr_yaw, next_vertex)
        #rotate_x_degrees(angle)
        # rotate_to_desired_yaw(curr_yaw + angle, curr_yaw)
        robot.rotate_deg(angle)
        distance = euclidean_dist(curr_x, curr_y, next_vertex.x, next_vertex.y)
        #time_to_move = distance / robot_speed
        # move_forward_x_seconds(time_to_move)
        #move_forward_x_meters(distance)
        robot.drive_forward(distance)
        curr_x = next_vertex.x
        curr_y = next_vertex.y
        curr_pos, curr_yaw = robot.pose

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

# ---------------------------------------------------------------------------
# Main control loop
# ---------------------------------------------------------------------------

def main():

    robot.set_ang_vel(-9000)
    return

    graph = Graph()
    decision_point_stack = []
    prev_vertex          = None
    traversal_vertex     = None

    counter = 0
    prev_traveled_yaw = None

    while True:
        # 1. Current pose ----------------------------------------------------
        starting_position, starting_yaw = robot.pose
        if counter == 0:
            robot.graph_origin = robot.pose

        starting_vertex = traversal_vertex or Vertex(starting_position["x"], starting_position["y"])
        if prev_vertex is not None and traversal_vertex is None:
            graph.create_edge(prev_vertex, starting_vertex)

        print(f"Pos: {starting_position} | Yaw: {starting_yaw:.1f}°")

        # 2. 360° sweep ------------------------------------------------------
        images_dir     = f"aarush_VLM_{counter}_color"
        os.makedirs(images_dir, exist_ok=True)
        multi = 360 // picture_interval

        starting_map = {}
        angles       = []

        for i in range(multi):
            pos_dict, raw_yaw = robot.pose
            print(f"Pos: {pos_dict} | Yaw: {raw_yaw:.1f}°")
            rgb, _   = robot.capture_rgbd()
            img_path = save_rgb(rgb, images_dir, i)

            heading  = (raw_yaw + 360) % 360
            angles.append(heading)
            starting_map[i] = [img_path, heading]

            robot.rotate_deg(-picture_interval)

        # 3. Save overlay map ------------------------------------------------
        occ_map_path = robot.overlay_graph_on_occupancy_map(
            graph, (starting_vertex.x, starting_vertex.y), starting_yaw,
            out_folder="aarush_map_graphs",
            out_filename=f"map_graph_{counter}.png")

        # 4. Gemini: choose best direction ----------------------------------
        if counter == 0:
            first_prompt = f"""
            You are a hexapod robot that has the skills "rotate x degrees" and "walk forward for x seconds". Your main goal is to explore
            new, accessible areas in an unknown environment as quickly as possible. You are given {len(angles)} images of your surroundings, and 
            your task is to select one of the images to walk towards in order to achieve your main goal. You are additionally given
            a 2D, top-down occupancy map of your surroundings, where black pixels represent occupied space, white pixels represent free
            space, and gray pixels represent unknown space (may not be accurate). There is also a scale in meters (red).

            Your answer should be a comma-separated ranking (no spaces, with nothing else) from best to worst of the images based on which images 
            would be best to travel towards in order to achieve your main goal.

            On the next line after your comma-separated rankings, if you believe that there is more than one reasonable (i.e. no 
            immediate walls or obstacles), unexplored direction that the robot could potentially travel towards, write 
            "decision point" (no quotes). Otherwise write "not decision point" (no quotes). 

            Starting on a new line after this, please explain your reasoning as to how you ranked the images.

            """
        else:
            string_graph = graph_to_string(graph)

            first_prompt = f"""
            You are a hexapod robot that has the skills "rotate x degrees" and "walk forward for x seconds". Your main goal is to explore
            new, accessible areas in an unknown environment as quickly as possible. You are given {len(angles)} images of your surroundings, and 
            your task is to select one of the images to walk towards in order to achieve your main goal. You are additionally given
            a 2D, top-down occupancy map of your surroundings, where black pixels represent occupied space, white pixels represent free
            space, and gray pixels represent unknown space (may not be accurate). There is also a scale in meters (red). Do not go towards 
            previously traversed areas unless necessary to continue exploration.

            Here is a string representation of the edge distances in your traversal graph so far:

            {string_graph}

            Your answer should be a comma-separated ranking (no spaces, with nothing else) from best to worst of the 
            images based on which images would be best to travel towards in order to achieve your main goal. If you believe that 
            absolutely none of the directions are suitable to travel towards, your ranking should be -1 and you will be navigated to the most
            recent decision point. If you are certain that exploration is complete (i.e. if there are no more unvisited areas/unexplored frontiers), 
            then instead of your ranking, just put "exploration complete".

            On the next line after your comma-separated rankings, if you believe that there is more than one reasonable (i.e. no 
            immediate walls or obstacles), unexplored direction that the robot could potentially travel towards, write 
            "decision point" (no quotes). Otherwise write "not decision point" (no quotes). 

            Starting on a new line after this, please explain your reasoning as to how you ranked the images.
            """
        parts: List[types.Part | str] = [
            first_prompt
        ]

        # attach camera frames
        for idx, heading in enumerate(angles):
            parts.append(make_image_part(starting_map[idx][0]))
            parts.append(f"Image {idx} at {heading}°.")

        # map + graph overlay
        parts.append(make_image_part(occ_map_path))
        parts.append(f"Current yaw = {starting_yaw:.1f}°")

        #full_response = safe_send(chat, parts)
        full_response = safe_send(parts)
        first_resp_path = Path(f"{images_dir}/first_response.txt")
        first_resp_path.write_text(full_response)

        image_rankings = full_response.split("\n")[0].split(",")
        if image_rankings[0] == "exploration complete":
            print("Exploration complete.")
            break
        if int(image_rankings[0]) == -1:
            # dead‑end: pop last decision point if any
            if not decision_point_stack:
                print("Exploration complete.")
                break
            traversal_vertex = decision_point_stack.pop()
            # (navigation to last decision point omitted for brevity)
            shortest_path = graph.dijkstra(starting_vertex, traversal_vertex)
            curr_pose, curr_yaw = robot.pose
            navigate_to_last_dp(shortest_path, curr_pose["x"], curr_pose["y"], curr_yaw)
            counter += 1
            continue

        if full_response.split("\n")[1] == "decision point":
            decision_point_stack.append(starting_vertex)

        # ------------------------------------------------------------------
        chosen_idx = int(image_rankings[0])
        desired_yaw = starting_map[chosen_idx][1]
        print(f"Chosen bearing {desired_yaw}° (image {chosen_idx})")
        robot.rotate_deg(desired_yaw - starting_yaw)

        # 5. Capture new frame for segmentation -----------------------------
        time.sleep(1)
        pos_dict, curr_yaw = robot.pose
        rgb, depth         = robot.capture_rgbd()
        selected_img_path  = save_rgb(rgb, images_dir, "selected_img")
        _, selected_depth_img_path = save_depth(depth, images_dir, "selected_depth_img")

        mask_paths, _ = run_sam_inference_and_download_masks(
            selected_img_path,
            f"{images_dir}/segments",
            "") #REPLICATE API TOKEN
        
        # ----------------------------------------------------------- depth‑per‑segment
        segment_depths = {}          # {label: median_in_metres or None}

        for m_path in mask_paths:
            # mask filenames are "mask_<label>.png"
            label = int(Path(m_path).stem.split("_")[1])
            mask  = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            pixels = depth[mask > 128]
            pixels = pixels[pixels > 0]         # drop invalid 0‑depth
            segment_depths[label] = float(np.median(pixels)) if pixels.size else None

        # Human‑readable block we’ll splice into the second prompt
        depth_lines = "\n".join(
            f"• segment {lbl}: {d:.2f} m" if d is not None else f"• segment {lbl}: unknown"
            for lbl, d in sorted(segment_depths.items())
        )


        print(depth_lines)
        
        if mask_paths:
            segment_headings = create_annotated_image(
                selected_img_path,
                mask_paths,
                f"{images_dir}/segments/annotated_image.png")

            # Gemini follow‑up: best segment --------------------------------
            '''
            The next line directly after the comma-separated ranking of the segments should be a comma-separated list of distances
            you would like to travel towards each ranked segment, just in case you do not want to go all the way to that segment.
            It should be in corresponding order to the segment rankings. Distances should be in meters to 2 decimal places of precision.
            '''
            second_prompt = f"""
            Based on your previous selection, you have rotated towards the chosen direction and captured a new raw image of 
            your current heading. You have also generated an annotated image with labeled segments. The first image provided
            to you is the raw RGB image of your current heading. This is followed by the annotated image. Finally, you are also
            given the depth image of your current heading.

            Below are approximate straight‑line distances (median depth) to each segment:

            {depth_lines}

            Your goal is to first consider the possible segments that would be the best to approach next in order to achieve your
            goal of efficiently exploring new areas in the environment, and discard any unreasonable segments. 
            Please make sure to use the previous prompt/response in your decision, especially the traversal graph - 
            you want to avoid repeated traversals. 
            
            For the most part, you should try to avoid segments on the floor or ceiling since you will be calculating the 
            distance to the segment based on the given depth image.
            
            Also, deprioritize segments for which there are objects blocking the straight-line path to that segment - 
            for example, if the segment is a door with a couch right in front of it, then that segment should be deprioritized 
            since you cannot go through the couch.

            If there is any segment which is less than or equal to 0.2 meters away, you will need to rotate away from that segment
            and walk away from that obstacle before continuing your main objective. 
            
            Your answer should be a comma-separated ranking (with nothing else) of the numerical labels of the segments 
            that are still in consideration with NOTHING else (from best to worst).

            The next line directly after the comma-separated ranking of the segments should be a comma-separated list of distances
            you would like to travel towards each ranked segment, just in case you do not want to go all the way to that segment.
            It should be in corresponding order to the segment rankings. Distances should be in meters to 2 decimal places of precision.

            The next line(s) directly after this should be the actions you would like to execute for 
            each segment in your ranking, for example if the ranking was 5,3,2: 

            Rotate towards segment 5, walk along the hallway towards the living room.
            Rotate towards segment 3, walk towards the open doorway to the garage.
            Rotate towards segment 2, walk to the back wall of the bedroom.
            
            Starting on a new line afterwards, please explain your reasoning for your choices (on a new line). 
            """
            seg_parts: List[types.Part | str] = [
                second_prompt
            ]
            seg_parts.append(make_image_part(selected_img_path))
            seg_parts.append("Raw RGB frame.")
            seg_parts.append(make_image_part(f"{images_dir}/segments/annotated_image.png"))
            seg_parts.append("Annotated segments.")
            seg_parts.append(make_image_part(selected_depth_img_path))
            seg_parts.append("Depth preview (8 bit).")

            #seg_response = safe_send(chat, seg_parts)
            seg_response = safe_send(seg_parts)
            second_resp_path = Path(f"{images_dir}/second_response.txt")
            second_resp_path.write_text(seg_response)

            segment_rankings = seg_response.split("\n")[0].split(",")
            selected_segment = int(segment_rankings[0])
            # if selected_segment == -1:
            #     curr_pose, curr_yaw = robot.pose
            #     if not decision_point_stack:
            #         print("Exploration complete.")
            #         break
            #     traversal_vertex = decision_point_stack.pop()
            #     shortest_path = graph.dijkstra(starting_vertex, traversal_vertex)
            #     navigate_to_last_dp(shortest_path, curr_pose["x"], curr_pose["y"], curr_yaw)
            #     counter += 1
            #     continue

            # Depth‑based distance calc (same as original)
            chosen_mask = cv2.imread(
                f"{images_dir}/segments/mask_{selected_segment}.png", cv2.IMREAD_GRAYSCALE)
            depth_pixels = depth[chosen_mask > 128]
            depth_pixels = depth_pixels[depth_pixels > 0]
            distance = float(np.median(depth_pixels)) if len(depth_pixels) else 0.75

            print(f"Rotate to segment {selected_segment}")
            angle_to_rotate_seg = segment_headings[selected_segment]
            robot.rotate_deg(angle_to_rotate_seg)

            segment_distances = seg_response.split("\n")[1].split(",")
            selected_segment_distance = float(segment_distances[0])

            #print(f"→ Segment {selected_segment}, travel {distance:.2f} m")
            print(f"→ Segment {selected_segment}, VLM travel {selected_segment_distance:.2f} m")

            #robot.drive_forward(distance)
            print(selected_segment_distance)
            robot.drive_forward(selected_segment_distance)

            with second_resp_path.open("a", encoding="utf‑8") as f:   # "a" = append
                f.write("\n")
                f.write(depth_lines)
                f.write("\n")
                f.write(f"Chose Segment {selected_segment}, rotated {angle_to_rotate_seg} degrees, final heading {robot.pose[1]} degrees, travel {selected_segment_distance:.2f} m")
                #f.write(f"Chose Segment {selected_segment}, rotated {angle_to_rotate_seg} degrees, final heading {robot.pose[1]} degrees, travel {distance:.2f} m")

            # Graph bookkeeping -------------------------------------------
            prev_vertex      = starting_vertex
            traversal_vertex = None
            counter         += 1

        else:
            print("SAM produced no usable masks – skipping movement.")
            counter += 1


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    main()
