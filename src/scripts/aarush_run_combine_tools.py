#api_key = "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"
api_key = "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"

import base64
import requests
import time

from typing import Tuple, List, Dict

def safe_chat_completion(
    session: requests.Session,
    payload: Dict,
    headers: Dict,
    *,
    max_retries: int = 5,
    wait_s: int = 60
) -> Dict:
    """POST to /chat/completions with automatic retry on rate‑limit or network errors."""
    url = "https://api.openai.com/v1/chat/completions"

    for attempt in range(1, max_retries + 1):
        try:
            resp = session.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()                       # HTTP‑level problems
            data = resp.json()

            # ---------- OpenAI JSON‑level problems ----------
            if "error" in data:
                if data["error"].get("code") == "rate_limit_exceeded":
                    print(f"[{attempt}/{max_retries}]  Rate‑limit hit – waiting {wait_s}s…")
                    time.sleep(wait_s)
                    continue
                raise RuntimeError(f"OpenAI error: {data}")

            if "choices" not in data or not data["choices"]:
                raise RuntimeError(f"Unexpected response structure: {data}")

            return data                                    # Success!

        except requests.exceptions.RequestException as e:
            # Covers timeouts, connection resets, etc.
            print(f"[{attempt}/{max_retries}]  Request failed: {e} – retrying in {wait_s}s…")
            time.sleep(wait_s)

    raise RuntimeError("Max retries exceeded when calling OpenAI /chat/completions")

session = requests.Session()
'''
    You are a hexapod robot that has the skills "rotate x degrees" and "walk forward for x seconds". 
    You are placed in an unknown, indoor environment. Your goal is to travel around the environment and map it as fast 
    and efficiently as possible, as well as maximize full coverage of the area. You may want to explore open doorways,
    hallways that might lead to new areas, etc. You can assume that anything in your camera image/line 
    of sight will update a live occupancy map via depth information. You have just rotated 360 degrees clockwise, 
    taking a picture every {360/len(images)} degrees, with the last image at {(360/len(images)) * (len(images) - 1)}. 
    You are given the {len(images)} images provided, the current occupancy map mentioned previously (white pixels are free
    space, gray pixels are unexplored space, and black pixels are occupied space), 
    along with a graph representation, in which the vertices represent places where you previously made a decision, 
    and it also shows your location as well as the angle you are currently facing (denoted by the arrow on the graph). 
    First, determine the images that could possibly lead to a new area and discard the rest. Then, your answer should be 
    a comma-separated ranking among the images left in consideration. 
    
    For example, given some images 1-9, you may decide that image 9 will lead to the same area as image 1,
    but image 1 shows a straighter shot towards the correct path, so we discard image 9. Images 2, 5, and 7
    might be images of walls or unavoidable obstacles, or they might lead to areas that we've already explored
    based on the graph traversal map, so we discard those as well. Then, you might rank the remaining
    images as "3,1,4,6,8" depending on their individual characteristics, and that should be your answer that you return 
    (with absolutely nothing else).
    
    If you are quite sure that none of the images will lead to a new area/are easily traversable, your answer should just be "-1"
    (with absolutely nothing else).
'''

'''
You can assume that you will be walking straight towards the center of the images you choose. You should try to make sure that the images 
    in consideration have a clear path in front of you, without many obstacles in the way. Consider future decisions as well.
'''

'''
You are a hexapod robot that has the skills "rotate x degrees" and "walk forward for x seconds". Your main goal is to explore
new areas in an unknown environment as quickly as possible. You are given {len(images)} images of your surroundings 
'''

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        file_bytes = image_file.read()
        return base64.b64encode(file_bytes).decode('utf-8')
    
def query_with_sequence(current_yaw, prev_vertex_angle, images, map_graph_path, angles, prev_selected_images = [], prev_selected_segments = [], prev_reasonings = [], prev_occ_maps = [], string_graph=""):
    """
    Sends a query to the VLM/LLM API with a sequence of images, each accompanied by a text
    description of its rotation angle, and a final map graph image.
    
    Args:
        images (list of str): List of file paths for images captured during rotation.
        map_graph_path (str): File path to the image of the current traversal graph.
        angles (list): List of rotation angles (in degrees) corresponding to each image.
    
    Returns:
        tuple: (payload, model_response) where payload is the JSON payload sent and
               model_response is the API's textual response.
    """
    prompt = None
    if prev_selected_images == []:
        prompt = f"""
        You are a hexapod robot that has the skills "rotate x degrees" and "walk forward for x seconds". Your main goal is to explore
        new areas in an unknown environment as quickly as possible. You are given {len(images)} images of your surroundings, and 
        your task is to select one of the images to walk towards in order to achieve your main goal. You are additionally given
        a 2D, top-down occupancy map of your surroundings, where black pixels represent occupied space, white pixels represent free
        space, and gray pixels represent unknown space. You should prioritize exploring nearby "frontiers", or borders between 
        gray and white pixels in the occupancy map. 

        Your answer should be a comma-separated ranking (no spaces, with nothing else) from best to worst of the images based on which images 
        would be best to travel towards in order to achieve your main goal.

        On the next line after your comma-separated rankings, if you believe that there is more than one reasonable (i.e. no 
        immediate walls or obstacles), unexplored direction that the robot could potentially travel towards, write 
        "decision point" (no quotes). Otherwise write "not decision point" (no quotes). 

        Starting on a new line after this, please explain your reasoning as to how you ranked the images.

        """
        # prompt = f"""
        # You are a hexapod robot that has the skills "rotate x degrees" and "walk forward for x seconds". Your goal is to travel 
        # around the environment and map it as fast and efficiently as possible, as well as maximize full coverage of the area. You
        # can assume that as you travel around, a live occupancy map will be updated simultaneously. You have just taken some 
        # pictures of your surroundings (the first {len(images)} images provided to you). Next, you are given your traversal path so
        # far, overlaid onto the current occupancy map. Using the occupancy map/traversal, as well as the color images of your
        # surroundings, your answer should be a comma-separated ranking (no spaces) of the images based on which images would be 
        # best to travel towards (deprioritize directions whose images have lots of untraversable obstacles, or immediate walls/objects). 
        # You are currently facing {current_yaw} degrees on the map. On the occupancy map, west is 0 degrees, south is 90 degrees, 
        # east is 180/-180 degrees, and north is -90 degrees. This is the first decision you have made so far.

        # Sometimes, the camera captures images incorrectly. If images 0 and 1 are the same image, then you can assume that image 1
        # is supposed to be in between images 0 and 2. Likewise, if the angles for images 0 and 1 are within 1 degree, then the 
        # angle for image 1 should be the average of image 0's angle and image 2's angle. This applies to any adjacent images which 
        # are the same/whose angles are similar, not just 0 and 1.

        # On the next line after your comma-separated rankings, if you believe that there is more than one reasonable,
        # unexplored direction that the robot could potentially travel towards, write "decision point" (no quotes). Otherwise write 
        # "not decision point" (no quotes). 

        # Starting on a new line after this, please explain your reasoning as to how you ranked the images.
        # """
        # prompt = f"""
        # You are a hexapod robot that has the skills "rotate x degrees" and "walk forward for x seconds".
        # You are placed in an unknown, indoor environment. Your goal is to travel around the environment and map it as fast
        # and efficiently as possible, as well as maximize full coverage of the area. You may want to explore open doorways,
        # hallways that might lead to new areas, open areas, etc. Be aware that you cannot climb stairs, but going near them is fine.
        # Also, avoid furniture that your legs could get caught on, such as chairs or table legs.
        # You can assume that anything in your camera image/line of sight will update a live occupancy map via depth information. 
        # You have just rotated counter-clockwise, taking {len(images)} pictures along the way. You will decide which image 
        # to walk towards.

        # You are given the {len(images)} images provided. The last image is your traversal path so far, 
        # overlayed on the current occupancy map. You should aim to explore areas that are uncovered by the traversal graph so far, and make sure to fully 
        # explore a given path before back tracking and exploring the next one - for example, if you see a long hallway and you 
        # see that the traversal path so far has been going down that hallway, try continuing until you reach the end and have to 
        # go somewhere else. Additionally, you should target to explore frontiers, or boundaries between free and unknown space on
        # the occupancy map. 

        # First, determine the images/directions that could possibly lead to a new area/frontier and discard the rest. Please use
        # the traversal path to aid in your decision — you should try to avoid repeated traversals whenever possible.  You can assume 
        # that any people or animals in the chosen camera image will move out of your way.

        # Then, the very first line of your response should be a comma-separated ranking among the images left in consideration, 
        # from best to worst. If you believe that adjacent images are similar but one has less obstacles in the immediate vicinity 
        # in front of you, please rank the less-cluttered image before the adjacent one.

        # If you are quite sure that none of the images will lead to a new area or are easily traversable (i.e. you are in a dead end),
        # your answer should just be "-1".

        # Starting on a new line after your comma-separated rankings, please explain your reasoning as to how you ranked the images.
        # """
    else:
        print(string_graph)

        print((prev_vertex_angle - 45 + 360) % 360)

        print((prev_vertex_angle + 45 + 360) % 360)

        prompt = f"""
        You are a hexapod robot that has the skills "rotate x degrees" and "walk forward for x seconds". Your main goal is to explore
        new areas in an unknown environment as quickly as possible. You are given {len(images)} images of your surroundings, and 
        your task is to select one of the images to walk towards in order to achieve your main goal. You are additionally given
        a 2D, top-down occupancy map of your surroundings, where black pixels represent occupied space, white pixels represent free
        space, and gray pixels represent unknown space. You should prioritize exploring nearby "frontiers", or borders between 
        gray and white pixels in the occupancy map. Do not repeat previously traversed areas unless absolutely necessary. You
        should deprioritize images in the range of {(prev_vertex_angle - 45 + 360) % 360} degrees to 
        {(prev_vertex_angle + 45 + 360) % 360} (and around this range), since this is where you just came from. 
        Only select these images unless absolutely necessary to continue exploration.

        Here is a string representation of the current traversal graph, with each line representing one vertex and the vertices 
        that it is connected to along with the distance to each vertex:

        {string_graph}

        Your answer should be a comma-separated ranking (no spaces, with nothing else) from best to worst of the 
        images based on which images would be best to travel towards in order to achieve your main goal.

        On the next line after your comma-separated rankings, if you believe that there is more than one reasonable (i.e. no 
        immediate walls or obstacles), unexplored direction that the robot could potentially travel towards, write 
        "decision point" (no quotes). Otherwise write "not decision point" (no quotes). 

        Starting on a new line after this, please explain your reasoning as to how you ranked the images.
        """

        # prompt = f"""
        # You are a hexapod robot that has the skills "rotate x degrees" and "walk forward for x seconds". Your goal is to travel 
        # around the environment and map it as fast and efficiently as possible, as well as maximize full coverage of the area. You
        # can assume that as you travel around, a live occupancy map will be updated simultaneously. You have just taken some 
        # pictures of your surroundings (the first {len(images)} images provided to you). Next, you are given your traversal path so
        # far, overlaid onto the current occupancy map. 

        # Since you have already been traversing the area, the subsequent images are the occupancy maps from previous steps, to aid 
        # in your decision. You should deprioritize backtracking to previously visited locations on the traversal path. You are 
        # currently facing {current_yaw} degrees on the map, and this general direction should be given slightly 
        # extra priority since you can continue exploring in that direction until you must change course. You just came 
        # from the direction of {prev_vertex_angle} degrees, so that general direction should be deprioritized since you just 
        # explored that direction. Angles that go along previously traversed paths on the occupancy path (blue lines)
        # should also be deprioritized. On the occupancy map, west is 0 degrees, south is 90 degrees, east is 180/-180 degrees, and 
        # north is -90 degrees.

        # After the previous occupancy maps, you are also given the selected images (annotated with segment labels) from the previous
        # steps.

        # Sometimes, the camera captures images incorrectly. If images 0 and 1 are the same image, then you can assume that image 1
        # is supposed to be in between images 0 and 2. Likewise, if the angles for images 0 and 1 are within 1 degree, then the 
        # angle for image 1 should be the average of image 0's angle and image 2's angle. This applies to any adjacent images which 
        # look the same/whose angles are similar, not just 0 and 1.
         
        # Using primarily the current occupancy map, traversal path, and previous traversal paths along with the color images to
        # aid your decision (deprioritize directions whose images have lots of untraversable obstacles, or immediate walls/objects). Your
        # answer should be a comma-separated ranking of the current images based on which images would be best to travel towards.
        
        # If you believe that you are in a dead end and absolutely must backtrack to continue growing the occupancy map, your
        # answer should just be "-1".

        # On the next line after your comma-separated rankings, if you believe that there is more than one reasonable,
        # unexplored direction that the robot could potentially travel towards, write "decision point" (no quotes). Otherwise write 
        # "not decision point" (no quotes). 

        # Starting on a new line after this, please explain your reasoning as to how you ranked the images.
        # """
        
        # prompt = f"""
        # You are a hexapod robot that has the skills "rotate x degrees" and "walk forward for x seconds".
        # You are placed in an unknown, indoor environment. Your goal is to travel around the environment and map it as fast
        # and efficiently as possible, as well as maximize full coverage of the area. You may want to explore open doorways,
        # hallways that might lead to new areas, open areas, etc. Be aware that you cannot climb stairs, but going near them is fine.
        # Also, avoid furniture that your legs could get caught on, such as chairs or table legs.
        # You can assume that anything in your camera image/line of sight will update a live occupancy map via depth information. 
        # You have just rotated 360 degrees counter-clockwise, taking {len(images)} pictures along the way. You will decide which image 
        # to walk towards.

        # You are given the {len(images)} images provided. The second to last image is your traversal path so far, 
        # overlayed on the current occupancy map. The last {len(prev_selected_images)} images are the images you selected in the 
        # previous iterations. In general, you should aim to do a depth-first search of the environment (i.e. follow the current path
        # you are on until you are right in front of a wall or obstacle). Deprioritize directions that go the opposite way as the 
        # last iteration - if the last iteration was going towards 90 degrees, you should not go towards the general direction of -90
        # degrees.

        # First, determine the images/directions that continue searching down a branch of the area, or could possibly lead to a 
        # new area/frontier and discard the rest. Please use the traversal path to aid in your decision — you should try to avoid backtracking 
        # whenever possible, although you will need to backtrack at dead-ends.  You can assume that any people or animals in 
        # the chosen camera image will move out of your way.

        # Then, the very first line of your response should be a comma-separated ranking among the images left in consideration, 
        # from best to worst. If you believe that adjacent images are similar but one has less obstacles in the immediate vicinity 
        # in front of you, please rank the less-cluttered image before the adjacent one.

        # If you are quite sure that none of the images will lead to a new area or are easily traversable (i.e. you are in a dead end),
        # your answer should just be "-1".

        # Starting on a new line after your comma-separated rankings, please explain your reasoning as to how you ranked the images.
        # """


    encoded_images = [encode_image(img) for img in images]
    encoded_map_graph = encode_image(map_graph_path)

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "max_tokens": 600
    }

    for idx, image in enumerate(encoded_images):
        payload["messages"][0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"}
        })
        payload["messages"][0]["content"].append({
            "type": "text",
            "text": f"This is image number {idx}. It was captured when you were facing towards {angles[idx]} degrees."
        })
    
    payload["messages"][0]["content"].append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{encoded_map_graph}"}
    })
    payload["messages"][0]["content"].append({
        "type": "text",
        "text": f"""This is the traversal graph so far, with your current location indicated by a red node. The green nodes 
        represent locations at which you were previously queried for the same purpose. Your current heading is indicated by the 
        orange arrow pointing from your current location, which is currently {current_yaw} degrees. West is 0 degrees, South is 90 
        degrees, East is 180 degrees, and North is 270 degrees. Counterclockwise rotations are positive degrees. The 
        traversal graph is overlayed onto the current occupancy map, where black pixels represent occupied space, gray is unknown 
        space, and white is free space. The nodes are labeled in chronological order."""
    })

    if prev_occ_maps != []:
        encoded_occ_maps = [encode_image(img) for img in prev_occ_maps]
        for i in range(len(encoded_occ_maps)):
            payload["messages"][0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_occ_maps[i]}"}
            })
            payload["messages"][0]["content"].append({
                "type": "text",
                "text": f"""This is the occupancy map with your traversal at step {i}. """
            })

    if prev_selected_images != []:
        encoded_prev_selected_images = [encode_image(img) for img in prev_selected_images]
        for i in range(len(prev_selected_images)):
            payload["messages"][0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_prev_selected_images[i]}"}
            })
            payload["messages"][0]["content"].append({
                "type": "text",
                "text": f"""This is the image you selected in step {i}, and you further selected segment 
                {prev_selected_segments[i]} to walk towards. You described your action as follows: {prev_reasonings[i]} """
            })

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    #response = session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    response = safe_chat_completion(session, payload, headers)
    print(response["choices"][0]["message"]["content"])
    return payload, response["choices"][0]["message"]["content"]

def get_best_segment(prev_payload, prev_response, selected_image_path, annotated_image_path, depth_image_path, mask_paths):
    """
    Continues the conversation by appending a new user message that provides:
      - The newly captured raw image,
      - The annotated image with segmented regions,
      - And individual mask images for each segment.
    Then asks: "Which segment is best to approach next?"
    
    Args:
        prev_payload (dict): Previous conversation payload.
        prev_response (str): The model's previous answer.
        selected_image_path (str): File path to the newly captured raw image.
        annotated_image_path (str): File path to the annotated image.
        mask_paths (list of str): List of file paths for individual mask images.
        api_key (str): Your API key for the VLM/LLM service.
    
    Returns:
        tuple: (new_payload, model_answer)
    """
    new_payload = dict(prev_payload)
    if "messages" not in new_payload:
        new_payload["messages"] = []
    
    new_payload["max_tokens"] = 600

    # Append previous assistant response
    assistant_message = {
        "role": "assistant",
        "content": [{"type": "text", "text": prev_response}]
    }
    new_payload["messages"].append(assistant_message)

    # Build new user message
    user_message = {"role": "user", "content": []}
    prompt = f"""
        Based on your previous selection, you have rotated towards the chosen direction and captured a new raw image of 
        your current heading. You have also generated an annotated image with labeled segments. The first image provided
        to you is the raw RGB image of your current heading. This is followed by the annotated image. Finally, you are also
        given the depth image of your current heading.

        Your goal is to first consider the possible segments that would be the best to approach next in order to achieve your
        goal of efficiently exploring new areas in the environment, and discard any unreasonable segments. 
        Please make sure to use the previous prompt/response in your decision, especially the traversal graph - 
        you want to avoid repeated traversals. 
        
        For the most part, you should try to avoid segments on the floor or ceiling since you will be calculating the 
        distance to the segment based on the given depth image.
        
        Also, deprioritize segments for which there are objects blocking the straight-line path to that segment - 
        for example, if the segment is a door with a couch right in front of it, then that segment should be deprioritized 
        since you cannot go through the couch.
        
        Your answer should be a comma-separated ranking (with nothing else) of the numerical labels of the segments 
        that are still in consideration with NOTHING else (from best to worst).

        The next line(s) directly after your comma-separated rankings should be the actions you would like to execute for 
        each segment in your ranking, for example if the ranking was 5,3,2: 

        Rotate towards segment 5, walk along the hallway towards the living room.
        Rotate towards segment 3, walk towards the open doorway to the garage.
        Rotate towards segment 2, walk to the back wall of the bedroom.
        
        Starting on a new line afterwards, please explain your reasoning for your choices (on a new line). 
        """

    user_message["content"].append({"type": "text", "text": prompt})

    # Attach raw image
    raw_image_encoded = encode_image(selected_image_path)
    user_message["content"].append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{raw_image_encoded}"}
    })
    user_message["content"].append({
        "type": "text",
        "text": "This is the raw image of your current heading."
    })

    # Attach annotated image
    annotated_image_encoded = encode_image(annotated_image_path)
    user_message["content"].append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{annotated_image_encoded}"}
    })
    user_message["content"].append({
        "type": "text",
        "text": "This is the annotated image with labeled segments."
    })

    # Attach each mask image
    # for i, mask_path in enumerate(mask_paths):
    #     mask_encoded = encode_image(mask_path)
    #     user_message["content"].append({
    #         "type": "image_url",
    #         "image_url": {"url": f"data:image/jpeg;base64,{mask_encoded}"}
    #     })
    #     user_message["content"].append({
    #         "type": "text",
    #         "text": f"This is mask_{i}, corresponding to segment {i}."
    #     })

    new_payload["messages"].append(user_message)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    #response = session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=new_payload).json()
    response = safe_chat_completion(session, new_payload, headers)
    if "choices" not in response or not response["choices"]:
        raise RuntimeError(f"Unexpected response structure: {response}")
    model_answer = response["choices"][0]["message"]["content"]
    return model_answer

# For testing purposes, you can include an example main section.
if __name__ == "__main__":
    # Example usage of query_with_sequence
    test_images = ["path/to/image1.png", "path/to/image2.png"]  # Replace with actual image paths
    test_map_graph = "path/to/map_graph.png"  # Replace with the actual path to your map graph image
    test_angles = [0, 40]  # Example rotation angles for each image

    payload, response_text = query_with_sequence(test_images, test_map_graph, test_angles)
    print("Response from initial query:")
    print(response_text)

    # Example usage of get_best_segment
    test_selected_image = "path/to/selected_image.png"       # Replace with your actual selected image path
    test_annotated_image = "path/to/annotated_image.png"       # Replace with your actual annotated image path
    test_mask_paths = ["path/to/mask_0.png", "path/to/mask_1.png"]  # Replace with your actual mask image paths

    new_payload, best_segment_response = get_best_segment(payload, response_text, test_selected_image, test_annotated_image, test_mask_paths, api_key)
    print("Response from get_best_segment:")
    print(best_segment_response)