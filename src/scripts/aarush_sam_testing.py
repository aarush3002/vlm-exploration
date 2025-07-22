import os
import time
import base64
import requests
import cv2
import numpy as np


def run_sam_inference_and_download_masks(
    input_image_path: str,
    output_folder: str,
    replicate_api_token: str,
    poll_interval: float = 2.0
):
    """
    1) Sends a local image to the Replicate SAM model (meta/sam-2) via raw HTTP calls.
    2) Polls until it's done or fails.
    3) Downloads each individual mask to `mask_i.png` in the output_folder.
    4) Returns the list of mask file paths, plus an optional combined_mask_path if present.

    Args:
        input_image_path (str): Local image to be segmented.
        output_folder (str): Folder where mask files will be downloaded.
        replicate_api_token (str): Your Replicate API token string.
        poll_interval (float): Seconds between status polls.

    Returns:
        mask_paths (List[str]): List of file paths for the individual masks.
        combined_mask_path (str|None): Path to the combined mask image, or None if not present.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Base64-encode the local image
    with open(input_image_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode("utf-8")

    # Create the prediction
    create_url = "https://api.replicate.com/v1/predictions"
    headers = {
        "Authorization": f"Token {replicate_api_token}",
        "Content-Type": "application/json",
    }
    json_data = {
        "version": "fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83",
        "input": {
            # The model expects either a URL or a base64 data URI for "image"
            "image": "data:image/png;base64," + b64_image
        },
    }

    create_resp = requests.post(create_url, headers=headers, json=json_data)
    create_resp.raise_for_status()
    prediction = create_resp.json()

    # Poll until "succeeded", "failed", or "canceled"
    get_url = prediction["urls"]["get"]
    status = prediction["status"]
    prediction_id = prediction["id"]
    print(f"Created prediction: {prediction_id}, initial status={status}")

    while status not in ["succeeded", "failed", "canceled"]:
        time.sleep(poll_interval)
        poll_resp = requests.get(get_url, headers=headers)
        poll_resp.raise_for_status()
        prediction = poll_resp.json()
        status = prediction["status"]
        print(f"Current status: {status}")

    # If prediction not successful, return empty results
    if status != "succeeded":
        print(f"Prediction did not succeed. Status: {status}. Error: {prediction.get('error')}")
        return [], None

    # Grab outputs from the final response
    output = prediction["output"]
    combined_mask_url = output.get("combined_mask")
    individual_masks = output.get("individual_masks", [])

    print("Downloading individual masks into:", output_folder)

    # Download individual masks
    mask_paths = []
    for i, mask_url in enumerate(individual_masks):
        resp = requests.get(mask_url)
        resp.raise_for_status()
        mask_filename = f"mask_{i}.png"
        mask_path = os.path.join(output_folder, mask_filename)
        with open(mask_path, "wb") as f:
            f.write(resp.content)
        mask_paths.append(mask_path)
        print(f"Saved {mask_path}")

    # Download combined mask (optional)
    combined_mask_path = None
    if combined_mask_url:
        resp = requests.get(combined_mask_url)
        resp.raise_for_status()
        combined_filename = "combined_mask.png"
        combined_mask_path = os.path.join(output_folder, combined_filename)
        with open(combined_mask_path, "wb") as f:
            f.write(resp.content)
        print(f"Saved combined mask to {combined_mask_path}")

    return mask_paths, combined_mask_path


def create_annotated_image(
    original_image_path: str,
    mask_paths: list,
    annotated_output_path: str,
    camera_hfov_deg: float = 90.0
):
    """
    Reads the original image and a list of mask file paths.
    Overlays each mask with a random color, labels it using smaller text
    centered in the bounding box region, and computes the heading
    of each segment relative to the camera's yaw.

    Args:
        original_image_path (str): Path to the original, unsegmented image.
        mask_paths (List[str]): A list of local mask filenames to overlay.
        annotated_output_path (str): Where to save the final annotated image.
        camera_hfov_deg (float): The camera's horizontal field of view in degrees.

    Returns:
        List[float]: A list of segment headings (in degrees) corresponding
                     to each mask in `mask_paths`.
    """
    # Read original image
    original_image = cv2.imread(original_image_path)
    if original_image is None:
        raise ValueError(f"Could not open {original_image_path}")
    height, width = original_image.shape[:2]

    # Make a copy for annotation
    annotated_image = original_image.copy()

    segment_headings = []  # store final headings for each mask

    # For each mask, overlay color and label
    for i, mask_path in enumerate(mask_paths):
        # Load mask as grayscale
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            print(f"Could not load mask: {mask_path}, skipping.")
            segment_headings.append(None)
            continue

        # Resize mask if needed
        if mask_gray.shape[:2] != (height, width):
            mask_gray = cv2.resize(mask_gray, (width, height), interpolation=cv2.INTER_NEAREST)

        # Convert to boolean
        mask_bool = mask_gray > 128

        # Random color (BGR)
        color = np.random.randint(0, 255, size=3, dtype=np.uint8)

        # Blend color into annotated_image at mask positions
        annotated_image[mask_bool] = (
            0.5 * annotated_image[mask_bool] + 0.5 * color
        ).astype(np.uint8)

        # Find bounding box to place text
        mask_uint8 = (mask_bool.astype(np.uint8)) * 255
        x, y, w, h = cv2.boundingRect(mask_uint8)

        if w > 0 and h > 0:
            # We'll label the region with (i), centered in the bounding box

            text = str(i)
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4  # [CHANGED] smaller text
            thickness = 1     # [CHANGED] thinner text
            text_color = (255, 255, 255)  # white

            # Compute text size
            (text_w, text_h), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)

            # Coordinates to center the text in the bounding box
            center_x = x + w // 2
            center_y = y + h // 2

            # Adjust so text is centered
            text_x = center_x - (text_w // 2)
            # Add half text height so it's visually centered
            text_y = center_y + (text_h // 2)

            # [CHANGED] Clamp text_x/text_y so text is not cut off at edges
            text_x = max(0, min(text_x, width - text_w))
            # For vertical clamping, make sure top of text is not above 0,
            # and bottom of text is not below image bottom.
            # (text_y is the baseline, so we clamp accordingly.)
            text_y = max(text_h, min(text_y, height - baseline))

            # Place the text
            cv2.putText(
                annotated_image,
                text,
                (text_x, text_y),
                font_face,
                font_scale,
                text_color,
                thickness
            )

            # ---- Compute Heading Relative to Camera Yaw ----
            # 1) Horizontal offset from image center
            delta_x = center_x - (width / 2.0)
            # 2) Fraction of half-image
            frac_of_half = delta_x / (width / 2.0)
            # 3) Multiply by half of the HFOV
            angle_offset_deg = frac_of_half * (camera_hfov_deg / 2.0)
            angle_offset_deg = -angle_offset_deg

            segment_headings.append(angle_offset_deg)
            print(f"Segment {i} => bounding box center=({center_x}, {center_y}), "
                  f"heading ~ {angle_offset_deg:.2f}°")
        else:
            # If the bounding box is empty, no heading
            segment_headings.append(None)

    # Finally save the annotated image
    cv2.imwrite(annotated_output_path, annotated_image)
    print(f"Annotated image saved to {annotated_output_path}")

    return segment_headings
