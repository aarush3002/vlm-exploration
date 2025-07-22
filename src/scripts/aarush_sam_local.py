import os
import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# -----------------------------------------------------------------------------
#  Global model cache – so we only pay the load/compile cost once per process.
# -----------------------------------------------------------------------------
_SAM_MODEL = None  # type: ignore


def _get_sam_model():
    """Lazily load a ViT‑H SAM model onto the fastest available device."""
    global _SAM_MODEL
    if _SAM_MODEL is not None:
        return _SAM_MODEL

    ckpt_path = os.getenv("SAM_CHECKPOINT", "/home/aarush/Downloads/sam_vit_h_4b8939.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"SAM checkpoint not found at '{ckpt_path}'.\n"
            "Set the environment variable SAM_CHECKPOINT to the full path "
            "of your sam_vit_h_*.pth file, or place the file in the current "
            "directory."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sam_local_utils] Loading SAM ViT‑H from {ckpt_path} → {device} …")
    sam = sam_model_registry["vit_h"](checkpoint=ckpt_path)
    sam.to(device)
    sam.eval()

    _SAM_MODEL = sam
    return sam


# -----------------------------------------------------------------------------
# 1) Inference + mask export – drop‑in replacement for the old Replicate call
# -----------------------------------------------------------------------------
def run_sam_inference_and_download_masks(
    input_image_path: str,
    output_folder: str,
    replicate_api_token: str,  # <- kept for API‑compat; ignored now
    poll_interval: float = 2.0  # <- kept for API‑compat; ignored now
):
    """Run SAM locally and write each mask to ``output_folder``.

    Parameters are preserved for backward compatibility.  ``replicate_api_token``
    and ``poll_interval`` are silently ignored.

    Returns
    -------
    mask_paths : List[str]
        Absolute paths to *binary* PNG mask files (0 background, 255 object).
    combined_mask_path : str | None
        Path to a union‑of‑all‑masks image (same encoding) or ``None`` if no
        masks were produced.
    """
    # ----------------------------------------------------------------
    # Prep I/O
    # ----------------------------------------------------------------
    os.makedirs(output_folder, exist_ok=True)

    bgr = cv2.imread(input_image_path)
    if bgr is None:
        raise ValueError(f"Cannot read input image: {input_image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # ----------------------------------------------------------------
    # Run SAM
    # ----------------------------------------------------------------
    sam = _get_sam_model()
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,               # tweak for speed/quality as needed
        pred_iou_thresh=0.9,
        stability_score_thresh=0.92,
        min_mask_region_area=0.02 * rgb.shape[0] * rgb.shape[1],
    )
    print("[sam_local_utils] Generating masks …")
    masks = mask_generator.generate(rgb)  # list[dict]
    print(f"[sam_local_utils] Got {len(masks)} mask(s).")

    MAX_MASKS = 20                  # ← pick your limit
    masks = sorted(masks, key=lambda m: m["area"], reverse=True)[:MAX_MASKS]
    print(f"[sam_local_utils] Trimmed down to {len(masks)} mask(s).")

    # ----------------------------------------------------------------
    # Export masks to disk (like the old HTTP version did)
    # ----------------------------------------------------------------
    mask_paths: list[str] = []
    combined = np.zeros((h, w), dtype=np.uint8)

    for i, m in enumerate(masks):
        binary = m["segmentation"].astype(np.uint8) * 255  # 0/255 image
        path = os.path.join(output_folder, f"mask_{i}.png")
        cv2.imwrite(path, binary)
        mask_paths.append(path)
        combined = np.maximum(combined, binary)
        print(f"[sam_local_utils] Saved {path}")

    combined_mask_path: str | None = None
    if combined.any():
        combined_mask_path = os.path.join(output_folder, "combined_mask.png")
        cv2.imwrite(combined_mask_path, combined)
        print(f"[sam_local_utils] Saved combined mask to {combined_mask_path}")

    return mask_paths, combined_mask_path


# -----------------------------------------------------------------------------
# 2) Annotated overlay – unchanged except for minor doc tweaks
# -----------------------------------------------------------------------------
def create_annotated_image(
    original_image_path: str,
    mask_paths: list,
    annotated_output_path: str,
    camera_hfov_deg: float = 90.0,
):
    """Overlay masks and write a labelled diagnostic JPEG/PNG.

    This function is identical to the version that worked with Replicate – it
    makes a semi‑transparent overlay with random colours and writes a heading
    (in degrees) for each mask index.

    It returns a list of heading angles (center of mask → optical axis), one
    entry per mask path.  If a bounding box cannot be computed, the entry is
    ``None``.
    """
    original_image = cv2.imread(original_image_path)
    if original_image is None:
        raise ValueError(f"Could not open {original_image_path}")

    height, width = original_image.shape[:2]
    annotated_image = original_image.copy()

    segment_headings: list[float | None] = []

    for i, mask_path in enumerate(mask_paths):
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            print(f"[sam_local_utils] Could not load mask: {mask_path}, skipping.")
            segment_headings.append(None)
            continue

        if mask_gray.shape[:2] != (height, width):
            mask_gray = cv2.resize(mask_gray, (width, height), interpolation=cv2.INTER_NEAREST)

        mask_bool = mask_gray > 128
        colour = np.random.randint(0, 255, size=3, dtype=np.uint8)
        annotated_image[mask_bool] = (
            0.5 * annotated_image[mask_bool] + 0.5 * colour
        ).astype(np.uint8)

        x, y, w, h = cv2.boundingRect((mask_bool.astype(np.uint8)) * 255)
        if w == 0 or h == 0:
            segment_headings.append(None)
            continue

        # Text label
        text = str(i)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        cx, cy = x + w // 2, y + h // 2
        tx = int(np.clip(cx - text_w // 2, 0, width - text_w))
        ty = int(np.clip(cy + text_h // 2, text_h, height - baseline))
        cv2.putText(annotated_image, text, (tx, ty), font, font_scale, (255, 255, 255), thickness)

        # Heading calculation
        delta_x = cx - (width / 2.0)
        frac = delta_x / (width / 2.0)
        heading_deg = -frac * (camera_hfov_deg / 2.0)
        segment_headings.append(float(heading_deg))
        print(f"[sam_local_utils] Segment {i}: heading ≈ {heading_deg:.2f}°")

    cv2.imwrite(annotated_output_path, annotated_image)
    print(f"[sam_local_utils] Annotated image saved to {annotated_output_path}")
    return segment_headings
