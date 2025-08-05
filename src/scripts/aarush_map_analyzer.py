import cv2
import numpy as np
import os

def analyze_map_coverage(image_path):
    """
    Loads a clean exploration map and calculates the percentage of explored space
    using predefined, exact color values.

    Args:
        image_path (str): The path to the input map image file.
    """
    # --- 1. Load the Image ---
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        return

    map_image = cv2.imread(image_path)
    if map_image is None:
        print(f"Error: Could not read the image file. It may be corrupted or in an unsupported format.")
        return

    # --- 2. Define Exact Color Values (in BGR format) ---
    # IMPORTANT: Replace these with the exact BGR values you have identified.
    # BGR is (Blue, Green, Red).
    # For example, if your seen color is RGB(128, 128, 128), the BGR value is (128, 128, 128).
    # If your unseen color is RGB(105, 105, 105), the BGR value is (105, 105, 105).
    SEEN_COLOR_BGR = np.array([150, 150, 150])  # CHANGE THIS - The light gray color for explored areas
    UNSEEN_COLOR_BGR = np.array([75, 75, 75]) # CHANGE THIS - The dark gray color for unexplored areas

    # --- 3. Create Masks and Count Pixels ---
    # Create a mask for each color. This will be True where the pixel color matches exactly.
    mask_seen = np.all(map_image == SEEN_COLOR_BGR, axis=-1)
    mask_unseen = np.all(map_image == UNSEEN_COLOR_BGR, axis=-1)

    # Count the number of True values in each mask
    seen_pixels = np.count_nonzero(mask_seen)
    unseen_pixels = np.count_nonzero(mask_unseen)

    print("--- Pixel Counts ---")
    print(f"Seen (Color: {SEEN_COLOR_BGR}):      {seen_pixels} pixels")
    print(f"Unseen (Color: {UNSEEN_COLOR_BGR}):    {unseen_pixels} pixels")
    print("-" * 22)

    # --- 4. Calculate Exploration Percentage ---
    total_explorable_pixels = seen_pixels + unseen_pixels

    if total_explorable_pixels == 0:
        print("Could not find any explorable (seen or unseen) pixels.")
        print("Please verify the BGR color values in the script.")
        exploration_percentage = 0.0
    else:
        exploration_percentage = (seen_pixels / total_explorable_pixels) * 100

    print("\n--- Coverage Calculation ---")
    print(f"Formula: (Seen Pixels) / (Seen Pixels + Unseen Pixels)")
    print(f"Calculation: {seen_pixels} / ({seen_pixels} + {unseen_pixels})")
    print(f"Exploration Percentage: {exploration_percentage:.2f}%")


if __name__ == '__main__':
    # Replace with the actual path to your image file
    image_file_path = '/home/aarush/Results/2t7WUuJeko7/final_clean_exploration_map_1753826910.png'
    analyze_map_coverage(image_file_path)