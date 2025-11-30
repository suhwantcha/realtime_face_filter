import cv2
import numpy as np
from typing import List

def apply_blur_to_boxes(
    image_np: np.ndarray, 
    boxes: List[np.ndarray], 
    blur_strength: int = 25
) -> np.ndarray:
    """
    Applies a Gaussian blur to the specified bounding box regions in an image.

    Args:
        image_np (np.ndarray): The input image as a NumPy array (BGR format).
        boxes (List[np.ndarray]): A list of face bounding boxes [x1, y1, x2, y2].
        blur_strength (int): The strength of the Gaussian blur. Must be an odd number.

    Returns:
        np.ndarray: The image with the specified faces blurred.
    """
    if blur_strength % 2 == 0:
        blur_strength += 1  # Ensure blur_strength is odd

    blurred_image = image_np.copy()
    
    for box in boxes:
        # Ensure box coordinates are integers and within image bounds
        x1, y1, x2, y2 = [int(coord) for coord in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_np.shape[1], x2), min(image_np.shape[0], y2)

        if x2 > x1 and y2 > y1:  # Ensure valid region
            # Extract the face region
            face_roi = blurred_image[y1:y2, x1:x2]
            # Apply Gaussian blur to the face region
            blurred_face = cv2.GaussianBlur(face_roi, (blur_strength, blur_strength), 30)
            # Put the blurred face back into the image
            blurred_image[y1:y2, x1:x2] = blurred_face
            
    return blurred_image