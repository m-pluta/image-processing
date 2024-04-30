import cv2
import numpy as np

MASK_STROKE_WIDTH = 2


def detect_circle(image: np.ndarray):
    """Detects the largest circular shape in an image assuming it is darker than the background

    Args:
        image (np.ndarray): The input image

    Returns:
        np.ndarray: Boolean mask containing the circle to be inpainted
    """
    # Get the gray image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a mask for the black circle
    mask = (gray_image < 40).astype(np.uint8)

    # Find the contours
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask for the largest contour
    largest_contour_mask = np.zeros_like(mask)

    if not contours:
        return largest_contour_mask.astype(bool)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Draw the contour with particular thickness as to fully capture the black circle
    cv2.drawContours(largest_contour_mask, [
        largest_contour], -1, 255, thickness=MASK_STROKE_WIDTH)

    # Fill in the contour
    cv2.fillPoly(largest_contour_mask, [largest_contour], 255)

    # Return the boolean mask for the circle
    return largest_contour_mask.astype(bool)
