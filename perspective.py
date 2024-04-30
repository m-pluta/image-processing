import cv2
import numpy as np


def detect_corners(image):
    # Convert to gray-scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get the binary thresholded image
    _, thresh = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)

    # Find all the contours in the image
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the contour with max area - this should be the image
    contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a quadrilateral and get the corners
    peri = cv2.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, 0.02 * peri, True)

    return corners


def perspective_correction(image, corners):
    # Define the destination position
    destination = np.array([
        [257, 2], [-3, -2], [-2, 255], [255, 255],
    ], dtype="float32")

    # Define the matrix for the perspective transform
    matrix = cv2.getPerspectiveTransform(
        corners.astype(np.float32), destination)

    # Apply the perspective transform to the image
    image = cv2.warpPerspective(image, matrix, (256, 256))

    return image
