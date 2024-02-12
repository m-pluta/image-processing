import cv2
import numpy as np

def identify_R(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_violet = np.array([135, 50, 50])
    upper_violet = np.array([155, 255, 255])

    # Create mask for the violet color
    mask = cv2.inRange(hsv, lower_violet, upper_violet)

    

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    blank_image = np.zeros_like(image)

    # Optional: Draw contours on the original image for visualization
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5:  # Filter based on area, adjust according to your needs
            cv2.drawContours(blank_image, [cnt], -1, (0, 255, 0), 3)
            
    cv2.imshow('Contours on Blank Image', blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
