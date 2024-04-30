import cv2
import numpy as np


def gamma_correction(image, gamma=1.0):
    # Create look up table for gamma correction
    table = np.array([((i / 255.0) ** gamma) *
                     255 for i in np.arange(0, 256)]).astype("uint8")

    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)
