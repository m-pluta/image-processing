import cv2
import numpy as np


def gamma_correction(image: np.ndarray, gamma: float = 1.0):
    """Performs gamma correction on an input `image`

    Args:
        image (np.ndarray): The input image 
        gamma (float, optional): The parameter for varying strength. Defaults to 1.0.

    Returns:
        _type_: Gamma corrected image
    """
    # Create look up table for gamma correction
    table = np.array([((i / 255.0) ** gamma) *
                     255 for i in np.arange(0, 256)]).astype("uint8")

    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)
