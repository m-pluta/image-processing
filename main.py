import os
import sys
import cv2

from perspective import *
from detect_circle import detect_circle
from inpaint import inpaint
from noise import *
from color import *

DEFAULT_IMAGE_DIR = 'image_processing_files/xray_images/'
RESULT_DIR = 'Results/'


def process_images(IN_DIR: str, OUT_DIR: str):
    """Processes all the images in `IN_DIR` and saves them to `OUT_DIR`

    Args:
        IN_DIR (str): Source directory containing noisy unclean images
        OUT_DIR (str): Target directory to save the clean images to
    """
    # Read in all image names
    image_names = sorted(os.listdir(image_dir))

    for image_name in image_names:
        print(image_name)

        # Read in the image
        image_path = os.path.join(IN_DIR, image_name)
        image = cv2.imread(image_path)

        # Perspective Correction
        corners = detect_corners(image)
        image = perspective_correction(image, corners)

        # Inpainting the image
        circle = detect_circle(image)
        image = inpaint(image, circle)

        # Noise detection/thresholding
        image = remove_noise(image)

        # Colour and Contrast Adjustment
        image = cv2.convertScaleAbs(image, alpha=1.31, beta=-41)

        image = gamma_correction(image, 1.5)

        # Save the images
        output_path = os.path.join(OUT_DIR, image_name)
        cv2.imwrite(output_path, image)


if __name__ == '__main__':
    # Create the output directory
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Identify the input directory
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    else:
        image_dir = DEFAULT_IMAGE_DIR

    # Call main routine and measure the quality
    process_images(image_dir, RESULT_DIR)
