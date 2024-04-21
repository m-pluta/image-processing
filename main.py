import os
import sys
import cv2

from color import *
from detect_circle import detect_circle
from util import *
from measure import *
from perspective import *
from blur import *
from filter import *
from noise import *
from noise_old import *
from Inpainter import inpaint

DEFAULT_IMAGE_DIR = 'image_processing_files/xray_images/'
RESULT_DIR = 'Results/'


def process_images(image_names: list[str], in_dir: str, out_dir: str):
    # random.shuffle(image_names)
    for _, image_name in enumerate(image_names):
        # Read in the image
        image_path = os.path.join(in_dir, image_name)
        image = cv2.imread(image_path)

        # Perspective Correction
        corners = detect_corners(image)
        image = perspective_correction(image, corners, 0)

        # # Inpainting
        circle = detect_circle(image)
        image = inpaint(image, circle.astype(np.uint8) * 255)

        # Noise detection/thresholding
        # image = remove_noise(image, image_name, circle)

        # Colour and Contrast Adjustment

        # Save the images
        output_path = os.path.join("temp", image_name)
        cv2.imwrite(output_path, image)

    # Get full paths
    full_image_paths = [os.path.join(out_dir, image_name)
                        for image_name in image_names]

    # Diagnostic
    show_random_images(full_image_paths)
    show_random_split_image_gray(full_image_paths)


if __name__ == '__main__':
    # Create the output directory
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Identify the input directory
    image_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_DIR

    # Read in all image names
    images = os.listdir(image_dir)

    # Call main routine and measure the quality
    process_images(images, image_dir, RESULT_DIR)

    # Measure using model
    measure(show_dist=False, outpath="dev/dist.png")
