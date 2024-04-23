import os
import sys
import cv2

from detect_circle import detect_circle
from inpaint import inpaint
from util import *
from measure import *
from perspective import *
from blur import *
from filter import *
from noise import *

DEFAULT_IMAGE_DIR = 'image_processing_files/xray_images/'
INPAINTED_IMAGE_DIR = 'temp/'
RESULT_DIR = 'Results/'

USE_INPAINTED = False


def process_images(image_names: list[str], in_dir: str, out_dir: str):
    # random.shuffle(image_names)
    for _, image_name in enumerate(image_names[6:]):
        # Read in the image
        image_path = os.path.join(in_dir, image_name)
        image = original = cv2.imread(image_path)

        if not USE_INPAINTED:
            # Perspective Correction
            corners = detect_corners(image)
            image = perspective_correction(image, corners, 0)

            # Inpainting
            circle = detect_circle(image)
            image = inpaint(image, circle.astype(np.uint8) * 255)
        else:
            # Noise detection/thresholding
            image = remove_noise(image, image_name, True)

            # Colour and Contrast Adjustment

            # eval(original, image, image_name)

            # Save the images
            output_path = os.path.join(out_dir, image_name)
            cv2.imwrite(output_path, image)

        exit()
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
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    else:
        if USE_INPAINTED:
            image_dir = INPAINTED_IMAGE_DIR
        else:
            image_dir = DEFAULT_IMAGE_DIR

    # Read in all image names
    images = os.listdir(image_dir)

    # Call main routine and measure the quality
    process_images(images, image_dir, RESULT_DIR)

    # Measure using model
    measure(show_dist=False, outpath="dev/dist.png")
