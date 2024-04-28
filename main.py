import os
import sys
import cv2

from detect_circle import detect_circle
from inpaint import inpaint
from util import *
from measure import *
from perspective import *
from blur import *
from noise import *
from color import *
import itertools

DEFAULT_IMAGE_DIR = 'image_processing_files/xray_images/'
INPAINTED_IMAGE_DIR = 'checkpoint/'
RESULT_DIR = 'Results/'

USE_CHECKPOINT = False
OUTPUT_TYPE = "jpg"


def process_images(image_names: list[str], IN_DIR: str, OUT_DIR: str, comb):
    # random.shuffle(image_names)
    full_image_paths = []
    view = True
    for image_name in image_names:
        # print(image_name)

        # Read in the image
        image_path = os.path.join(IN_DIR, image_name)
        original = cv2.imread(image_path)
        image = original.copy()

        if not USE_CHECKPOINT:
            # Perspective Correction
            corners = detect_corners(image)
            image = perspective_correction(image, corners, *comb)

            # # Inpainting the YCrCb image
            # circle = detect_circle(image)
            # image = inpaint(image, circle, debug=True)

        # Noise detection/thresholding
        image = remove_noise(image, image_name, view=view)
        view = False

        # Colour and Contrast Adjustment
        image = cv2.convertScaleAbs(image, alpha=1.4, beta=-65)

        image = gamma_correction(image, 1.4)

        # eval(original, image, image_name)

        # Save the images
        image_name = image_name.split('.')[0] + '.' + OUTPUT_TYPE
        output_path = os.path.join(OUT_DIR, image_name)
        full_image_paths.append(output_path)
        cv2.imwrite(output_path, image)

    # Diagnostic
    show_random_images(full_image_paths)
    show_split_image_RGB(full_image_paths)
    show_split_image_LAB(full_image_paths)
    show_split_image_YCrCB(full_image_paths)


if __name__ == '__main__':
    # Create the output directory
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Identify the input directory
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    else:
        if USE_CHECKPOINT:
            image_dir = INPAINTED_IMAGE_DIR
        else:
            image_dir = DEFAULT_IMAGE_DIR

    # Read in all image names
    images = sorted(os.listdir(image_dir))

    combinations = list(itertools.product((-2, -1, 0, 1, 2), repeat=8))
    # combinations = [(0, 0, -1, -1, 1, 1, -1, 1)]
    random.shuffle(combinations)
    scores = {}

    for combination in combinations:
        print(combination)
        # Call main routine and measure the quality
        process_images(images, image_dir, RESULT_DIR, combination)

        # Measure using model
        acc, mse = measure(show_dist=False, outpath="dev/dist.png")

        if not acc in scores:
            scores[acc] = []

        scores[acc].append(combination)

        with open('bests7.txt', 'w') as file:
            for key in sorted(scores.keys(), reverse=True)[:5]:
                file.write(f'{key}: {scores[key]}\n')
