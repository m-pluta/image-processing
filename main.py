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
CHECKPOINT_DIR = 'checkpoint/'
RESULT_DIR = 'Results/'

USE_CHECKPOINT = False
OUTPUT_TYPE = "png"


def process_images(image_names: list[str], IN_DIR: str, OUT_DIR: str, comb, beta):
    # random.shuffle(image_names)
    full_image_paths = []
    view = True
    for image_name in image_names:
        print(image_name)

        # Read in the image
        image_path = os.path.join(IN_DIR, image_name)
        original = cv2.imread(image_path)
        image = original.copy()

        if not USE_CHECKPOINT:
            # Perspective Correction
            corners = detect_corners(image)
            image = perspective_correction(image, corners, *comb)

            # Inpainting the image
            circle = detect_circle(image)
            image = inpaint(image, circle, debug=True)

        # # Noise detection/thresholding
        # image = remove_noise(image, image_name, view=view)
        # view = False

        # # Colour and Contrast Adjustment
        # image = cv2.convertScaleAbs(image, alpha=1.3, beta=-30)

        # image = gamma_correction(image, 1.6)

        # eval(original, image, image_name)

        # Save the images
        image_name = image_name.split('.')[0] + '.' + OUTPUT_TYPE
        output_path = os.path.join(OUT_DIR, image_name)
        full_image_paths.append(output_path)
        cv2.imwrite(output_path, image)

    # Diagnostic
    # show_random_images(full_image_paths)
    # show_split_image_RGB(full_image_paths)
    # show_split_image_LAB(full_image_paths)
    # show_split_image_YCrCB(full_image_paths)


if __name__ == '__main__':
    # Create the output directory
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Identify the input directory
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    else:
        if USE_CHECKPOINT:
            image_dir = CHECKPOINT_DIR
        else:
            image_dir = DEFAULT_IMAGE_DIR

    # Read in all image names
    images = sorted(os.listdir(image_dir))

    images = [image for image in images if int(image[2:5]) in (
        14, 22, 23, 29, 32, 33, 41, 84, 97, 100)]

    process_images(images, image_dir, CHECKPOINT_DIR,
                   (-2, -2, -2, 0, -1, 2, 0, 0), -40)

    # combinations = list(itertools.product((-2, -1, 0, 1, 2), repeat=8))
    # beta_range = list(np.arange(-35, -70, -5))
    # combinations = [((-2, -2, -2, 0, -1, 2, 0, 0), -40), ((2, -2, -2, 1, -1, 2, 0, 2), -45),
    #                 ((-1, 1, 2, -2, -2, 2, -1, -1), -55)]
    # # combinations = list(itertools.product(combinations, beta_range))
    # # random.shuffle(combinations)

    # scores = {}

    # for combination, beta in combinations:
    #     print(combination)
    #     # Call main routine and measure the quality
    #     process_images(images, image_dir, RESULT_DIR, combination, beta)

    #     # Measure using model
    #     acc, mse = measure(show_dist=False, outpath="dev/dist.png")

    #     if not acc in scores:
    #         scores[acc] = []

    #     scores[acc].append((combination, beta))

    #     with open('bests9.txt', 'w') as file:
    #         for key in sorted(scores.keys(), reverse=True)[:5]:
    #             file.write(f'{key}: {scores[key]}\n')
