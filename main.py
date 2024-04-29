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

USE_CHECKPOINT = True
OUTPUT_TYPE = "jpg"


def process_images(image_names: list[str], IN_DIR: str, OUT_DIR: str, a, b, y):
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
            image = perspective_correction(image, corners, None)

            # Inpainting the image
            circle = detect_circle(image)
            image = inpaint(image, circle, debug=False)
        else:

            # Noise detection/thresholding
            image = remove_noise(image, image_name, view=view)
            view = False

            # Colour and Contrast Adjustment
            image = cv2.convertScaleAbs(image, alpha=a, beta=b)

            image = gamma_correction(image, y)

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
    plt.close('all')


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

    # images = [image for image in images if int(image[2:5]) in (
    #     14, 22, 23, 29, 32, 33, 41, 84, 97, 100)]

    # process_images(images, image_dir, RESULT_DIR,
    #                (-2, -2, -2, 0, -1, 2, 0, 0), -40)

    # acc, mse = measure(show_dist=False, outpath="dev/dist.png")

    # combinations = list(itertools.product((-2, -1, 0, 1, 2), repeat=8))
    # alpha_range = list(np.arange(1.25, 1.36, 0.02))
    # beta_range = list(np.arange(-35, -46, -2))
    # gamma_range = list(np.arange(1.4, 1.51, 0.02))
    # combinations = list(itertools.product(
    #     alpha_range, beta_range, gamma_range))
    # random.shuffle(combinations)

    combinations = [(1.31, -41, 1.5)]

    scores = {}

    for combination in combinations:
        print(combination)
        # Call main routine and measure the quality
        process_images(images, image_dir, RESULT_DIR, *combination)

        # Measure using model
        acc, mse = measure(show_dist=False, outpath="dev/dist.png")

        if not acc in scores:
            scores[acc] = []

        scores[acc].append(combination)

        with open('bests9.txt', 'w') as file:
            for key in sorted(scores.keys(), reverse=True)[:5]:
                file.write(f'{key}: {scores[key]}\n')
