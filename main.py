import os
import sys
import cv2

from detect_circle import detect_circle
from inpaint import inpaint
from util import *
from measure import *
from perspective import *
from noise import *
from color import *

DEFAULT_IMAGE_DIR = 'image_processing_files/xray_images/'
CHECKPOINT_DIR = 'checkpoint/'
RESULT_DIR = 'Results/'

USE_CHECKPOINT = False
OUTPUT_TYPE = "jpg"


def process_images(image_names: list[str], IN_DIR: str, OUT_DIR: str):
    # random.shuffle(image_names)
    full_image_paths = []
    view = False

    # show_random_images([os.path.join(IN_DIR, image_name)
    #                    for image_name in image_names], 2, 2)

    for image_name in image_names[16:17]:
        print(image_name)

        # Read in the image
        image_path = os.path.join(IN_DIR, image_name)
        image = cv2.imread(image_path)

        if not USE_CHECKPOINT:
            # Perspective Correction
            corners = detect_corners(image)
            image = perspective_correction(image, corners)

            # Inpainting the image
            circle = detect_circle(image)
            # image = cv2.inpaint(image, circle.astype(
            #     np.uint8), 3, cv2.INPAINT_TELEA)
            image = inpaint(image, circle, debug=True)

        # # Noise detection/thresholding
        # image = remove_noise(image, image_name, view=view)
        # view = False

        # # Colour and Contrast Adjustment
        # image = cv2.convertScaleAbs(image, alpha=1.31, beta=-41)

        # image = gamma_correction(image, 1.5)

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

    # Call main routine and measure the quality
    process_images(images, image_dir, RESULT_DIR)

    # Measure using model
    acc, rmse, b_score, type2, type1 = measure(
        dir=RESULT_DIR, debug=False)
    print(f"{acc=}")
    print(f"{rmse=}")
    print(f"{b_score=}")
    print(f"{type2=}")
    print(f"{type1=}")
