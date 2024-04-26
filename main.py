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

DEFAULT_IMAGE_DIR = 'image_processing_files/xray_images/'
INPAINTED_IMAGE_DIR = 'temp/'
RESULT_DIR = 'checkpoint/'

USE_CHECKPOINT = False
OUTPUT_TYPE = "png"


def process_images(image_names: list[str], IN_DIR: str, OUT_DIR: str):
    # random.shuffle(image_names)
    for image_name in image_names[12:]:
        print(image_name)

        # Read in the image
        image_path = os.path.join(IN_DIR, image_name)
        original = cv2.imread(image_path)
        image = original.copy()

        if not USE_CHECKPOINT:
            # Perspective Correction
            corners = detect_corners(image)
            image = perspective_correction(image, corners, 0)

            # Inpainting the YCrCb image
            circle = detect_circle(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            image = inpaint(image, circle.astype(np.uint8) * 255, debug=True)
            # y, cr, cb = cv2.split(image)

            # cv2.imshow("image", cv2.hconcat([y, cr, cb]))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            image = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
        else:
            # Noise detection/thresholding
            image = remove_noise(image, image_name, view=True)

            # Colour and Contrast Adjustment

        # eval(original, image, image_name)

        # Save the images
        image_name = image_name.split('.')[0] + '.' + OUTPUT_TYPE
        output_path = os.path.join(OUT_DIR, image_name)
        cv2.imwrite(output_path, image)
        exit()

    # Diagnostic
    full_image_paths = [os.path.join(OUT_DIR, image_name)
                        for image_name in image_names]
    show_random_images(full_image_paths)
    show_random_split_image_gray(full_image_paths)


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
    images = os.listdir(image_dir)

    # Call main routine and measure the quality
    process_images(images, image_dir, RESULT_DIR)

    # Measure using model
    measure(show_dist=False, outpath="dev/dist.png")
