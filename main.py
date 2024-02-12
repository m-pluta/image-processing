import os
import sys
import cv2

from color import equalizeHist
from util import show_random_images
from measure import measure
from perspective import perspective_correction

DEFAULT_IMAGE_DIR = 'image_processing_files/xray_images/'
RESULT_DIR = 'Results/'


def process_image(image):
    image = equalizeHist(image, 'LAB')
    return image


def process_images(images: list[str], in_dir: str, out_dir: str):
    # show_random_images([os.path.join(in_dir, image) for image in images])
    for _, image_name in enumerate(images):
        # Read in the image
        image_path = os.path.join(in_dir, image_name)
        image = cv2.imread(image_path)
        
        perspective_correction(image)
        break

        image = process_image(image)

        output_path = os.path.join(out_dir, image_name)
        cv2.imwrite(output_path, image)


if __name__ == '__main__':
    os.makedirs(RESULT_DIR, exist_ok=True)

    image_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_DIR
    images = os.listdir(image_dir)

    process_images(images, image_dir, RESULT_DIR)
