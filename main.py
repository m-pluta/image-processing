import os
import sys
import cv2

from color import equalizeHist
from util import show_random_images
from measure import measure
from perspective import get_image_corners, perspective_correction, drawBorders
from blur import median_blur, N1_means_blur, bilateral_blur
from filter import butterworth

DEFAULT_IMAGE_DIR = 'image_processing_files/xray_images/'
RESULT_DIR = 'Results/'

def process_images(image_names: list[str], in_dir: str, out_dir: str):

    for _, image_name in enumerate(image_names):
        # Read in the image
        image_path = os.path.join(in_dir, image_name)
        image = cv2.imread(image_path)

        # Processing images
        corners = get_image_corners(image)
        image = perspective_correction(image, corners, 1)
        # image = equalizeHist(image)
        # image = bilateral_blur(image)
        # image = N1_means_blur(image)

        # Save the image
        output_path = os.path.join(out_dir, image_name)
        cv2.imwrite(output_path, image)

    show_random_images([os.path.join(out_dir, image_name) for image_name in image_names])

if __name__ == '__main__':
    # Create the output directory
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Identify the input directory
    image_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_DIR

    # Read in all image names
    images = os.listdir(image_dir)

    # Call main routine and measure the quality
    process_images(images, image_dir, RESULT_DIR)
    measure(show_dist=False, outpath="dist.png")
