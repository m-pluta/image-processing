import glob
import os
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
import sys

DEFAULT_IMAGE_DIR = 'image_processing_files/xray_images/'
RESULT_DIR = 'Results/'


def get_fourier_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    return 20*np.log(np.abs(fshift))


def show_fourier_image(image_path):
    f_image = get_fourier_image(image_path)
    plt.imshow(f_image, cmap='gray')
    plt.title('Fourier Transform')
    plt.axis('off')
    plt.show()


def show_random_images(image_paths):
    GRID = 4

    # Select random images
    random_paths = random.sample(image_paths, GRID * GRID)

    _, axes = plt.subplots(GRID, GRID, figsize=(10, 10))

    for i, image_path in enumerate(random_paths):
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        # image = getFourierImage(image_path)

        axes[i // GRID, i % GRID].imshow(image, cmap='gray')
        # axes[i // GRID, i % GRID].set_title(image_file)
        axes[i // GRID, i % GRID].axis('off')

    plt.show()


def process_images(images: list[str], in_dir: str, out_dir: str):
    show_random_images([os.path.join(in_dir, image) for image in images])

    for _, image in enumerate(images):
        # Read in the image
        image_path = os.path.join(in_dir, image)
        image = cv2.imread(image_path)

        output_path = os.path.join(out_dir, image)
        cv2.imwrite(output_path, image)

        pass


if __name__ == '__main__':
    os.makedirs(RESULT_DIR, exist_ok=True)

    image_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_DIR
    images = os.listdir(image_dir)

    process_images(images, image_dir, RESULT_DIR)
