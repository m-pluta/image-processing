import os
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
import sys

GRID = 4
SAVE_DIR = 'Results/'

def getFourierImage(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    return 20*np.log(np.abs(fshift))


def process_images(images_path):
    file_list = os.listdir(images_path)

    # Select random images
    random_images = random.sample(file_list, GRID * GRID)

    # Plot the images
    _, axes = plt.subplots(GRID, GRID, figsize=(10, 10))
    for i, image_file in enumerate(random_images):
        print(i, image_file)
        image_path = os.path.join(images_path, image_file)
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        # image = getFourierImage(image_path)
        
        axes[i // GRID, i % GRID].imshow(image, cmap = 'gray')
        axes[i // GRID, i % GRID].set_title(image_file)
        axes[i // GRID, i % GRID].axis('off')
        
    plt.show()


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'image_processing_files/xray_images/'

    process_images(path)
