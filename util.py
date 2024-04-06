import cv2
import random
import matplotlib.pyplot as plt
import numpy as np


def show_random_images(image_paths):
    GRID = 2

    # Select random images
    random_paths = random.sample(image_paths, GRID * GRID)

    _, axes = plt.subplots(GRID, GRID, figsize=(10, 10))

    for i, image_path in enumerate(random_paths):
        image = cv2.imread(image_path)
        # image = getFourierImage(image_path)

        axes[i // GRID, i % GRID].imshow(image, cmap='gray')
        # axes[i // GRID, i % GRID].set_title(image_file)
        axes[i // GRID, i % GRID].axis('off')

    plt.tight_layout()
    plt.savefig("view.png")

def show_random_split_image_color(image_paths):

    # Select the random image
    random_path = random.choice(image_paths)
    image = cv2.imread(random_path)

    # Create empty matrices with zeros for each color channel image
    blue_channel = np.zeros_like(image)
    green_channel = np.zeros_like(image)
    red_channel = np.zeros_like(image)

    # Assign the respective channel to each matrix.
    blue_channel[:, :, 2] = image[:, :, 0]
    green_channel[:, :, 1] = image[:, :, 1]
    red_channel[:, :, 0] = image[:, :, 2]

    # Prepare the figure
    _, axes = plt.subplots(2, 2, figsize=(10, 10))
    images = [image, red_channel, green_channel, blue_channel]
    titles = [random_path, 'Red Channel', 'Green Channel', 'Blue Channel']

    # Create the figure
    for i, img in enumerate(images):
        ax = axes[i // 2, i % 2]
        ax.imshow(img)
        ax.set_title(titles[i])
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("split.png")

def show_random_split_image_gray(image_paths):
    # Ensure the random choice can be made.
    if not image_paths:
        print("The list of image paths is empty.")
        return

    # Select a random image
    random_path = random.choice(image_paths)
    image = cv2.imread(random_path)

    # Check if the image was successfully loaded
    if image is None:
        print(f"Failed to load image at {random_path}")
        return

    # Split the image into its channels
    channels = cv2.split(image)

    # Prepare the figure
    _, axes = plt.subplots(2, 2, figsize=(10, 10))
    images = [image] + list(reversed(channels))

    titles = [random_path, 'Red Channel', 'Green Channel', 'Blue Channel']

    for i, img in enumerate(images):
        ax = axes[i // 2, i % 2]
        if i == 0:
            ax.imshow(img)  # Original image does not need a colormap
        else:
            ax.imshow(img, cmap='gray')  # Color channels in grayscale
        ax.axis('off')
        ax.set_title(titles[i])

    plt.tight_layout()
    plt.savefig("split.png")