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
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        # image = getFourierImage(image_path)

        axes[i // GRID, i % GRID].imshow(image, cmap='gray')
        # axes[i // GRID, i % GRID].set_title(image_file)
        axes[i // GRID, i % GRID].axis('off')

    plt.tight_layout()
    plt.savefig("view.png")
    
def show_random_split_image(image_paths):
    
    # Select the random image
    random_path = random.choice(image_paths)
    image = cv2.imread(random_path)
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create empty matrices with zeros for each color channel image
    red_channel = np.zeros_like(image_rgb)
    green_channel = np.zeros_like(image_rgb)
    blue_channel = np.zeros_like(image_rgb)

    # Assign the respective channel to each matrix. Note that RGB ordering is used.
    red_channel[:, :, 0] = image_rgb[:, :, 0]
    green_channel[:, :, 1] = image_rgb[:, :, 1]
    blue_channel[:, :, 2] = image_rgb[:, :, 2]
    
    # Prepare the figure
    _, axes = plt.subplots(2, 2, figsize=(10, 10))
    images = [image_rgb, red_channel, green_channel, blue_channel]
    titles = [random_path, 'Red Channel', 'Green Channel', 'Blue Channel']
    
    # Create the figure
    for i, img in enumerate(images):
        ax = axes[i // 2, i % 2]
        ax.imshow(img)
        ax.set_title(titles[i])
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("split.png")
    
