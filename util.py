import cv2
import random
import matplotlib.pyplot as plt

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