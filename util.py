import cv2
import random
import matplotlib.pyplot as plt


def show_random_images(image_paths, row=6, col=3):
    # Select random images
    random_paths = random.sample(image_paths, row * col)

    _, axes = plt.subplots(col, row, figsize=(5 * row, 5 * col))

    for i, image_path in enumerate(random_paths):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        axes[i // row, i % row].imshow(image, cmap='gray')
        axes[i // row, i % row].set_title(image_path.split('/')[-1])
        axes[i // row, i % row].axis('off')

    plt.tight_layout()
    plt.savefig("dev/view.png")


def show_split_image_RGB(image_paths):

    # Select the random image
    random_path = random.choice(image_paths)
    original = cv2.imread(random_path)
    image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    channels = cv2.split(image)

    # Prepare the figure
    _, axes = plt.subplots(2, 2, figsize=(10, 10))
    images = [original] + list(channels)
    titles = [random_path, 'Red Channel', 'Green Channel', 'Blue Channel']

    # Create the figure
    for i, img in enumerate(images):
        ax = axes[i // 2, i % 2]
        if i == 0:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap='gray')
        ax.set_title(titles[i])
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("dev/splitRGB.png")


def show_split_image_LAB(image_paths):
    # Select a random image
    random_path = random.choice(image_paths)
    original = cv2.imread(random_path)
    image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # Split the image into its channels
    channels = cv2.split(image)

    # Prepare the figure
    _, axes = plt.subplots(2, 2, figsize=(10, 10))
    images = [original] + list(channels)

    titles = [random_path, 'L Channel', 'A Channel', 'B Channel']

    for i, img in enumerate(images):
        ax = axes[i // 2, i % 2]
        if i == 0:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(titles[i])

    plt.tight_layout()
    plt.savefig("dev/splitLAB.png")


def show_split_image_YCrCB(image_paths):
    # Select a random image
    random_path = random.choice(image_paths)
    original = cv2.imread(random_path)
    image = cv2.cvtColor(original, cv2.COLOR_BGR2YCrCb)

    # Split the image into its channels
    channels = cv2.split(image)

    # Prepare the figure
    _, axes = plt.subplots(2, 2, figsize=(10, 10))
    images = [original] + list(channels)

    titles = [random_path, 'Y Channel', 'Cr Channel', 'Cb Channel']

    for i, img in enumerate(images):
        ax = axes[i // 2, i % 2]
        if i == 0:
            ax.imshow(img)  # Original image does not need a colormap
        else:
            ax.imshow(img, cmap='gray')  # Color channels in grayscale
        ax.axis('off')
        ax.set_title(titles[i])

    plt.tight_layout()
    plt.savefig("dev/splitYCrCB.png")
