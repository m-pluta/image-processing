import cv2
from skimage.metrics import structural_similarity as SSIM
import random
import matplotlib.pyplot as plt
import numpy as np


def show_random_images(image_paths):
    GRID_X = 6
    GRID_Y = 3
    # Select random images
    random_paths = random.sample(image_paths, GRID_X * GRID_Y)

    _, axes = plt.subplots(GRID_Y, GRID_X, figsize=(5 * GRID_X, 5 * GRID_Y))

    for i, image_path in enumerate(random_paths):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        axes[i // GRID_X, i % GRID_X].imshow(image, cmap='gray')
        axes[i // GRID_X, i % GRID_X].set_title(image_path)
        axes[i // GRID_X, i % GRID_X].axis('off')

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


def eval(original, image, image_name):
    # Calculate PSNR
    psnr_value = cv2.PSNR(original, image)

    # Convert images to grayscale (if necessary)
    noisy_image_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    denoised_image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate SSIM
    ssim_value = SSIM(noisy_image_gray, denoised_image_gray)

    # Calculate MSE
    mse = np.mean((original - image) ** 2)

    # Calculate MAE
    mae = np.mean(np.abs(original - image))

    # Calculate RMSE
    rmse = np.sqrt(mse)

    # Calculate entropy
    _, noisy_entropy = cv2.meanStdDev(noisy_image_gray)
    _, denoised_entropy = cv2.meanStdDev(denoised_image_gray)

    print(f"Evaluation Metrics - {image_name}")
    print("PSNR:", psnr_value)
    print("SSIM:", ssim_value)
    print("MSE:", mse)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("Noisy Image Entropy:", noisy_entropy[0][0])
    print("Denoised Image Entropy:", denoised_entropy[0][0])

    pass
