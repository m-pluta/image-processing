import cv2
from skimage.metrics import structural_similarity as SSIM
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = getFourierImage(image_path)

        axes[i // GRID, i % GRID].imshow(image, cmap='gray')
        axes[i // GRID, i % GRID].set_title(image_path)
        axes[i // GRID, i % GRID].axis('off')

    plt.tight_layout()
    plt.savefig("dev/view.png")


def show_random_split_image_color(image_paths):

    # Select the random image
    random_path = random.choice(image_paths)
    image = cv2.imread(random_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create empty matrices with zeros for each color channel image
    red_channel = np.zeros_like(image)
    green_channel = np.zeros_like(image)
    blue_channel = np.zeros_like(image)

    # Assign the respective channel to each matrix.
    red_channel[:, :, 0] = image[:, :, 0]
    green_channel[:, :, 1] = image[:, :, 1]
    blue_channel[:, :, 2] = image[:, :, 2]

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
    plt.savefig("dev/split.png")


def show_random_split_image_gray(image_paths):
    # Select a random image
    random_path = random.choice(image_paths)
    image = cv2.imread(random_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Split the image into its channels
    channels = cv2.split(image)

    # Prepare the figure
    _, axes = plt.subplots(2, 2, figsize=(10, 10))
    images = [image] + list(channels)

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
    plt.savefig("dev/split.png")


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
