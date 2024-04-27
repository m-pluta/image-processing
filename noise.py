import numpy as np
from blur import *
import cv2
import matplotlib.pyplot as plt


def rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def show(image, ax, title):
    ax.imshow(image, cmap='gray', interpolation='nearest')
    ax.set_title(title)
    ax.axis('off')


def remove_noise(image, image_name, view=True):
    image_name = image_name.split(".")[0]

    # Setup for displaying the process
    if view:
        _, axes = plt.subplots(5, 5, figsize=(15, 15))

    # Initialise noise detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_image = image.copy()
    median_image = cv2.medianBlur(processed_image, ksize=15)
    pepper = threshFilter(gray_image < 130, 8)
    processed_image[pepper] = median_image[pepper]

    if view:
        # Display the original images
        show(rgb(median_image), axes[0, 0], "Median")
        show(gray_image, axes[0, 1], f'Original ({image_name})')
        show(rgb(image), axes[0, 2], f'Original ({image_name})')

        show(pepper.astype(np.uint8), axes[0, 3], "Definitely Pepper")
        show(rgb(processed_image), axes[0, 4], "Processed Image 1")

    # Phase 2
    gaussian_image = cv2.GaussianBlur(processed_image, ksize=(5, 5), sigmaX=0)
    diff = gray(processed_image).astype(np.int8) - \
        gray(gaussian_image).astype(np.int8)
    pep = (diff < -30)

    if view:
        show(rgb(gaussian_image), axes[1, 3], "Gaussian")
        show(pep.astype(np.uint8), axes[1, 4], "Gaussian pepper")

    processed_image[pep] = gaussian_image[pep]

    if view:
        show(rgb(processed_image), axes[2, 4], "Processed Image 2")

    b, g, r = cv2.split(processed_image)

    b = b * 1.8
    b = b.astype(np.uint8)

    # r = cv2.convertScaleAbs(r, alpha=0.9, beta=0)

    if view:
        show(b, axes[2, 0], "Blue")
        show(g, axes[2, 1], "Green")
        show(r, axes[2, 2], "Red")

    nlmeans_b = cv2.fastNlMeansDenoising(
        b,
        h=9,
        templateWindowSize=7,
        searchWindowSize=21
    )

    nlmeans_g = cv2.fastNlMeansDenoising(
        g,
        h=11,
        templateWindowSize=7,
        searchWindowSize=21
    )

    nlmeans_r = cv2.fastNlMeansDenoising(
        r,
        h=11,
        templateWindowSize=7,
        searchWindowSize=21
    )

    processed_image = cv2.merge([nlmeans_b, nlmeans_g, nlmeans_r])

    if view:
        show(nlmeans_b, axes[3, 0], "Denoised Blue")
        show(nlmeans_g, axes[3, 1], "Denoised Green")
        show(nlmeans_r, axes[3, 2], "Denoised Red")
        show(rgb(processed_image), axes[3, 3], "Denoised")

    if view:
        plt.tight_layout()
        plt.savefig("dev/contour.png")

    return processed_image


def threshFilter(thresh, max):
    # Create an empty mask to store the noise
    thresh_small = thresh.copy().astype(np.uint8)

    # Find contours in the binary image
    contours, _ = cv2.findContours(
        thresh_small, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Remove all contours >4 as these are not noise
    for contour in contours:
        if cv2.contourArea(contour) > max:
            cv2.drawContours(
                thresh_small, [contour], -1, 0, thickness=cv2.FILLED)

    return thresh_small.astype(bool)
