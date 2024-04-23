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
    # Setup for displaying the process
    if view:
        _, axes = plt.subplots(4, 4, figsize=(12, 12))

    # Initialise noise detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(image)
    processed_image = image.copy()

    # Identify definite pepper noise
    median_image = cv2.medianBlur(image, ksize=15)
    pepper = threshFilter(gray_image < 130, 8)
    processed_image[pepper] = median_image[pepper]

    # Identify potential pepper noise
    median_image = cv2.medianBlur(image, ksize=5)
    maybe_pepper = threshFilter((gray_image < 160) & ~pepper, 12)
    processed_image[maybe_pepper] = median_image[maybe_pepper]

    if view:
        # Display the original images
        show(gray(image), axes[0, 0], f'Original ({image_name.split(".")[0]})')
        show(rgb(image), axes[0, 1], f'Original ({image_name.split(".")[0]})')
        show(rgb(median_image), axes[0, 2], "Median")

        show(pepper.astype(np.uint8),
             axes[1, 0], "Definitely Pepper")
        show(maybe_pepper.astype(np.uint8), axes[1, 1], "Maybe Pepper")

        show(rgb(processed_image), axes[2, 3], "Processed")

    if view:
        show(b, axes[2, 0], 'Blue')
        show(g, axes[2, 1], 'Green')
        show(r, axes[2, 2], 'Red')

    b_denoised = cv2.fastNlMeansDenoising(b, h=10, templateWindowSize=7, searchWindowSize=21)
    g_denoised = cv2.fastNlMeansDenoising(g, h=20, templateWindowSize=7, searchWindowSize=21)
    r_denoised = cv2.fastNlMeansDenoising(r, h=20, templateWindowSize=7, searchWindowSize=21)
    
    # b_denoised = cv2.medianBlur(b, ksize=5)
    # g_denoised = cv2.medianBlur(g, ksize=5)
    # r_denoised = cv2.medianBlur(r, ksize=5)

    denoised_image = cv2.merge([b_denoised, g_denoised, r_denoised])

    if view:
        show(b_denoised, axes[3, 0], 'Blue Denoised')
        show(g_denoised, axes[3, 1], 'Green Denoised')
        show(r_denoised, axes[3, 2], 'Red Denoised')
        show(rgb(denoised_image), axes[3, 3], 'Denoised Image')

    # if view:
    #     show(y, axes[3, 0], 'Y')
    #     show(Cr, axes[3, 1], 'Cr')
    #     show(Cb, axes[3, 2], 'Cb')

    if view:
        plt.tight_layout()
        plt.savefig("dev/contour.png")
        exit()

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
