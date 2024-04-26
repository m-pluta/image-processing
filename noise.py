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
    print(image_name)
    image_name = image_name.split(".")[0]

    # Setup for displaying the process
    if view:
        _, axes = plt.subplots(4, 5, figsize=(15, 12))

    # # Initialise noise detection
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # processed_image = image.copy()
    # bilateral_image = cv2.bilateralFilter()
    # pepper = threshFilter(gray_image < 130, 8)
    # processed_image[pepper] = bilateral_image[pepper]

    # if view:
    #     # Display the original images
    #     show(rgb(bilateral_image), axes[0, 0], "Median")
    #     show(gray_image, axes[0, 1], f'Original ({image_name})')
    #     show(rgb(image), axes[0, 2], f'Original ({image_name})')

    #     show(pepper.astype(np.uint8), axes[0, 3], "Definitely Pepper")
    #     show(rgb(processed_image), axes[0, 4], "Processed Image 1")

    # # Identify potential pepper noise
    # gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    # bilateral_image = cv2.medianBlur(processed_image, ksize=15)
    # maybe_pepper = threshFilter((gray_image < 160) & ~pepper, 12)
    # processed_image[maybe_pepper] = bilateral_image[maybe_pepper]

    # if view:
    #     # Display the original images
    #     show(rgb(bilateral_image), axes[1, 0], "Median")
    #     show(gray_image, axes[1, 1], f'Original ({image_name})')
    #     show(rgb(processed_image), axes[1, 2], f'Original ({image_name})')

    #     show(maybe_pepper.astype(np.uint8), axes[1, 3], "Definitely Pepper")
    #     show(rgb(processed_image), axes[1, 4], "Processed Image 1")

    # gaussian_image = cv2.GaussianBlur(processed_image, ksize=(5, 5), sigmaX=0)
    # diff = gray(processed_image).astype(np.int8) - \
    #     gray(gaussian_image).astype(np.int8)
    # pep = (diff < -30)
    # salt = diff > 30

    # if view:
    #     show(rgb(gaussian_image), axes[2, 3], "")
    #     show(pep.astype(np.uint8), axes[2, 4], "")

    # processed_image = inpaint(processed_image, pep.astype(np.uint8) * 255)

    # if view:
    #     show(rgb(processed_image), axes[3, 4], "")

   # b_denoised = cv2.medianBlur(b, ksize=5)
   # g_denoised = cv2.medianBlur(g, ksize=5)
   # r_denoised = cv2.medianBlur(r, ksize=5)

    y_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, Cr, Cb = cv2.split(y_image)
    if view:
        show(y, axes[2, 0], 'Y')
        show(Cr, axes[2, 1], 'Cr')
        show(Cb, axes[2, 2], 'Cb')

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
