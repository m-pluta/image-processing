import pprint
import numpy as np
from blur import *
import cv2
import matplotlib.pyplot as plt


def rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def show(image, ax, title):
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    ax.axis('off')


def contourRemoval(image, image_name, view=True):
    b, g, r = cv2.split(image)

    if view:
        # Setup for displaying the process
        _, axes = plt.subplots(4, 3, figsize=(9, 12))

        # Display the original coloured image
        show(rgb(image), axes[0, 1], f'Original ({image_name.split(".")[0]})')
    if view:
        show(b, axes[1, 0], 'Blue')
        show(g, axes[1, 1], 'Green')
        show(r, axes[1, 2], 'Red')

    b = cv2.fastNlMeansDenoising(
        src=b,
        h=15,
        templateWindowSize=7,
        searchWindowSize=21
    )
    if view:
        show(b, axes[2, 0], "Blue Denoised")

    thresh_g = cv2.adaptiveThreshold(
        g,                  # Source image
        255,                # MaxVal: The maximum intensity for the white color
        cv2.ADAPTIVE_THRESH_MEAN_C,  # Adaptive method: Mean or Gaussian
        cv2.THRESH_BINARY,  # Threshold type
        11,                 # Block size: Size of the neighborhood area used to calculate the threshold for each pixel
        50                  # C: Constant subtracted from the calculated mean or weighted mean
    )
    thresh_g = 255 - thresh_g
    thresh_g = threshFilter(thresh_g, 12)

    thresh_g = gaussian_blur(thresh_g, 5)

    if view:
        show(thresh_g, axes[2, 1], "Green Threshold")

    g = g.astype(np.uint16) + thresh_g.astype(np.uint16)

    g[g > 255] = 255
    g = g.astype(np.uint8)

    g = cv2.fastNlMeansDenoising(
        src=g,
        h=15,
        templateWindowSize=7,
        searchWindowSize=21
    )

    if view:
        show(g, axes[3, 1], "Green Denoised")

    thresh_r = cv2.adaptiveThreshold(
        r,                  # Source image
        255,                # MaxVal: The maximum intensity for the white color
        cv2.ADAPTIVE_THRESH_MEAN_C,  # Adaptive method: Mean or Gaussian
        cv2.THRESH_BINARY,  # Threshold type
        11,                 # Block size: Size of the neighborhood area used to calculate the threshold for each pixel
        50                  # C: Constant subtracted from the calculated mean or weighted mean
    )
    thresh_r = 255 - thresh_r
    thresh_r = threshFilter(thresh_r, 5)

    thresh_r = gaussian_blur(thresh_r, 5)

    if view:
        show(thresh_r, axes[2, 2], "Red Threshold")

    r = r.astype(np.uint16) + thresh_r.astype(np.uint16)

    r[r > 255] = 255
    r = r.astype(np.uint8)

    r = cv2.fastNlMeansDenoising(
        src=r,
        h=15,
        templateWindowSize=7,
        searchWindowSize=21
    )

    if view:
        show(r, axes[3, 2], "Red Denoised")

    if view:
        plt.tight_layout()
        plt.savefig("dev/contour.png")
        exit()

    processed_image = cv2.merge((b, g, r))

    return processed_image


def contourRemovalOld(image, image_name: str, view=False):
    if view:
        # Setup for displaying the process
        _, axs = plt.subplots(4, 2, figsize=(6, 12))

        # Display the original coloured image
        colour = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axs[0, 0].imshow(colour)
        axs[0, 0].set_title(f'Original ({image_name.split(".")[0]})')
        axs[0, 0].axis('off')

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if view:
        # Display the original grayscale image
        axs[0, 1].imshow(gray, cmap='gray')
        axs[0, 1].set_title('Original (Grayscale)')
        axs[0, 1].axis('off')

    # Threshold the grayscale image to get a binary image
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_not(thresh)
    if view:
        # Display the threshold
        axs[1, 0].imshow(thresh, cmap='gray')
        axs[1, 0].set_title('Fixed Threshold')
        axs[1, 0].axis('off')

    adap_thresh = cv2.adaptiveThreshold(
        gray,               # Source image
        255,                # MaxVal: The maximum intensity for the white color
        cv2.ADAPTIVE_THRESH_MEAN_C,  # Adaptive method: Mean or Gaussian
        cv2.THRESH_BINARY,  # Threshold type
        11,                 # Block size: Size of the neighborhood area used to calculate the threshold for each pixel
        50                   # C: Constant subtracted from the calculated mean or weighted mean
    )
    adap_thresh = cv2.bitwise_not(adap_thresh)
    if view:
        # Display the threshold
        axs[1, 1].imshow(adap_thresh, cmap='gray')
        axs[1, 1].set_title('Adaptive Threshold')
        axs[1, 1].axis('off')

    thresh_small = threshFilter(thresh, 4)
    if view:
        # Display the updated threshold
        axs[2, 0].imshow(thresh_small, cmap='gray')
        axs[2, 0].set_title('Fix. Threshold - Small contours')
        axs[2, 0].axis('off')

    adap_thresh_small = threshFilter(adap_thresh, 4)
    if view:
        # Display the updated threshold
        axs[2, 1].imshow(adap_thresh_small, cmap='gray')
        axs[2, 1].set_title('Adap. Threshold - Small contours')
        axs[2, 1].axis('off')

    thresh_processed_image = noiseReduce(image, thresh_small)
    if view:
        # Display the processed image
        axs[3, 0].imshow(cv2.cvtColor(
            thresh_processed_image, cv2.COLOR_BGR2RGB))
        axs[3, 0].set_title('Fix. Processed Image')
        axs[3, 0].axis('off')

    adap_thresh_processed_image = noiseReduce(image, adap_thresh_small)
    if view:
        # Display the processed image
        axs[3, 1].imshow(cv2.cvtColor(
            adap_thresh_processed_image, cv2.COLOR_BGR2RGB))
        axs[3, 1].set_title('Adap. Processed Image')
        axs[3, 1].axis('off')

    if view:
        plt.tight_layout()
        plt.savefig("dev/contour.png")
        exit()

    return thresh_processed_image


def threshFilter(thresh, max):
    # Find contours in the binary image
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to store the noise
    thresh_small = thresh

    # Remove all contours >4 as these are not noise
    for contour in contours:
        if cv2.contourArea(contour) > max:
            cv2.drawContours(
                thresh_small, [contour], -1, 0, thickness=cv2.FILLED)

    return thresh_small


def noiseReduce(image, thresh):
    noise_areas = thresh == 255

    # Create a copy of the image to apply the selective blur
    processed_image = image.copy()

    # Apply Median Blur only on noise areas identified by the mask
    for i in range(3):
        # Apply the blur selectively based on the mask
        channel = processed_image[:, :, i]
        blurred_channel = cv2.medianBlur(channel, ksize=5)

        # Update the channel only where thresh_small indicates
        channel[noise_areas] = blurred_channel[noise_areas]
        processed_image[:, :, i] = channel

    return processed_image
