import os
import sys
import cv2

from color import *
from util import *
from measure import *
from perspective import *
from blur import *
from filter import *

DEFAULT_IMAGE_DIR = 'image_processing_files/xray_images/'
RESULT_DIR = 'Results/'

def process_images(image_names: list[str], in_dir: str, out_dir: str):
    for _, image_name in enumerate(image_names):
        # Read in the image
        image_path = os.path.join(in_dir, image_name)
        image = cv2.imread(image_path)

        # Processing images
        corners = get_image_corners(image)
        image = perspective_correction(image, corners, 0)
        image = contourRemoval(image, image_name)
        # image = equalizeHist(image)
        # image = bilateral_blur(image)
        # image = N1_means_blur(image)

        # Save the image
        output_path = os.path.join(out_dir, image_name)
        cv2.imwrite(output_path, image)

    full_image_paths = [os.path.join(out_dir, image_name) for image_name in image_names]

    show_random_images(full_image_paths)
    show_random_split_image_gray(full_image_paths)

def contourRemoval(image, image_name: str):
    view = True
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
        axs[3, 0].imshow(cv2.cvtColor(thresh_processed_image, cv2.COLOR_BGR2RGB))
        axs[3, 0].set_title('Fix. Processed Image')
        axs[3, 0].axis('off')

    adap_thresh_processed_image = noiseReduce(image, adap_thresh_small)
    if view:
        # Display the processed image
        axs[3, 1].imshow(cv2.cvtColor(adap_thresh_processed_image, cv2.COLOR_BGR2RGB))
        axs[3, 1].set_title('Adap. Processed Image')
        axs[3, 1].axis('off')

    if view:
        plt.tight_layout()
        plt.savefig("contour.png")
        exit()

    return thresh_processed_image

def threshFilter(thresh, max):
    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to store the noise
    thresh_small = thresh

    # Remove all contours >4 as these are not noise
    for contour in contours:
        if cv2.contourArea(contour) > max:
            cv2.drawContours(thresh_small, [contour], -1, 0, thickness=cv2.FILLED)

    return thresh_small

def noiseReduce(image, thresh):
    noise_areas = thresh == 255

    # Create a copy of the image to apply the selective blur
    processed_image = image.copy()

    # Apply Median Blur only on noise areas identified by the mask
    for i in range(3):
        # Apply the blur selectively based on the mask
        channel = processed_image[:,:,i]
        blurred_channel = cv2.medianBlur(channel, ksize=5)

        # Update the channel only where thresh_small indicates
        channel[noise_areas] = blurred_channel[noise_areas]
        processed_image[:,:,i] = channel

    return processed_image

if __name__ == '__main__':
    # Create the output directory
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Identify the input directory
    image_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_DIR

    # Read in all image names
    images = os.listdir(image_dir)

    # Call main routine and measure the quality
    process_images(images, image_dir, RESULT_DIR)
    measure(show_dist=False, outpath="dist.png")
