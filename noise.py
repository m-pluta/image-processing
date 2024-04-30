import numpy as np
import cv2


def gray(image: np.ndarray):
    """Convert an image in BGR color space to grayscale

    Args:
        image (np.ndarray): Image in BGR color space

    Returns:
        np.ndarray: Grayscale image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def np_int(image: np.ndarray):
    """Convert a numpy array to 8 bit integer type

    Args:
        image (np.ndarray): Input array

    Returns:
        np.ndarray: 8 bit integer type array
    """
    return image.astype(np.int8)


def remove_noise(image: np.ndarray):
    """Performs multiple types of noise removal on an input `image`

    Args:
        image (np.ndarray): Input image

    Returns:
        np.ndarray: Image with noise removed
    """
    processed_image = image.copy()

    # Phase 1 - Median Thresholding
    gray_image = gray(image)
    median_image = cv2.medianBlur(processed_image, ksize=15)
    pepper = threshFilter(gray_image < 130, 8)
    processed_image[pepper] = median_image[pepper]

    # Phase 2 - Gaussian Thresholding
    gaussian_image = cv2.GaussianBlur(processed_image, ksize=(5, 5), sigmaX=0)
    diff = np_int(gray(processed_image)) - np_int(gray(gaussian_image))
    pep = (diff < -30)
    processed_image[pep] = gaussian_image[pep]

    # Phase 3 - Non-local means denoising
    b, g, r = cv2.split(processed_image)
    b = cv2.fastNlMeansDenoising(
        b,
        h=9,
        templateWindowSize=7,
        searchWindowSize=21
    )
    g = cv2.fastNlMeansDenoising(
        g,
        h=9,
        templateWindowSize=7,
        searchWindowSize=21
    )
    r = cv2.fastNlMeansDenoising(
        r,
        h=11,
        templateWindowSize=7,
        searchWindowSize=21
    )
    processed_image = cv2.merge([b, g, r])

    # Phase 4 - YCrCb Thresholding
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(processed_image)

    median_y = cv2.medianBlur(y, ksize=9)
    diff = np_int(y) - np_int(median_y)
    noise = threshFilter(diff < -25, 4)
    y[noise] = median_y[noise]

    processed_image = cv2.merge([y, cr, cb])
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_YCrCb2BGR)

    return processed_image


def threshFilter(thresh: np.ndarray, maxArea: int):
    """Given an input threshold image, filters out any contours with an area greater than `maxArea`

    Args:
        thresh (np.ndarray): Input threshold image
        maxArea (int): Maximum area of contours

    Returns:
        np.ndarray: Filtered Threshold image
    """
    # Create an empty mask to store the noise
    thresh_small = thresh.copy().astype(np.uint8)

    # Find contours in the binary image
    contours, _ = cv2.findContours(
        thresh_small, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Remove all contours >4 as these are not noise
    for contour in contours:
        if cv2.contourArea(contour) > maxArea:
            cv2.drawContours(
                thresh_small, [contour], -1, 0, thickness=cv2.FILLED)

    return thresh_small.astype(bool)
