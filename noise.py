import numpy as np
import cv2


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_noise(image):
    # Phase 1 - Median Thresholding
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_image = image.copy()
    median_image = cv2.medianBlur(processed_image, ksize=15)
    pepper = threshFilter(gray_image < 130, 8)
    processed_image[pepper] = median_image[pepper]

    # Phase 2 - Gaussian Thresholding
    gaussian_image = cv2.GaussianBlur(processed_image, ksize=(5, 5), sigmaX=0)
    diff = gray(processed_image).astype(np.int8) - \
        gray(gaussian_image).astype(np.int8)
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
    y, cr, cb = cv2.split(cv2.cvtColor(processed_image, cv2.COLOR_BGR2YCrCb))

    median_y = cv2.medianBlur(y, ksize=9)
    diff = y.astype(np.int8) - median_y.astype(np.int8)
    noise = threshFilter(diff < -25, 4)

    y[noise] = median_y[noise]

    processed_image = cv2.merge([y, cr, cb])
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_YCrCb2BGR)

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
