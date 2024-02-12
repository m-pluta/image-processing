import cv2


def median_blur(image, ksize=13):
    return cv2.medianBlur(src=image, ksize=ksize)


def N1_means_blur(image):
    return cv2.fastNlMeansDenoisingColored(
        src=image,
        dst=None,
        h=10,
        hColor=10,
        templateWindowSize=7,
        searchWindowSize=21
    )


def gaussian_blur(image, ksize=5, sigmaX=0):
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX)


def bilateral_blur(image, d=9, s=75):
    return cv2.bilateralFilter(image, d, s, s)
