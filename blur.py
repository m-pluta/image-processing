import cv2


def median_blur(image, ksize=3):
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
