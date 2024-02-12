import cv2


def equalizeHist_LAB(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)

    l_eq = cv2.equalizeHist(l)

    lab_image_eq = cv2.merge((l_eq, a, b))
    image_eq = cv2.cvtColor(lab_image_eq, cv2.COLOR_LAB2BGR)

    return image_eq


def equalizeHist_HSV(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)

    v_eq = cv2.equalizeHist(v)

    hsv_image_eq = cv2.merge((h, s, v_eq))
    image_eq = cv2.cvtColor(hsv_image_eq, cv2.COLOR_HSV2BGR)

    return image_eq


def equalizeHist_YUV(image):
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv_image)

    y_eq = cv2.equalizeHist(y)

    yuv_image_eq = cv2.merge((y_eq, u, v))
    image_eq = cv2.cvtColor(yuv_image_eq, cv2.COLOR_YUV2BGR)

    return image_eq


def equalizeHist(image, mode='LAB'):
    if mode == 'LAB':
        return equalizeHist_LAB(image)
    elif mode == 'HSV':
        return equalizeHist_HSV(image)
    elif mode == 'YUV':
        return equalizeHist_YUV(image)
    else:
        raise ValueError(
            "Invalid mode. Supported modes are: LAB, HSV, YUV")
