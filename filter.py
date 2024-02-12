import cv2
import numpy as np


def butterworth_lowpass_filter(image, cutoff, order):
    rows, cols = image.shape
    x, y = np.ogrid[:rows, :cols]
    center = (rows / 2, cols / 2)
    distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    filter_mask = 1 / (1 + (distance / cutoff)**(2 * order))
    return filter_mask


def butterworth(image):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Create Butterworth low-pass filter mask
    cutoff = 30  # Cutoff frequency
    order = 2    # Order of the filter
    butterworth_filter = butterworth_lowpass_filter(image, cutoff, order)

    # Apply filter
    filtered = dft_shift * butterworth_filter[:, :, None]

    # Shift back and inverse DFT
    f_ishift = np.fft.ifftshift(filtered)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Normalize and convert to uint8
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    blurred_image = np.uint8(img_back)

    return blurred_image
