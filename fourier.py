import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_fourier_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    return 20*np.log(np.abs(fshift))


def show_fourier_image(image_path):
    f_image = get_fourier_image(image_path)
    plt.imshow(f_image, cmap='gray')
    plt.title('Fourier Transform')
    plt.axis('off')
    plt.show()
