import cv2
import numpy as np
from math import sqrt
from pprint import pprint

H = 1
EPSILON = 0.001

W_MAX = 39
OSET = W_MAX + 1
PAD = [OSET] * 4

BORDER_FILL_COLOR = (0, 0, 0)


def generate_D():
    # Create grids of x and y coordinates representing the distances from the center
    x = np.arange(-W_MAX, W_MAX + 1)
    y = np.arange(-W_MAX, W_MAX + 1)
    xx, yy = np.meshgrid(x, y)

    # Calculate the distances from the center
    distances = np.sqrt(xx**2 + yy**2)

    # Apply the function to calculate the values
    return 1 / (EPSILON + distances) ** 4


D = generate_D()


def IAWMF(image):
    channels = cv2.split(image)

    new_channels = [IAWMF_channel(channel) for channel in channels]

    return cv2.merge(new_channels)


def get_window(x, i, j, w):
    return x[i - w:i + w + 1, j - w:j + w + 1]


def weighted_mean(window, S_ij_min, S_ij_max):
    # All noise free pixels
    mask = (S_ij_min < window) & (window < S_ij_max)

    # Calculate the sum and count
    sum = np.sum(window[mask])
    count = np.sum(mask)

    return sum / count if count else -1


def eps_mean(window, w, S_ij_min, S_ij_max):
    # All noise free pixels
    mask = (S_ij_min < window) & (window < S_ij_max)

    count = np.sum(mask)
    if not count:
        return -1

    window_D = D[W_MAX - w: W_MAX + w + 1, W_MAX - w: W_MAX + w + 1]

    return np.sum(mask * window_D * window) / np.sum(mask * window_D)


def IAWMF_channel(y):
    cv2.imwrite('output.jpg', y)
    # Get shape on input image
    width, height = y.shape

    # Pad the image borders
    padded_y = cv2.copyMakeBorder(y, *PAD, cv2.BORDER_CONSTANT,
                                  value=BORDER_FILL_COLOR)

    # Declare output image
    z = np.zeros((height, width), dtype=np.uint8)

    count = 0
    # Process all pixels in the image
    for i in range(height):
        for j in range(width):
            cv2.imwrite('output.jpg', z)
            # Find a suitable window size
            w = 1
            while w <= W_MAX:
                # Get two successive windows
                window = get_window(padded_y, i + OSET, j + OSET, w)
                next_window = get_window(padded_y, i + OSET, j + OSET, w + H)

                # Calculate values for current window
                S_ij_min = np.min(window)
                S_ij_max = np.max(window)
                S_ij_mean = weighted_mean(window, S_ij_min, S_ij_max)

                # Calculate values for next window
                S_ij_min_next = np.min(next_window)
                S_ij_max_next = np.max(next_window)

                # Check if window if the successive windows have the same min and max grey values
                if S_ij_min == S_ij_min_next and S_ij_max == S_ij_max_next and S_ij_mean != -1:
                    if S_ij_min < y[i][j] < S_ij_max:
                        # Pixel is noise-free
                        z[i][j] = y[i][j]
                        count += 1
                    else:
                        # Pixel is a noise-candidate
                        mean = eps_mean(window, w, S_ij_min, S_ij_max)
                        z[i][j] = mean if mean != -1 else y[i][j]
                    break
                else:
                    w += H
                    if w > W_MAX:
                        # No window found so its a noise candidate
                        mean = eps_mean(window, w, S_ij_min, S_ij_max)
                        z[i][j] = mean if mean != -1 else y[i][j]
    print(count)
    cv2.imwrite('output.jpg', z)
    return z


if __name__ == '__main__':
    # mat = [[255, 255, 146, 107,  59],
    #        [255, 172, 255,   0, 255],
    #        [0,   178, 255, 110, 255],
    #        [192, 187,   0, 255, 255],
    #        [0,     0, 176, 146,  78]]

    # val = eps_mean(np.array(mat), 2, 0, 255)

    # val

    image = cv2.imread('im001-healthy-adjusted.jpg')
    image = IAWMF(image)

    cv2.imwrite('output.jpg', image)
