import timeit
import os
import cv2

SCALING = 1.0 / 255
IMG_SIZE = (256, 256)
MEAN = (0, 0, 0)


def measure():
    model = cv2.dnn.readNetFromONNX('image_processing_files/classifier.model')

    image_names = os.listdir('Results/')

    correct = 0
    for image_name in image_names:
        img = cv2.imread(os.path.join('Results/', image_name))

        blob = cv2.dnn.blobFromImage(
            img, SCALING, IMG_SIZE, MEAN, swapRB=True, crop=False)
        model.setInput(blob)
        output = model.forward()

        if (output > 0.5):
            if (image_name[6] == 'p'):
                correct += 1
        else:
            if (image_name[6] == 'h'):
                correct += 1

    return correct / len(image_names)


if __name__ == '__main__':
    measure()
