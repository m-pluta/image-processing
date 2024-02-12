import os
import cv2

def measure():
    model =  cv2.dnn.readNetFromONNX('image_processing_files/classifier.model')

    names = []
    healthys = []
    pneumonias = []

    # the first 50 images are healthy and the next 50 are not:
    for i in range(1, 51):
        healthys.append(f'im{str(i).zfill(3)}')

    for i in range(51, 101):
        pneumonias.append(f'im{str(i).zfill(3)}')

    # read all the images from the directory

    for file in os.listdir('Results/'):
        names.append(file)
    names.sort()

    # keeping track of the number of correct predictions for accuracy:
    correct = 0

    # main loop:
    for filename in names:

        # read image:
        img = cv2.imread(os.path.join('Results/', filename))

        if img is not None:

            # pass the image through the neural network:
            blob = cv2.dnn.blobFromImage(img, 1.0 / 255, (256, 256),(0, 0, 0), swapRB=True, crop=False)
            model.setInput(blob)
            output = model.forward()

            # identify what the predicted label is:
            if(output > 0.5):
                # print(f'{filename}: pneumonia')
                if(filename.startswith(tuple(pneumonias))):
                    correct += 1
            else:
                # print(f'{filename}: healthy')
                if(filename.startswith(tuple(healthys))):
                    correct += 1

    return correct / len(names)