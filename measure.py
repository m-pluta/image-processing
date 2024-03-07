import os
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

SCALING = 1.0 / 255
IMG_SIZE = (256, 256)
MEAN = (0, 0, 0)


def measure(outpath: str):
    model = cv2.dnn.readNetFromONNX('image_processing_files/classifier.model')

    image_names = os.listdir('Results/')

    correct = 0
    df = []
    
    for image_name in image_names:
        img = cv2.imread(os.path.join('Results/', image_name))

        blob = cv2.dnn.blobFromImage(
            img, SCALING, IMG_SIZE, MEAN, swapRB=True, crop=False)
        model.setInput(blob)
        
        output = model.forward()[0][0]
        
        label = image_name[6]
        df.append({'label': label, 'pred': output})

        if (output > 0.5):
            if (label == 'p'):
                correct += 1
        else:
            if (label == 'h'):
                correct += 1
    
    df = pd.DataFrame(df)
    sns.catplot(x='label', y='pred', kind='swarm', data=df)
    plt.savefig(outpath)

    return correct / len(image_names)


if __name__ == '__main__':
    measure()
