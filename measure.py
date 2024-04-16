import os
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

SCALING = 1.0 / 255
IMG_SIZE = (256, 256)
MEAN = (0, 0, 0)


def measure(show_dist: bool = False, outpath: str = None):
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
        df.append({'label': label, 'pred': output, 'image': image_name})

        if (output > 0.5):
            if (label == 'p'):
                correct += 1
        else:
            if (label == 'h'):
                correct += 1

    print(correct / len(image_names))
    df = pd.DataFrame(df)

    # Show all wrong predictions
    pneumonia_low_pred = list(
        df[(df['label'] == 'p') & (df['pred'] < 0.5)]['image'])
    healthy_high_pred = list(
        df[(df['label'] == 'h') & (df['pred'] > 0.5)]['image'])

    pneumonia_low_pred = [name[2:5] for name in pneumonia_low_pred]
    healthy_high_pred = [name[2:5] for name in healthy_high_pred]

    print("Wrong pneumonia images")
    print(", ".join(pneumonia_low_pred))
    print("Wrong healthy images")
    print(", ".join(healthy_high_pred))

    # Plotting a stacked histogram
    plt.figure(figsize=(6, 6))

    # Filtering data based on the label
    df_p = df[df['label'] == 'p']['pred']
    df_h = df[df['label'] == 'h']['pred']

    # Stacked histogram without edge color for bars
    plt.hist([df_h, df_p], bins=30, stacked=True, color=[
             'blue', 'red'], label=['Healthy', 'Pneumonia'], alpha=0.7)

    # Adding a vertical dotted line at 0.5
    plt.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5)

    plt.title('Predictions')
    plt.xlabel('Prediction')
    plt.ylabel('Frequency')
    plt.legend(loc='upper center')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath)
    if show_dist:
        plt.show()


if __name__ == '__main__':
    measure()
