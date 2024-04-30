import os
import cv2
import numpy as np
import pandas as pd
from brisque import BRISQUE
import matplotlib.pyplot as plt

SCALING = 1.0 / 255
IMG_SIZE = (256, 256)
MEAN = (0, 0, 0)


def measure(dir='Results/', debug=False):
    # Models
    model = cv2.dnn.readNetFromONNX('image_processing_files/classifier.model')
    obj = BRISQUE(url=False)

    # Load images
    image_names = sorted(os.listdir(dir))

    # Metrics
    correct = 0     # Number accurate
    se = 0          # Squared Error
    b_score = 0     # Brisque Score

    # Store results of measurements
    df = []

    for image_name in image_names:
        # Read the image
        img = cv2.imread(os.path.join(dir, image_name))

        # Set the model
        blob = cv2.dnn.blobFromImage(
            img, SCALING, IMG_SIZE, MEAN, swapRB=True, crop=False)
        model.setInput(blob)

        # Use the model on the image
        output = model.forward()[0][0]

        # Interpret result
        label = image_name[6]
        df.append({'label': label, 'pred': output, 'image': image_name})

        # Update metrics
        if label == 'p':
            if output > 0.5:
                correct += 1
            se += (1. - output) ** 2
        else:
            if output <= 0.5:
                correct += 1
            se += output ** 2

        score = obj.score(img)
        print(f"{score=}")
        b_score += score

    # Average metrics
    num_images = len(image_names)
    accuracy = correct / num_images
    rmse = np.sqrt(se / num_images)
    b_score /= num_images

    # Convert to dataframe
    df = pd.DataFrame(df)

    # Show all wrong predictions
    pneumonia_low_pred = list(
        df[(df['label'] == 'p') & (df['pred'] < 0.5)]['image'])
    healthy_high_pred = list(
        df[(df['label'] == 'h') & (df['pred'] > 0.5)]['image'])

    pneumonia_low_pred = [name[2:5] for name in pneumonia_low_pred]
    healthy_high_pred = [name[2:5] for name in healthy_high_pred]

    if debug:
        print("Wrong pneumonia images")
        print(", ".join(pneumonia_low_pred))
        print("Wrong healthy images")
        print(", ".join(healthy_high_pred))
        print(f"Accuracy: {accuracy}")
        print(f"MSE: {se}\n")

    showPlot(df)

    return accuracy, rmse, b_score, len(pneumonia_low_pred), len(healthy_high_pred)


def showPlot(df):
    # Plotting a stacked histogram
    plt.figure(figsize=(6, 6))

    # Filtering data based on the label
    df_p = df[df['label'] == 'p']['pred']
    df_h = df[df['label'] == 'h']['pred']

    # Stacked histogram without edge color for bars
    plt.hist([df_h, df_p], bins=25, stacked=True, color=[
             'blue', 'red'], label=['Healthy', 'Pneumonia'], alpha=0.7)

    # Adding a vertical dotted line at 0.5
    plt.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5)

    plt.title('Predictions')
    plt.xlabel('Prediction')
    plt.ylabel('Frequency')
    plt.legend(loc='upper center')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()

    plt.savefig("dev/dist.png")


if __name__ == '__main__':
    measure()
