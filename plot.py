import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data preparation
data = {'Corner\ndetection': 0.013891220092773438, 'Perspective\ncorrection': 0.05194997787475586, 'Missing\nregion\ndetection': 0.017333269119262695, 'Inpainting': 1077.489801645279, 'Median\nThres-\nholding': 1.0106837749481201,
        'Gaussian\nThres-\nholding': 1.122204303741455, 'Non-local\nmeans\ndenoising': 14.109996318817139, 'YCrCb\nThres-\nholding': 14.638205289840698, 'Absolute\nScale\nCorrection': 0.0044384002685546875, 'Gamma\ncorrection': 0.04329538345336914}

# Apply logarithmic transformation
data = {key: np.log10(val) - 2 for key, val in data.items()}

# Group definitions
groups = {
    'De-warping': list(data.keys())[:2],
    'Inpainting': list(data.keys())[2:4],
    'Denoising': list(data.keys())[4:8],
    'Contrast': list(data.keys())[8:10]
}

# Create DataFrame
data_list = [{'Category': label, 'Value': value, 'Group': group}
             for group, items in groups.items()
             for label, value in data.items() if label in items]
df = pd.DataFrame(data_list)

# Seaborn plot
plot = sns.catplot(data=df, kind='bar', x='Category',
                   y='Value', hue='Group', height=6, aspect=2)
plot.set_axis_labels(
    "", "Average processing time for one image (s) (Log Scale)")

# Adjusting the title
# Adjust the top of the subplots to make room for the title
plot.fig.subplots_adjust(top=0.9)

plot._legend.set_bbox_to_anchor((0.95, 0.89))
plt.tight_layout(rect=[0.05, 0, 0.95, 1])
plt.show()
