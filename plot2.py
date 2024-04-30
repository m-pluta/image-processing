import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel("Data.xlsx")
# Cleaning up the data
# Dropping the unnecessary initial columns and rows, and setting appropriate headers
# Drop empty columns if exist
data_cleaned = data.drop(columns=["Unnamed: 0", "Unnamed: 1"], errors='ignore')
data_cleaned.columns = data_cleaned.iloc[0]  # Set the first row as the header
# Drop the first row now that it's set as header
data_cleaned = data_cleaned.drop(data_cleaned.index[0])

# Convert relevant columns to numerical types
numerical_cols = ["Accuracy", "RMSE", "BRISQUE", "Type2", "Type1"]
data_cleaned[numerical_cols] = data_cleaned[numerical_cols].apply(
    pd.to_numeric, errors='coerce')

# Show the cleaned data
data_cleaned.reset_index(drop=True, inplace=True)


# Adjusting the subplot to correctly align the x labels with the x ticks

# Replotting with adjusted x-tick alignment
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5.5))

# Plotting Accuracy
axes[0].plot(data_cleaned['Processing Phase'], data_cleaned['Accuracy'],
             marker='o', linestyle='-', color='skyblue')
axes[0].set_title('Accuracy by Processing Phase')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_xticks(data_cleaned['Processing Phase'])
axes[0].set_xticklabels(data_cleaned['Processing Phase'],
                        rotation=45, ha="right", rotation_mode='anchor')
axes[0].axhline(y=0.95, color='grey', linestyle='--', linewidth=1)
axes[0].text(-0.2, 0.96, '95%', color='grey')

# Plotting RMSE
axes[1].plot(data_cleaned['Processing Phase'], data_cleaned['RMSE'],
             marker='o', linestyle='-', color='salmon')
axes[1].set_title('RMSE by Processing Phase')
axes[1].set_ylabel('RMSE')
axes[1].set_xticks(data_cleaned['Processing Phase'])
axes[1].set_xticklabels(data_cleaned['Processing Phase'],
                        rotation=45, ha="right", rotation_mode='anchor')

# Adjust layout
plt.tight_layout()

plt.show()
