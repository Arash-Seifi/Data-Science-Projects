import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load the data
data = pd.read_excel('Project_Data.xlsx')

# Exclude the last row
# : : Select all elements (rows in this context).
# -1 : Up to, but not including, the last element (row).
# Except first and last --> data.iloc[1:-1]
data = data.iloc[:-1]

# Convert all columns to numeric (excluding the first column which is names)
# errors='coerce' tells pd.to_numeric to handle errors by converting non-numeric values to NaN (Not a Number). 
data[data.columns[1:]] = data[data.columns[1:]].apply(pd.to_numeric, errors='coerce')

# Step 1: Manage missing and incorrect data
# Replace missing or incorrect data with mode for each feature
# fillna is a method used to fill missing values in a DataFrame : fillna operates on the entire    column at once.
for col in data.columns[1:]:
    # ([0]) is used to access the first element of the Series returned by mode()
    # The mode is the value that appears most frequently in the data.
    mode_val = data[col].mode()[0]
    data[col].fillna(mode_val, inplace=True)

# Drop rows with missing or incorrect data that couldn't be filled
data.dropna(inplace=True)




print("Data after handling missing and incorrect data:\n", data)
# Box plot before handling outliers
plt.figure(figsize=(10, 6))
plt.title('Box plot of data before handling outliers')
sns.boxplot(data=data.iloc[:, 1:])
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('Values')
plt.tight_layout()
plt.show()

# Calculate the value at the 25th percentile of each column in the DataFrame data, excluding the first column.
# Step 2: Identify outliers using IQR method
Q1 = data.iloc[:, 1:].quantile(0.25)
Q3 = data.iloc[:, 1:].quantile(0.75)
print("Quantiles: ",Q1,Q3)
IQR = Q3 - Q1
outliers = (data.iloc[:, 1:] < (Q1 - 2.5 * IQR)) | (data.iloc[:, 1:] > (Q3 + 2.5 * IQR))

print("Outliers identified using IQR method:\n", outliers)

# Step 3: Manage outliers by filling with mode
for col in data.columns[1:]:
    mode_val = data[col].mode()[0]
    data.loc[outliers[col], col] = mode_val

print("Data after handling outliers:\n", data)

# Box plot after handling outliers
plt.figure(figsize=(10, 6))
plt.title('Box plot of data after handling outliers')
sns.boxplot(data=data.iloc[:, 1:])
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('Values')
plt.tight_layout()
plt.show()
