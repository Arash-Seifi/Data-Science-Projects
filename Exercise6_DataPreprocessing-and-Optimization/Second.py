import pandas as pd
import numpy as np

# Load the data
data = pd.read_excel('Project_Data.xlsx')
data = data.iloc[:-1]

# Convert all columns to numeric
data[data.columns[1:]] = data[data.columns[1:]].apply(pd.to_numeric, errors='coerce') 
# Replace non-numeric values with NaN
data.fillna(np.nan, inplace=True)

# Manage missing and incorrect data
# Replace missing or incorrect data with mode for each feature
for col in data.columns[1:]:
    mode_val = data[col].mode()[0]
    data[col].fillna(mode_val, inplace=True)

# Drop rows with missing or incorrect data that couldn't be filled
data.dropna(inplace=True)

# Normalize data
# mean --> AVERAGE
# median --> Middle
normalized_data = (data.iloc[:, 1:] - data.iloc[:, 1:].mean()) / data.iloc[:, 1:].std()

# Add the names column back
# The first argument 0 specifies the position at which the new column should be inserted. Here, 0 means the new column will be inserted as the first column.
# inserts the first column (both its name(data.iloc[:, 0]) and its data(data.columns[0])) from the original DataFrame back into the 
normalized_data.insert(0, data.columns[0], data.iloc[:, 0])

print("Normalized Data:\n", normalized_data)
