import pandas as pd

# Load the data
data = pd.read_excel('Project_Data.xlsx')

data = data.iloc[:-1]
data = data.drop(columns=data.columns[0])

# Calculate correlation matrix
correlation_matrix = data.corr()

# Exclude the label column from the correlation matrix calculations
correlation_matrix = correlation_matrix.drop(index=['Total'], columns=['Total'])
print(correlation_matrix)
# Initialize list to store removed features
removed_features = []

# Repeat steps until two features are removed
while len(removed_features) < 2:
    # Find the two features pair with the highest correlation
    ''' 
    .stack() converts the DataFrame into a Series by stacking the columns on top of each other. This effectively creates a multi-index Series where the index is a pair of feature names and the value is the absolute correlation between those features.
    .idxmax() returns the index of the maximum value in the Series created by .stack(). '''
    max_corr = correlation_matrix.abs().stack().idxmax()
    feature1, feature2 = max_corr

    # Calculate correlation of both features with the 'Total' column from the original data
    # Calculates the correlation between feature1 and the sll DATA.
    corr_with_label_feature1 = data[feature1].corr(data['Total'])
    corr_with_label_feature2 = data[feature2].corr(data['Total'])

    # Remove the feature with lower correlation with the 'Total' column
    # lower correlation -> less important and is removed.
    if corr_with_label_feature1 < corr_with_label_feature2:
        removed_feature = feature1
    else:
        removed_feature = feature2

    # Remove the feature from the correlation matrix and add to removed features list
    correlation_matrix.drop(index=removed_feature, columns=removed_feature, inplace=True)
    removed_features.append(removed_feature)

print("Removed features:", removed_features)
