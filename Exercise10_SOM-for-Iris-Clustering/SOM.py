# Required libraries
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix, accuracy_score

# 1. Loading and checking data
# Load Iris dataset
iris = datasets.load_iris()
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_data['target'] = iris.target
iris_data['target_names'] = iris.target_names[iris.target]

# Initial data review
print("First few rows of data:")
print(iris_data.head())

print("\nData Types:")
print(iris_data.dtypes)

print("\nMissing values check:")
print(iris_data.isnull().sum())

print("\nData distribution:")
print(iris_data.describe())

# 2. Data Preprocessing
# Data standardization
scaler = StandardScaler()
iris_scaled = scaler.fit_transform(iris_data.iloc[:, :4])  # Standardize only feature columns

# 3. Design and implementation of SOM
# Define SOM parameters
som_x, som_y = 10, 10  # Grid dimensions
learning_rate = 0.5
neighborhood_radius = 1
num_iterations = 100  # You can increase this for better results

# Initialize SOM
som = MiniSom(x=som_x, y=som_y, input_len=4, sigma=neighborhood_radius, learning_rate=learning_rate)
som.random_weights_init(iris_scaled)

# Training process
print("Training SOM...")
som.train_random(data=iris_scaled, num_iteration=num_iterations)

# 4. Data clustering using SOM
# Assign each data point to its closest neuron
win_map = som.win_map(iris_scaled)

# Assign each data sample to a neuron
iris_clusters = [som.winner(d) for d in iris_scaled]
iris_data['cluster'] = [str(x[0]) + "-" + str(x[1]) for x in iris_clusters]

# 5. Analysis and visualization of results
# Plotting SOM clusters
plt.figure(figsize=(10, 8))
for i, (x, y) in enumerate(iris_clusters):
    plt.scatter(iris_data.iloc[i, 0], iris_data.iloc[i, 1], c='C'+str(iris.target[i]), label=iris_data.target_names[i])
plt.title("SOM Clusters (Iris Dataset)")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.grid()
plt.show()

# Checking the number of samples in each cluster
cluster_counts = iris_data['cluster'].value_counts()
print("Number of samples in each cluster:")
print(cluster_counts)

# Evaluation of clustering quality
# Mapping clusters to target names
target_names = list(iris.target_names)
clusters = iris_data['cluster'].unique()

# Map clusters to original labels
label_map = {cluster: iris_data[iris_data['cluster'] == cluster]['target_names'].mode()[0] for cluster in clusters}
predicted_labels = [target_names.index(label_map[cluster]) for cluster in iris_data['cluster']]

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(iris.target, predicted_labels)
conf_matrix = confusion_matrix(iris.target, predicted_labels)

print("\nClustering Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(conf_matrix)
