import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist

# Load dataset
data = pd.read_csv('data.csv')

# 1. Data Preprocessing

# a. Incomplete Data Management
data.fillna(data.mean(), inplace=True)

# b. Coding of Hand Features
encoder = OneHotEncoder(sparse_output=False)
study_env_encoded = encoder.fit_transform(data[['StudyEnvironment']])
study_env_df = pd.DataFrame(study_env_encoded, columns=encoder.get_feature_names_out(['StudyEnvironment']))
data = pd.concat([data, study_env_df], axis=1)
data.drop(['StudyEnvironment'], axis=1, inplace=True)

# c. Data Normalization
scaler = StandardScaler()
numerical_features = ['StudyHours', 'Attendance', 'HomeworkScores', 'ProjectScores', 'MidtermScores', 'ExtracurricularActivities']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# 2. Custom Distance Function Definition
def custom_distance(x, y, weights):
    return np.sum(weights * np.abs(x - y))

# 3. Choosing the Optimal Number of Clusters (K)
K_range = range(1, 10)

# Method 1: Elbow Method
wcss = []
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data.drop(['StudentID'], axis=1))
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(K_range, wcss, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters K')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid()
plt.show()

# Method 2: Silhouette Method
silhouette_scores = []
for k in K_range[1:]:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(data.drop(['StudentID'], axis=1))
    silhouette_scores.append(silhouette_score(data.drop(['StudentID'], axis=1), cluster_labels))

# Plot Silhouette Scores
plt.figure(figsize=(10, 6))
plt.plot(K_range[1:], silhouette_scores, marker='o')
plt.title('Silhouette Method for Optimal K')
plt.xlabel('Number of Clusters K')
plt.ylabel('Silhouette Score')
plt.grid()
plt.show()

# Method 3: Gap Statistic
def calculate_gap_statistic(data, n_refs=20):
    gaps = []
    for k in K_range[1:]:
        # Fit K-means
        kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
        original_dispersion = sum(np.min(cdist(data, kmeans.cluster_centers_, 'euclidean'), axis=1))
        
        # Generate reference datasets and calculate dispersions
        reference_dispersion = np.zeros(n_refs)
        for i in range(n_refs):
            random_reference = np.random.uniform(low=np.min(data, axis=0), high=np.max(data, axis=0), size=data.shape)
            kmeans_reference = KMeans(n_clusters=k, random_state=42).fit(random_reference)
            reference_dispersion[i] = sum(np.min(cdist(random_reference, kmeans_reference.cluster_centers_, 'euclidean'), axis=1))
        
        # Calculate the gap statistic
        gap = np.log(np.mean(reference_dispersion)) - np.log(original_dispersion)
        gaps.append(gap)
    return gaps

gaps = calculate_gap_statistic(data.drop(['StudentID'], axis=1))

# Plot Gap Statistic
plt.figure(figsize=(10, 6))
plt.plot(K_range[1:], gaps, marker='o')
plt.title('Gap Statistic for Optimal K')
plt.xlabel('Number of Clusters K')
plt.ylabel('Gap Statistic')
plt.grid()
plt.show()

# Method 4: Dunn Index
def calculate_dunn_index(data, labels, centroids):
    intercluster_distances = euclidean_distances(centroids)
    np.fill_diagonal(intercluster_distances, np.inf)  # Ignore zero distances to self
    min_intercluster_distance = np.min(intercluster_distances)

    intracluster_distances = []
    for label in np.unique(labels):
        cluster_points = data[labels == label]
        cluster_diameter = np.max(euclidean_distances(cluster_points))
        intracluster_distances.append(cluster_diameter)
    max_intracluster_distance = np.max(intracluster_distances)

    return min_intercluster_distance / max_intracluster_distance

dunn_indices = []
for k in K_range[1:]:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(data.drop(['StudentID'], axis=1))
    centroids = kmeans.cluster_centers_
    dunn_index = calculate_dunn_index(data.drop(['StudentID'], axis=1).to_numpy(), cluster_labels, centroids)
    dunn_indices.append(dunn_index)

# Plot Dunn Index
plt.figure(figsize=(10, 6))
plt.plot(K_range[1:], dunn_indices, marker='o')
plt.title('Dunn Index for Optimal K')
plt.xlabel('Number of Clusters K')
plt.ylabel('Dunn Index')
plt.grid()
plt.show()

# Choosing Optimal K
optimal_k = 3  # Based on the combined results of all methods
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(data.drop(['StudentID'], axis=1))
data['Cluster'] = cluster_labels

# 4. PCA Visualization
pca = PCA(n_components=2)
pca_results = pca.fit_transform(data.drop(['StudentID', 'Cluster'], axis=1))

# Plot PCA results
plt.figure(figsize=(10, 6))
plt.scatter(pca_results[:, 0], pca_results[:, 1], c=data['Cluster'], cmap='viridis')
plt.title('PCA of Student Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.grid()
plt.show()
