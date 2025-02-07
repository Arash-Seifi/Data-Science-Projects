import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

data = pd.read_csv('data.csv')

# 1. Data Preprocessing

# a. Handling Missing Values
data.fillna(data.mean(), inplace=True)

# b. One-Hot Encoding of Categorical Features
encoder = OneHotEncoder(sparse_output=False)
study_env_encoded = encoder.fit_transform(data[['StudyEnvironment']])
study_env_df = pd.DataFrame(study_env_encoded, columns=encoder.get_feature_names_out(['StudyEnvironment']))
data = pd.concat([data, study_env_df], axis=1)
data.drop(['StudyEnvironment'], axis=1, inplace=True)

# c. Feature Scaling
scaler = StandardScaler()
numerical_features = ['StudyHours', 'Attendance', 'HomeworkScores', 'ProjectScores', 'MidtermScores', 'ExtracurricularActivities']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# 2. Custom Distance Function
def custom_distance(x, y, weights=None):
    if weights is None:
        weights = np.ones_like(x)
    return np.sum(weights * np.abs(x - y))

# 3. Custom KMeans Implementation with Custom Distance
def custom_kmeans(data, k, max_iter=100, tol=1e-4):
    n_samples, n_features = data.shape
    weights = np.ones(n_features)  # Equal weights for simplicity
    
    # Initialize centroids randomly
    np.random.seed(42)
    centroids = data[np.random.choice(n_samples, k, replace=False)]
    
    for _ in range(max_iter):
        # Step 1: Assign clusters
        labels = np.argmin([[custom_distance(point, centroid, weights) for centroid in centroids] for point in data], axis=1)
        
        # Step 2: Update centroids
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        
        # Convergence check
        if np.max(np.abs(new_centroids - centroids)) < tol:
            break
        
        centroids = new_centroids
    
    return labels, centroids

# 4. Determine Optimal Number of Clusters (K)
K_range = range(1, 10)
wcss = []

for k in K_range:
    labels, centroids = custom_kmeans(data.drop(['StudentID'], axis=1).to_numpy(), k)
    wcss.append(np.sum([custom_distance(data.iloc[i, :-1].to_numpy(), centroids[label]) for i, label in enumerate(labels)]))

optimal_k = 3 
data_array = data.drop(['StudentID'], axis=1).to_numpy()
labels, centroids = custom_kmeans(data_array, optimal_k)
data['Cluster'] = labels

pca = PCA(n_components=2)
pca_results = pca.fit_transform(data.drop(['StudentID', 'Cluster'], axis=1))

plt.figure(figsize=(10, 6))
plt.scatter(pca_results[:, 0], pca_results[:, 1], c=data['Cluster'], cmap='viridis')
plt.title('PCA Visualization with Custom Distance Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.grid()
plt.show()
