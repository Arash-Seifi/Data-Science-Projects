import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Output Dir
results_dir = 'BeanResult'
os.makedirs(results_dir, exist_ok=True)
df = pd.read_excel('Bean/bean.xlsx')

# Only 200 rows per class
df_sampled = df.groupby('Class').apply(lambda x: x.sample(n=100, random_state=42)).reset_index(drop=True)
X = df_sampled.drop('Class', axis=1)
y = df_sampled['Class']

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Get all possible labels
all_labels = np.unique(y_encoded)

# Evaluation methods
def evaluate_model(model, X, y_encoded, method_name):
    results = []

    # Percentage split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append(f"Percentage Split (20% test):\n{classification_report(y_test, y_pred, target_names=label_encoder.classes_)}")

    # Leave-one-out cross-validation
    loo = LeaveOneOut()
    scores = cross_val_score(model, X, y_encoded, cv=loo)
    results.append(f"Leave-One-Out Cross-Validation Accuracy: {scores.mean()}")

    # 10-fold cross-validation
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    scores = cross_val_score(model, X, y_encoded, cv=kf)
    results.append(f"10-Fold Cross-Validation Accuracy: {scores.mean()}")

    with open(os.path.join(results_dir, f'bean-result-{method_name}.txt'), 'w') as f:
        for result in results:
            f.write(result + '\n')

# Naive Bayes
nb = GaussianNB()
evaluate_model(nb, X, y_encoded, 'naive-bayes')

# KNN (1 to 10 neighbors)
best_k = 1
best_precision = 0
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    evaluate_model(knn, X, y_encoded, f'knn-{k}')

    # Evaluate the precision to find the best k
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    if precision > best_precision:
        best_k = k
        best_precision = precision

with open(os.path.join(results_dir, 'bean-result-knn-best.txt'), 'w') as f:
    f.write(f"Best K for KNN: {best_k} with precision: {best_precision}")

# Random Forest
rf = RandomForestClassifier(random_state=42)
evaluate_model(rf, X, y_encoded, 'random-forest')

# Decision Tree (Rep Tree equivalent)
dt = DecisionTreeClassifier(random_state=42)
evaluate_model(dt, X, y_encoded, 'rep-tree')

# Unsupervised Learning
def cluster_evaluation(model, X, y_encoded, method_name):
    model.fit(X)
    y_pred = model.labels_

    # Map each cluster to the most frequent class label
    label_mapping = {}
    for cluster in np.unique(y_pred):
        cluster_indices = np.where(y_pred == cluster)
        true_labels = y_encoded[cluster_indices]
        most_frequent_label = np.bincount(true_labels).argmax()
        label_mapping[cluster] = most_frequent_label

    y_pred_mapped = np.vectorize(label_mapping.get)(y_pred)
    y_pred_labels = label_encoder.inverse_transform(y_pred_mapped)

    with open(os.path.join(results_dir, f'bean-result-cluster-{method_name}.txt'), 'w') as f:
        f.write(f"Cluster Centers:\n{model.cluster_centers_}\n")
        f.write(f"Confusion Matrix:\n{confusion_matrix(label_encoder.inverse_transform(y_encoded), y_pred_labels, labels=label_encoder.classes_)}\n")
        f.write(f"Classification Report:\n{classification_report(label_encoder.inverse_transform(y_encoded), y_pred_labels, labels=label_encoder.classes_)}\n")

# K-Means Clustering
kmeans = KMeans(n_clusters=len(np.unique(y_encoded)), random_state=42)
cluster_evaluation(kmeans, X, y_encoded, 'k-means')

# Farthest First Clustering (KMeans with max iterations)
ff = KMeans(n_clusters=len(np.unique(y_encoded)), random_state=42, max_iter=1)
cluster_evaluation(ff, X, y_encoded, 'farthest-first')
