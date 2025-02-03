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

# Output Dir
results_dir = 'WineResult'
os.makedirs(results_dir, exist_ok=True)

df = pd.read_csv('Wine/winequality-red.csv', delimiter=';').head(200)  # only the first 200 rows
# All columns except the Label column
X = df.drop('quality', axis=1)
# LAbel(Last column or result)
y = df['quality']

# Evaluation methods
def evaluate_model(model, X, y, method_name):
    results = []

    # 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append(f"Percentage Split (20% test):\n{classification_report(y_test, y_pred)}")

    # Leave-one-out cross-validation
    loo = LeaveOneOut()
    scores = cross_val_score(model, X, y, cv=loo)
    results.append(f"Leave-One-Out Cross-Validation Accuracy: {scores.mean()}")

    # 10-fold cross-validation
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    scores = cross_val_score(model, X, y, cv=kf)
    results.append(f"10-Fold Cross-Validation Accuracy: {scores.mean()}")

    with open(os.path.join(results_dir, f'wine-result-{method_name}.txt'), 'w') as f:
        for result in results:
            f.write(result + '\n')

# Naive Bayes
nb = GaussianNB()
evaluate_model(nb, X, y, 'naive-bayes')

# KNN (1 to 10 neighbors)
best_k = 1
best_precision = 0
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    evaluate_model(knn, X, y, f'knn-{k}')

    # Evaluate the precision to find the best k
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    if precision > best_precision:
        best_k = k
        best_precision = precision

with open(os.path.join(results_dir, 'wine-result-knn-best.txt'), 'w') as f:
    f.write(f"Best K for KNN: {best_k} with precision: {best_precision}")

# Random Forest
rf = RandomForestClassifier(random_state=42)
evaluate_model(rf, X, y, 'random-forest')

# Decision Tree (Rep Tree equivalent)
dt = DecisionTreeClassifier(random_state=42)
evaluate_model(dt, X, y, 'rep-tree')

# Unsupervised Learning Evaluations
def cluster_evaluation(model, X, y, method_name):
    model.fit(X)
    y_pred = model.labels_

    with open(os.path.join(results_dir, f'wine-result-cluster-{method_name}.txt'), 'w') as f:
        f.write(f"Cluster Centers:\n{model.cluster_centers_}\n")
        f.write(f"Confusion Matrix:\n{confusion_matrix(y, y_pred)}\n")
        f.write(f"Classification Report:\n{classification_report(y, y_pred)}\n")

# K-Means Clustering
kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=42)
cluster_evaluation(kmeans, X, y, 'k-means')

# Farthest First Clustering (KMeans with max iterations)
ff = KMeans(n_clusters=len(np.unique(y)), random_state=42, max_iter=1)
cluster_evaluation(ff, X, y, 'farthest-first')
