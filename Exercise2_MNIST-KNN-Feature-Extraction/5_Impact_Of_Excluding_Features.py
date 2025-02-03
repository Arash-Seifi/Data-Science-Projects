import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data_with_features = np.load("mnist_image_vectors_with_features.npy")
X = data_with_features[:, :-1]
y = data_with_features[:, -1]

# 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# K-NN classifier with k=1
knn = KNeighborsClassifier(n_neighbors=1)

# Perform 10-fold cross-validation for each feature combination
for excluded_feature in range(5):
    included_features = [i for i in range(5) if i != excluded_feature]
    feature_data = X[:, included_features]

    fold_accuracy = []
    for train_index, test_index in kf.split(feature_data):
        X_train, X_test = feature_data[train_index], feature_data[test_index]
        y_train, y_test = y[train_index], y[test_index]
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        # Calculate accuracy for this fold
        accuracy = accuracy_score(y_test, y_pred)
        fold_accuracy.append(accuracy)

    # Calculate and print the average accuracy
    average_accuracy = np.mean(fold_accuracy)
    print(f"Excluded Feature {excluded_feature + 1} - Average Accuracy: {average_accuracy:.4f}")
