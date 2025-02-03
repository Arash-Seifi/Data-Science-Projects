import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the data with features and labels
data_with_features = np.load("mnist_image_vectors_with_features.npy")

# Extract features (all columns except the last one) and labels (last column)
X = data_with_features[:, :-1]
y = data_with_features[:, -1]

# Set up 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize K-NN classifier with k=1
knn = KNeighborsClassifier(n_neighbors=1)

# Perform 10-fold cross-validation for each feature
for feature_index in range(5):
    feature_data = X[:, -5 + feature_index].reshape(-1, 1)  # Select one feature at a time

    fold_accuracy = []
    for train_index, test_index in kf.split(feature_data):
        X_train, X_test = feature_data[train_index], feature_data[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit the classifier on the training data
        knn.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = knn.predict(X_test)

        # Calculate accuracy for this fold
        accuracy = accuracy_score(y_test, y_pred)
        fold_accuracy.append(accuracy)

    # Calculate and print the average accuracy across all folds for the current feature
    average_accuracy = np.mean(fold_accuracy)
    print(f"Feature {feature_index + 1} - Average Accuracy: {average_accuracy:.4f}")
