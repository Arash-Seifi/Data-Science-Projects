import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data_with_labels = np.load("mnist_image_vectors_with_labels.npy")
np.random.shuffle(data_with_labels)

# Extract features (all columns except the last one) and labels (last column)
X = data_with_labels[:, :-1]
y = data_with_labels[:, -1]

# 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize K-NN classifier with k=1
knn = KNeighborsClassifier(n_neighbors=1)

# Perform 10-fold cross-validation
fold_accuracy = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the classifier on the training data
    knn.fit(X_train, y_train)
    # Make predictions on the test data
    y_pred = knn.predict(X_test)
    # Calculate accuracy for this fold
    accuracy = accuracy_score(y_test, y_pred)
    fold_accuracy.append(accuracy)

# average accuracy
average_accuracy = np.mean(fold_accuracy)
print("Average Accuracy:", average_accuracy)
