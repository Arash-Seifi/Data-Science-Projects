# MNIST Image Classification with K-Nearest Neighbors (K-NN)

This repository contains a series of Python scripts that perform image classification on the MNIST dataset using K-Nearest Neighbors (K-NN) classifier. The code is divided into four scripts that handle different stages of the machine learning pipeline: data preprocessing, feature extraction, model evaluation, and feature impact analysis.

## Scripts Overview

### 1. **Image Preprocessing and Data Preparation**

This script processes raw MNIST image data, converting it into numerical vectors that can be used in machine learning models.

#### Steps:
- Loads images from the MNIST dataset (digit folders from 0-9).
- Converts images to grayscale and flattens them into 1D vectors.
- Stores the image vectors and their corresponding labels in lists.
- Saves the processed data as a `.npy` file for further use.

#### Key Output:
- A NumPy file (`mnist_image_vectors_with_labels.npy`) containing the image vectors and their corresponding labels.

---

### 2. **K-NN Classification with 10-Fold Cross-Validation**

This script applies the K-Nearest Neighbors (K-NN) classifier using the data prepared in the first script. It performs 10-fold cross-validation to evaluate the performance of the model.

#### Steps:
- Loads the preprocessed data (`mnist_image_vectors_with_labels.npy`).
- Splits the data using 10-fold cross-validation.
- Trains a K-NN classifier (`k=1`) on the training data and evaluates the accuracy on the test fold.
- Computes and prints the average accuracy across all 10 folds.

#### Key Output:
- The average accuracy for the K-NN classifier after 10-fold cross-validation.

---

### 3. **Feature Extraction and Classification with All Features**

This script extracts additional features from the MNIST images and evaluates the performance of the K-NN classifier using these features with 10-fold cross-validation.

#### Steps:
- Computes five additional features for each image:
  - Average pixel value
  - Standard deviation of pixel values
  - Histogram of pixel values
  - Aspect ratio (width/height)
  - Edge count using Canny edge detection
- Adds these features to the original dataset and saves the new dataset as a `.npy` file.
- Performs 10-fold cross-validation using the K-NN classifier with the augmented feature set.
- Computes and prints the average accuracy across all 10 folds.

#### Key Output:
- A NumPy file (`mnist_image_vectors_with_features.npy`) containing both the original image vectors and the new features.
- The average classification accuracy after 10-fold cross-validation.

---

### 4. **Excluding Features for Impact Evaluation**

This script evaluates the impact of each feature on the K-NN classifier's performance by excluding one feature at a time and observing how the accuracy changes.

#### Steps:
- Iterates over all features and excludes one feature at a time.
- Performs 10-fold cross-validation on the remaining features and computes the accuracy.
- Prints the average accuracy after excluding each feature.

#### Key Output:
- The impact of excluding each feature on classification accuracy, showing how the accuracy changes with different feature combinations.
