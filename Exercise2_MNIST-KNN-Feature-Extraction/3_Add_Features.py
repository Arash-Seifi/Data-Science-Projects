import os
import cv2
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Average pixel value
def average_pixel_value(image):
    return np.mean(image)

# Standard deviation of pixel values
def pixel_value_std(image):
    return np.std(image)

# Histogram of pixel values
def pixel_value_histogram(image):
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0,256])
    return hist

# Aspect ratio of the image
def aspect_ratio(image):
    height, width = image.shape
    return width / height

# Number of edges 
def edge_count(image):
    img_uint8 = cv2.convertScaleAbs(image)
    # Apply Canny edge detection
    edges = cv2.Canny(img_uint8, 100, 200)
    # Return the sum of edges
    return np.sum(edges)

mnist_path = "./"

data_with_labels = np.load("mnist_image_vectors_with_labels.npy")
np.random.shuffle(data_with_labels)

X = data_with_labels[:, :-1]
y = data_with_labels[:, -1]

feature1 = []
feature2 = []
feature3 = []
feature4 = []
feature5 = []

# Extract features from each image
for img_vector in X:
    img = img_vector.reshape(28, 28)  # images are 28x28
    feature1.append(average_pixel_value(img))
    feature2.append(pixel_value_std(img))
    feature3.append(pixel_value_histogram(img))
    feature4.append(aspect_ratio(img))
    feature5.append(edge_count(img))

# Convert lists to NumPy arrays
feature1 = np.array(feature1)
feature2 = np.array(feature2)
feature3 = np.array(feature3)
feature4 = np.array(feature4)
feature5 = np.array(feature5)

# Add the new features as columns 
data_with_features = np.column_stack((X, feature1, feature2, feature3, feature4, feature5, y))

# Save the NumPy array
np.save("mnist_image_vectors_with_features.npy", data_with_features)
