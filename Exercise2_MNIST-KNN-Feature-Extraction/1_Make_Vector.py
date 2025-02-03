import os
import cv2
import numpy as np

# the path 
mnist_path = "./"
# number of images 
num_images_to_use = 300

# Create empty lists to store image vectors and corresponding labels
image_vectors = []
labels = []

# Go through each digit folder 
for digit in range(10):
    digit_folder_path = os.path.join(mnist_path, "train", str(digit))

    # Get a list of image files in the folder
    image_files = os.listdir(digit_folder_path)[:num_images_to_use]

    # Iterate through each image file
    for image_file in image_files:
        # Read the image using OpenCV
        image_path = os.path.join(digit_folder_path, image_file)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Flatten the image into a 1D vector
        img_vector = img.flatten()
        # Append the vector to the list
        image_vectors.append(img_vector)
        labels.append(digit)

# Convert the lists to NumPy arrays
image_vectors_array = np.array(image_vectors)
labels_array = np.array(labels)

# Add a new column for labels to the image_vectors_array
image_vectors_array_with_labels = np.column_stack((image_vectors_array, labels_array))

# Save the NumPy array
np.save("mnist_image_vectors_with_labels.npy", image_vectors_array_with_labels)
