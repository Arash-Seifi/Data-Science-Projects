# Data-Science-Projects
This repository is a collection of my data science projects, where I apply various analysis and visualization techniques.
### Exercise1_UsingDifferentPlots
This repository contains various data visualization examples using different types of charts, each suited for specific datasets and insights. The visualizations cover a wide range of data, including sales trends, weather conditions, sports performance, and market analysis. Each section includes a brief explanation of why a particular visualization was chosen and what insights it provides.

These examples can serve as a reference for selecting appropriate visualization techniques based on dataset characteristics and analysis goals.


### Exercise2_MNIST-KNN-Feature-Extraction
The set of scripts processes the MNIST dataset, first converting the raw images into flattened vectors with labels and saving them for use in machine learning. The second script performs K-Nearest Neighbors (K-NN) classification with 10-fold cross-validation on this preprocessed data, evaluating model performance. The third script enhances the dataset by extracting additional features, such as average pixel value, edge count, and aspect ratio, before performing the same cross-validation classification process. The final script evaluates the impact of excluding individual features on classification accuracy, helping assess the importance of each feature. Together, these scripts demonstrate a comprehensive workflow of data preparation, feature extraction, and model evaluation.


### Exercise3_MNIST- MultiDataset-ML-Evaluation
The provided code implements machine learning classification and clustering techniques on three datasets: Bank, Wine, and Bean. It preprocesses categorical features using label encoding and evaluates models using multiple validation methods, including train-test split, Leave-One-Out (LOO), and 10-fold cross-validation. The classifiers used include Naïve Bayes, K-Nearest Neighbors (KNN), Random Forest, and Decision Tree (Rep Tree equivalent), with KNN being tuned for optimal performance. Additionally, K-Means and a variant with limited iterations (Farthest First Clustering) are used for unsupervised learning. The results, including accuracy scores, confusion matrices, and classification reports, are saved in separate text files for analysis.


### Exercise4_HeartAttack-Prediction
This code explores the use of the K-Nearest Neighbors (KNN) algorithm for predicting heart attack likelihood based on patient health data. The dataset, containing features like age, sex, chest pain type, and cholesterol levels, was preprocessed through feature-target separation and standardization to improve KNN’s performance. The initial model was trained and evaluated with an 80-20 train-test split, achieving reasonable accuracy. To enhance performance, K-Fold Cross-Validation (10-fold) was applied, and different values of k (neighbors) were tested to optimize classification accuracy and F1-score. The findings suggest that KNN is a viable model for heart attack prediction, with performance improvements achieved through careful parameter tuning and cross-validation.


### Exercise5_Gesture-Identification
This implementation demonstrates real-time detection of thumbs-up and thumbs-down gestures using a custom-trained TensorFlow model. The approach leverages the Object Detection API, pre-trained models, and OpenCV for live video feed processing. The model successfully identifies the gestures and overlays bounding boxes with labels on the detected objects. Future improvements could include expanding the dataset for better accuracy and optimizing the model for real-time performance.


### Exercise6_DataPreprocessing-and-Optimization
This project focuses on data preprocessing and optimization, ensuring data quality for analysis and machine learning. The scripts handle missing values by replacing them with the mode, manage outliers using the IQR method, and normalize features with Z-score normalization for consistency. Additionally, a correlation-based feature selection method removes redundant features, improving model efficiency. The result is a clean, standardized, and optimized dataset, ready for further analysis and predictive modeling.


### Exercise7_MNIST-Classification-Report
This report details the implementation of an MLP model for MNIST digit classification using TensorFlow and Keras. The dataset is preprocessed, and a validation set is extracted for model evaluation. The architecture includes three dense layers with ReLU and softmax activations. The model is trained using the Adam optimizer and categorical cross-entropy loss function, achieving high accuracy. Evaluation metrics, accuracy, and loss curves are analyzed, and sample predictions are visualized. Future enhancements may involve CNNs for improved performance.