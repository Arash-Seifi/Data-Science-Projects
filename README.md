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


## Exercise8_Vehicle-Routing
The code implements a genetic algorithm to optimize vehicle routing for package delivery, considering constraints such as vehicle capacity, route time limits, and priority deliveries. It initializes a population of possible routes, evaluates them using a fitness function that penalizes delays and overcapacity, and iteratively improves solutions through selection, crossover, and mutation. Each vehicle follows a route starting and ending at a central depot, with dynamically generated distances and traffic conditions influencing travel times. The algorithm evolves over multiple generations to find the most efficient routing solution, minimizing total distance and penalties.


## Exercise9_Recommender-System
This code implements a Q-learning-based movie recommender system that adapts to user preferences through interaction. It defines a list of movies, each associated with multiple genres and popularity scores, and models user preferences based on their genre interests. A reinforcement learning agent selects movies using an epsilon-greedy strategy, updating Q-values based on user feedback. The reward function assigns positive or negative scores depending on how well a recommendation matches the user's interests. The system interacts with the user, refines recommendations over multiple iterations, and ultimately suggests movies based on learned preferences.


## Exercise10_SOM-for-Iris-Clustering
This script applies a Self-Organizing Map (SOM) to cluster the Iris dataset. It begins by loading and exploring the dataset, checking for missing values and summarizing data distribution. The features are then standardized using StandardScaler before training a 10x10 SOM with a learning rate of 0.5 and a neighborhood radius of 1 over 100 iterations. Each data point is assigned to its closest neuron, forming clusters that are visualized using scatter plots. The model's clustering quality is evaluated by mapping clusters to original labels, calculating accuracy, and generating a confusion matrix, providing insights into how well the SOM distinguishes between different Iris species.


## Exercise11_Custom-Kmeans
Both codes detail a process for clustering student performance data.  The first report focuses on implementing a custom KMeans algorithm using a Manhattan distance metric, alongside standard KMeans with Euclidean distance, and uses the WCSS method to determine the optimal number of clusters.  It then visualizes the results using PCA.

The second report expands on this by comparing multiple methods for determining the optimal number of clusters (K): the Elbow method, Silhouette method, Gap Statistic, and Dunn Index. It uses standard KMeans with Euclidean distance and visualizes the clustered data using PCA.  While both reports cover similar preprocessing steps (missing value imputation, one-hot encoding, and feature scaling), the second report emphasizes a more comprehensive approach to selecting the optimal K and uses standard KMeans whereas the first report implements a custom KMeans with Manhattan distance in addition to standard KMeans.  Essentially, the second report builds upon the first by adding more robust K selection methodologies and focusing solely on standard KMeans with Euclidean distance.


## Exercise12-LVQ-for-Cheater-Detection
This report details the use of Learning Vector Quantization (LVQ) to detect cheaters based on features like time taken, attempts, IP region, code similarity, and requests.  The data is preprocessed by selecting relevant features, handling categorical data with one-hot encoding, and standardizing numerical features.  The dataset is then split into training and testing sets.  An LVQ model with two prototypes per class is trained on the training data.  The model's performance is evaluated on the test set using accuracy and a confusion matrix.  The results provide a measure of the model's ability to classify cheaters, and the report discusses potential improvements like parameter tuning, feature engineering, and comparison with other classification algorithms.