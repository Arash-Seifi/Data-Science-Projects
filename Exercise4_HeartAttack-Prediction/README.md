# Report on Heart Attack Prediction Using K-Nearest Neighbors (KNN)

## 1. Objective
The goal of this project was to build a machine learning model using the K-Nearest Neighbors (KNN) algorithm to predict the likelihood of a heart attack based on patient data. The dataset, `heart.csv`, contains various features such as age, sex, chest pain type, cholesterol levels, and other health indicators that can help identify the risk of a heart attack.

## 2. Dataset Description
The dataset used consists of several important features related to heart health:

- **age**: Age of the patient.
- **sex**: Gender of the patient (1 = male, 0 = female).
- **cp**: Chest pain (categorical: 0-3).
- ...
- **output**: Target variable indicating whether the patient experienced a heart attack (1) or not (0).

## 3. Data Preprocessing
Before training the model, several preprocessing steps were applied:

- **Feature-target separation**: The dataset was split into input features (`X`) and the target variable (`y`), which indicated the likelihood of a heart attack.
- **Standardization**: Since KNN is a distance-based algorithm, it works best when the data is normalized. Therefore, all feature variables were standardized using a `StandardScaler` to ensure they had a mean of 0 and a standard deviation of 1. This helped improve the model’s performance by ensuring that features with larger numeric ranges did not dominate the distance metric.

## 4. Train-Test Split
The data was split into training and testing sets to evaluate the performance of the KNN model. The split was set to 80% training data and 20% testing data to ensure the model had enough data to learn from while retaining sufficient data to test its accuracy.

## 5. Model Selection
The K-Nearest Neighbors (KNN) algorithm was selected for this classification task. KNN is a non-parametric, distance-based algorithm that classifies a data point based on the majority class of its nearest neighbors. Initially, the number of neighbors (`k`) was set to 5, which is a common starting point for KNN models.

## 6. Model Training
The KNN model was trained using the preprocessed training data. The algorithm calculated the distance between a test instance and all training instances, then classified the test instance based on the majority vote among its 5 nearest neighbors.

## 7. Results
The KNN model achieved a reasonable accuracy on the test set, demonstrating that it can effectively predict the likelihood of a heart attack based on the given features. The confusion matrix provided further insight into the model's ability to correctly classify both high-risk (heart attack) and low-risk (no heart attack) cases. The classification report highlighted the model's precision and recall across both classes.

---

## 8. Improving Model Performance

### 1. Objective
The objective of this phase was to improve the predictive performance of the K-Nearest Neighbors (KNN) model by:

- Utilizing **K-Fold Cross-Validation** to ensure more reliable performance estimates.
- Testing **different values of k** (number of neighbors) to identify the optimal value.
- Introducing the **F1-score** as an additional evaluation metric, complementing accuracy.

### 2. Dataset Overview
The dataset remained the same as the previous step, containing features related to patient health data. The target variable (`output`) indicates the presence (1) or absence (0) of a heart attack.

- **Features**: Age, sex, chest pain type (`cp`), blood pressure (`trtbps`), cholesterol (`chol`), and other health indicators.
- **Target**: Likelihood of a heart attack (`output`).

### 3. Data Preprocessing
Similar to the previous experiment, the following preprocessing steps were applied:

- **Feature-target split**: The dataset was divided into `X` (features) and `y` (target).
- **Standardization**: A `StandardScaler` was used to normalize the feature values, ensuring that all features had the same scale, as KNN is sensitive to differences in scale.

### 4. K-Fold Cross-Validation
To enhance the reliability of the model's performance estimates, **K-Fold Cross-Validation** was used:

- A **10-Fold Cross-Validation** was applied, meaning the data was split into 10 subsets (folds). The model was trained on 9 folds and tested on the remaining fold, rotating until every fold had been used as a test set. The final performance score was the average of all the test results across the 10 folds.

### 5. Testing Different Values of K
A range of **k** values (from **1 to 20**) was tested to identify the optimal number of neighbors that yielded the best performance.

- For each value of **k**, the KNN model was trained and evaluated using the **10-fold cross-validation method**.
- Both **accuracy** and **F1-score** were computed for each value of **k**.
