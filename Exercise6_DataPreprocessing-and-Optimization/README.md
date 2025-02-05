# Data Preprocessing and Optimization

## Introduction
Data preprocessing is a crucial step in ensuring high-quality data for analysis and machine learning. The provided scripts perform **data cleaning, transformation, and feature selection** on a dataset from `Project_Data.xlsx`. These processes enhance data reliability by handling missing values, managing outliers, normalizing features, and reducing redundancy.

## Overview of Processes

### 1. Data Cleaning and Handling Missing Values
- The dataset is loaded, and non-numeric values are converted to numeric.
- Missing values are filled using the **mode** (most frequent value).
- Any remaining rows with missing values are removed to ensure data integrity.

### 2. Outlier Detection and Management
- The **Interquartile Range (IQR) method** is used to detect outliers.
- Identified outliers are replaced with the mode to prevent extreme values from affecting analysis.
- Box plots are generated to visualize data distributions **before and after** outlier handling.

### 3. Data Normalization
- The dataset is normalized using **Z-score normalization**, ensuring all features have a mean of **0** and a standard deviation of **1**.
- This transformation helps standardize data, making it more suitable for machine learning models.

### 4. Feature Selection
- A **correlation matrix** is calculated to identify relationships between features.
- Features with high correlation are evaluated based on their relationship with the target variable (`Total`).
- The **least important** features are removed to reduce redundancy and improve model efficiency.

## Conclusion
By following a structured preprocessing approach, the dataset is transformed into a **clean, standardized, and optimized** form. This ensures better performance in downstream tasks such as statistical analysis and machine learning.

---

## Usage
To use these scripts:
1. Place `Project_Data.xlsx` in the same directory.
2. Run the scripts in the following order:
   ```bash
   python First.py   # Handles missing values and outliers
   python Second.py  # Normalizes the dataset
   python Third.py   # Performs feature selection
   ```

This will generate a cleaned and processed dataset, ready for further analysis.
