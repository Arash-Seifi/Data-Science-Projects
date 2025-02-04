import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv('heart.csv')

# Step 2: Preprocess the data
X = data.drop(columns=['output'])  # Features
y = data['output']  # Target

# Standardize the feature variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Define K-Fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)  # 10-Fold Cross Validation

# Step 4: Test different values of K
k_values = range(1, 21)  # K values from 1 to 20
accuracies = []
f1_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Perform cross-validation for accuracy
    accuracy = cross_val_score(knn, X_scaled, y, cv=kf, scoring='accuracy').mean()
    accuracies.append(accuracy)
    
    # Perform cross-validation for F1-score
    f1 = cross_val_score(knn, X_scaled, y, cv=kf, scoring=make_scorer(f1_score)).mean()
    print(f"For {k} : ",accuracy,f1)
    f1_scores.append(f1)


# Step 6: Select the best K value (based on accuracy or F1-score)
best_k_accuracy = k_values[accuracies.index(max(accuracies))]
best_k_f1 = k_values[f1_scores.index(max(f1_scores))]

print(f"Best K (Accuracy): {best_k_accuracy}, Accuracy: {max(accuracies)*100:.2f}%")
print(f"Best K (F1 Score): {best_k_f1}, F1 Score: {max(f1_scores):.2f}")
