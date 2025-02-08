import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn_lvq import GlvqModel  # Use any compatible LVQ library
import numpy as np

# Load and preprocess dataset
data = pd.read_csv("LVQ.csv")
X = data[['TimeTaken', 'NumberOfAttempts', 'IPRegion', 'CodeSimilarity', 'NumberOfRequests']]
y = data['IsCheater']

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train LVQ model
lvq = GlvqModel(prototypes_per_class=2, initial_prototypes=None, max_iter=20)
lvq.fit(X_train, y_train)

# Predict and evaluate
y_pred = lvq.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
