import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

# Load data from Excel
file_path = "C:/Users/brais/OneDrive/Desktop/ML/STATISTICS/practise2.xlsx"
df = pd.read_excel(file_path)

# Split data into features (X) and target variable (y)
X = df[['x1', 'x2']]
y = df['y']

# Set seed for reproducibility
np.random.seed(1)

# Split data into training (60%) and validation (40%) sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=1, stratify=y)

# Perform 10-fold cross-validation to find the best k
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
k_values = range(1, 11)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=kf, scoring='accuracy')
    cv_scores.append(scores.mean())

# Find the optimal k
optimal_k = k_values[np.argmax(cv_scores)]
print(f"Optimal k: {optimal_k}")

# Train and evaluate the KNN model with the optimal k
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_val)

# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

# Compute performance metrics
accuracy = accuracy_score(y_val, y_pred)
specificity = tn / (tn + fp)  # True Negative Rate
sensitivity = tp / (tp + fn)  # Recall or True Positive Rate
precision = precision_score(y_val, y_pred)

# Print results
print(f"Accuracy: {accuracy:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"Sensitivity: {sensitivity:.2f}")
print(f"Precision: {precision:.2f}")

# Checking the least accurate statement
error_rate = 1 - accuracy
print(f"Error Rate: {error_rate:.2f}")
