import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import numpy as np

# Load the data from the Excel file
data = pd.read_excel('C:/Users/brais/OneDrive/Desktop/ML/STATISTICS/practise3.xlsx', engine='openpyxl')

# Separate features and target
X = data[['x1', 'x2', 'x3']]
y = data['y']

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=12345)  # 50% training, 50% remaining
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=12345)  # 40% validation, 60% test

# Initialize a list to store the results for different values of k
k_values = range(1, 11)
results = []

# Loop over k values to find the optimal k
for k in k_values:
    # Train the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict on the validation set
    y_val_pred = knn.predict(X_val)

    # Calculate misclassification rate
    misclassification_rate = 1 - accuracy_score(y_val, y_val_pred)

    results.append((k, misclassification_rate))

# Find the optimal k with the lowest misclassification rate
optimal_k = min(results, key=lambda x: x[1])[0]
print(f'Optimal k: {optimal_k}')

# Train the KNN model with the optimal k
knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)
knn_optimal.fit(X_train, y_train)

# Predict on the test set
y_test_pred = knn_optimal.predict(X_test)

# Compute the confusion matrix and other metrics
conf_matrix = confusion_matrix(y_test, y_test_pred)
accuracy = accuracy_score(y_test, y_test_pred)
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
auc = roc_auc_score(y_test, knn_optimal.predict_proba(X_test)[:, 1])

# Print the results
print(f'Accuracy: {accuracy:.2f}')
print(f'Specificity: {specificity:.2f}')
print(f'Sensitivity: {sensitivity:.2f}')
print(f'Precision: {precision:.2f}')
print(f'AUC: {auc:.2f}')

# Part c-2: Analyze the confusion matrix for the given multiple-choice question
print(f'Confusion Matrix:\n{conf_matrix}')
