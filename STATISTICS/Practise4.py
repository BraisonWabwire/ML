import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve

# Load the dataset
file_path = "C:/Users/brais/OneDrive/Desktop/ML/STATISTICS/practise4.xlsx"
data = pd.read_excel(file_path)

# Convert target variable to categorical
data['y'] = data['y'].astype(int)

# Split data into features (X) and target (y)
X = data.iloc[:, 1:]  # Predictor variables (x1, x2, x3)
y = data.iloc[:, 0]   # Target variable (y)

# Partition the dataset (60% training, 40% validation)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.4, random_state=1)

# Perform 10-fold cross-validation to find the best k
best_k = 1
best_score = 0
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    avg_score = np.mean(scores)
    if avg_score > best_score:
        best_score = avg_score
        best_k = k

print(f"Optimal value of k: {best_k}")

# Train KNN with optimal k
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
y_pred = knn_best.predict(X_valid)

# Confusion matrix
conf_matrix = confusion_matrix(y_valid, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Calculate metrics
accuracy = accuracy_score(y_valid, y_pred) * 100
sensitivity = recall_score(y_valid, y_pred) * 100  # Same as Recall
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) * 100
precision = precision_score(y_valid, y_pred) * 100
auc_value = roc_auc_score(y_valid, knn_best.predict_proba(X_valid)[:, 1])

# Misclassification rate
misclassification_rate = (1 - accuracy / 100) * 100

print(f"Misclassification Rate: {misclassification_rate:.2f}%")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Sensitivity: {sensitivity:.2f}%")
print(f"Specificity: {specificity:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"AUC: {auc_value:.2f}")

# Determine most accurate multiple-choice statement
if round(accuracy, 2) == 81:
    correct_statement = "D. Overall, the KNN model is able to correctly classify 81% of the cases."
elif round(sensitivity, 2) == 66:
    correct_statement = "B. The KNN model is able to correctly classify 66% of the Class 1 cases."
elif round(specificity, 2) == 66:
    correct_statement = "C. The KNN model is able to correctly classify 66% of the Class 0 cases."
elif round(misclassification_rate, 2) == 19:
    correct_statement = "A. The error rate of the KNN model is 19%."
else:
    correct_statement = "No exact match found."

print(f"Most accurate statement: {correct_statement}")

# Compute naïve rule accuracy (majority class prediction)
majority_class = y_train.value_counts().idxmax()
naive_pred = np.full_like(y_valid, majority_class)
naive_accuracy = accuracy_score(y_valid, naive_pred) * 100

print(f"Naïve Rule Accuracy: {naive_accuracy:.2f}%")

# Determine if the ROC curve is closer to the optimum or baseline model
roc_classification = "Optimum (Perfect) model" if auc_value > 0.75 else "Baseline model"
print(f"ROC Curve Classification: {roc_classification}")
