import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

# Set random seed
random_seed = 12345

# Load the data
file_path = "C:/Users/brais/OneDrive/Desktop/ML/STATISTICS/Ch12_Q1_V10_Data_File.xlsx"  # Change this to your actual file path
df = pd.read_excel(file_path)


# Splitting data into features (X) and target variable (y)
X = df[['x1', 'x2']]
y = df['y']

# Partition the data: 50% training, 30% validation, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=random_seed)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=random_seed)  # 40% of 50% = 20%

# Find the optimal k
best_k = 1
best_accuracy = 0

for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_val_pred = knn.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

# Train and test the final model with the best k
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train, y_train)
y_test_pred = knn_final.predict(X_test)

# Compute metrics
accuracy = accuracy_score(y_test, y_test_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
specificity = tn / (tn + fp)  # True Negative Rate
sensitivity = tp / (tp + fn)  # Recall or True Positive Rate
precision = precision_score(y_test, y_test_pred)

# Round metrics to 2 decimal places
accuracy = round(accuracy, 2)
specificity = round(specificity, 2)
sensitivity = round(sensitivity, 2)
precision = round(precision, 2)

# Display results
print(f"Optimal k: {best_k}")
print(f"Accuracy: {accuracy}")
print(f"Specificity: {specificity}")
print(f"Sensitivity: {sensitivity}")
print(f"Precision: {precision}")

# Identify the least accurate statement
error_rate = round(1 - accuracy, 2)

statements = {
    "A": error_rate == 0.09,
    "B": sensitivity == 1.00,
    "C": specificity == 0.89,
    "D": accuracy == 0.92
}

least_accurate_statement = [key for key, value in statements.items() if not value]
print(f"The least accurate statement is: {least_accurate_statement[0]}")
