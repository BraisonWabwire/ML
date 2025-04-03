import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, roc_auc_score, recall_score

# Set random seed
np.random.seed(12345)

# Load the Excel file
file_path = "C:/Users/brais/OneDrive/Desktop/ML/STATISTICS/practise5.xlsx"
df = pd.read_excel(file_path)

# Splitting features (X) and target variable (y)
X = df.iloc[:, 1:]  # Predictor variables: x1, x2, x3, x4
y = df.iloc[:, 0]   # Target variable: y

# Standardize the features (important for KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Partitioning the data (50% training, 30% validation, 20% test)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.5, random_state=12345)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=12345)  # 40% of 50% = 20%

# Finding the optimal k
k_values = range(1, 11)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

optimal_k = k_values[np.argmax(cv_scores)]
print(f"Optimal value of k: {optimal_k}")

# Train and evaluate the KNN model with the optimal k
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Compute required metrics
accuracy = accuracy_score(y_test, y_pred)
misclassification_rate = 1 - accuracy
sensitivity = recall_score(y_test, y_pred)  # True Positive Rate
specificity = tn / (tn + fp) if (tn + fp) != 0 else np.nan  # True Negative Rate
precision = precision_score(y_test, y_pred)
auc = roc_auc_score(y_test, knn.predict_proba(X_test)[:, 1])

# Print results rounded to 2 decimal places
print(f"Accuracy: {accuracy:.2f}")
print(f"Misclassification Rate: {misclassification_rate:.2f}")
print(f"Sensitivity: {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"Precision: {precision:.2f}")
print(f"AUC: {auc:.2f}")

# Identifying the least accurate statement
error_rate = misclassification_rate * 100
class_1_accuracy = sensitivity * 100
class_0_accuracy = specificity * 100
overall_accuracy = accuracy * 100

# Choose the least accurate statement
statements = {
    "A": abs(error_rate - 29),
    "B": abs(class_1_accuracy - 89),
    "C": abs(class_0_accuracy - 54),
    "D": abs(overall_accuracy - 74),
}

least_accurate = min(statements, key=statements.get)
print(f"Least accurate statement: {least_accurate}")

# Score new observations (assuming Exercise_12.7_Score worksheet exists in data.xlsx)
file_path2 = "C:/Users/brais/OneDrive/Desktop/ML/STATISTICS/scorefile.xlsx"
new_data = pd.read_excel(file_path2, sheet_name="Exercise_12.7_Score")

# Ensure new_data only has the features (columns x1 to x4)
new_data = new_data[['x1', 'x2', 'x3', 'x4']]

# Scale the new data using the scaler that was fit on the training data
new_data_scaled = scaler.transform(new_data)

# Make predictions with the trained model
predictions = knn.predict(new_data_scaled)

# Display the first two predictions
print(f"Predicted class for first two new observations: {predictions[:2]}")

# ===========================
# Lift Calculation for the Leftmost Decile
# ===========================

# Get predicted probabilities (for class 1, positive class)
y_pred_prob = knn.predict_proba(X_test)[:, 1]

# Calculate the overall response rate (average response rate) in the entire dataset
average_response_rate = y.mean()

# Sort the probabilities in descending order
sorted_indices = np.argsort(y_pred_prob)[::-1]

# Create a DataFrame with the predicted probabilities and actual values
df_results = pd.DataFrame({
    'pred_prob': y_pred_prob[sorted_indices],  # Sorted predicted probabilities
    'true_label': y_test.iloc[sorted_indices]   # Corresponding true labels
})

# Assign deciles to the DataFrame based on predicted probabilities
df_results['decile'] = np.digitize(df_results['pred_prob'], bins=np.percentile(df_results['pred_prob'], np.arange(0, 100, 10)))

# Calculate the response rate for the leftmost decile (highest predicted probabilities)
decile_1_response_rate = df_results[df_results['decile'] == 1]['true_label'].mean()

# Calculate lift for the leftmost decile
lift_leftmost = decile_1_response_rate / average_response_rate

# Print the lift value rounded to 2 decimal places
print(f"Lift value for the leftmost decile: {lift_leftmost:.2f}")
