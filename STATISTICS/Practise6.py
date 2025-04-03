import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, roc_curve

# Load the Excel data
file_path = "C:/Users/brais/OneDrive/Desktop/ML/STATISTICS/practise6.xlsx"
df = pd.read_excel(file_path)
# Check the data
print(df.head())

# Split data into features (X) and target (y)
X = df[['x1', 'x2', 'x3', 'x4']]
y = df['y']

# Split the dataset into 60% training and 40% validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=1)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Initialize KNN classifier
knn = KNeighborsClassifier()

# Perform 10-fold cross-validation to search for the optimal k
k_values = range(1, 11)
mean_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    accuracies = cross_val_score(knn, X_train_scaled, y_train, cv=10, scoring='accuracy')
    mean_accuracies.append(accuracies.mean())

# Plot the accuracies for different k values
plt.plot(k_values, mean_accuracies, marker='o')
plt.title("KNN: Accuracy vs. K Value")
plt.xlabel("K Value")
plt.ylabel("Mean Accuracy")
plt.show()

# Optimal K
optimal_k = k_values[mean_accuracies.index(max(mean_accuracies))]
print(f"The optimal value of k is: {optimal_k}")

# Train the model with the optimal k
knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)
knn_optimal.fit(X_train_scaled, y_train)

# Make predictions on the validation set
y_pred = knn_optimal.predict(X_val_scaled)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)
precision = precision_score(y_val, y_pred)
sensitivity = recall_score(y_val, y_pred)
specificity = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1])
roc_auc = roc_auc_score(y_val, knn_optimal.predict_proba(X_val_scaled)[:, 1])

print(f"Accuracy: {accuracy:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"Sensitivity: {sensitivity:.2f}")
print(f"Precision: {precision:.2f}")
print(f"AUC: {roc_auc:.2f}")

# Calculate the lift value for the leftmost bar (assuming this is related to the confusion matrix)
# This depends on how the decile-wise lift chart is defined, but here's a basic lift calculation
lift_value = (conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[0,1])) / (sum(y_val == 1) / len(y_val))
print(f"Lift value: {lift_value:.2f}")

# For scoring new observations (assuming a separate dataset)
file_path = "C:/Users/brais/OneDrive/Desktop/ML/STATISTICS/test6.xlsx"
new_data = pd.read_excel(file_path)
new_data_scaled = scaler.transform(new_data[['x1', 'x2', 'x3', 'x4']])
predictions = knn_optimal.predict(new_data_scaled)
print(f"Predicted class memberships for the new observations: {predictions[:2]}")
