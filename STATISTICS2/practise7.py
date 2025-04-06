import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Sample data
df = pd.read_excel("C:/Users/brais/OneDrive/Desktop/ML/STATISTICS2/data7.xlsx")




# a. Equal count binning with 2 bins
df['x1_bin'] = pd.qcut(df['x1'], q=2, labels=[0, 1])
df['x2_bin'] = pd.qcut(df['x2'], q=2, labels=[0, 1])

# Show bin numbers for first two observations
print("Bin numbers for the first two observations:")
print(df[['x1_bin', 'x2_bin']].head(2))

# Convert bins to categorical (as required for Naive Bayes classifier)
X = df[['x1_bin', 'x2_bin']].astype('category')
y = df['y']

# b. Split data into 60% training, 40% validation
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.6, random_state=12345, stratify=y)

# Train Naive Bayes
model = CategoricalNB()
model.fit(X_train, y_train)

# Predict on validation
y_pred = model.predict(X_val)
y_proba = model.predict_proba(X_val)[:, 1]

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

# Metrics
accuracy = accuracy_score(y_val, y_pred)
sensitivity = recall_score(y_val, y_pred)                # Sensitivity = Recall
specificity = tn / (tn + fp)
precision = precision_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_proba)

# Print results
print("\n--- Evaluation Metrics ---")
print(f"Accuracy: {accuracy:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"Sensitivity (Recall): {sensitivity:.2f}")
print(f"Precision: {precision:.2f}")
print(f"AUC: {auc:.4f}")

# c-2. ROC Curve Statement
print("\nStatement: The Naive Bayes model performs better than the baseline model:",
      "True" if auc > 0.5 else "False")

# c-1. Plot ROC curve
fpr, tpr, _ = roc_curve(y_val, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"Naive Bayes (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label="Baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
