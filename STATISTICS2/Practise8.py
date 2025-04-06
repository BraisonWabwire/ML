import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

# Sample data
df = pd.read_excel("C:/Users/brais/OneDrive/Desktop/ML/STATISTICS2/data8.xlsx")


# Binning as per the instructions
df['x1_bin'] = pd.cut(df['x1'], bins=[0, 6, 14, 30], labels=[0, 1, 2], right=False)
df['x2_bin'] = pd.cut(df['x2'], bins=[0, 10, 20, 61], labels=[0, 1, 2], right=False)
df['x3_bin'] = pd.cut(df['x3'], bins=[0, 3, 5, 10], labels=[0, 1, 2], right=False)

# Display bin numbers for first two observations
print("Bin numbers for the first two observations:")
print(df[['x1_bin', 'x2_bin', 'x3_bin']].head(2))

# Prepare binned data for modeling
X = df[['x1_bin', 'x2_bin', 'x3_bin']].astype('int')
y = df['y']

# Partition data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=1, stratify=y)

# Train Naive Bayes model
model = CategoricalNB()
model.fit(X_train, y_train)

# Predict probabilities and classes (cutoff 0.5)
y_prob = model.predict_proba(X_val)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# Evaluation metrics
def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    return accuracy, specificity, sensitivity, precision

accuracy, specificity, sensitivity, precision = compute_metrics(y_val, y_pred)
auc = roc_auc_score(y_val, y_prob)

print("\n--- Evaluation Metrics (cutoff = 0.5) ---")
print(f"Accuracy: {accuracy:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"Sensitivity: {sensitivity:.2f}")
print(f"Precision: {precision:.2f}")
print(f"AUC: {auc:.4f}")
print(f"Statement: The Naive Bayes model performs better than the baseline model: {auc > 0.5}")

# ROC curve
fpr, tpr, _ = roc_curve(y_val, y_prob)
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

# Change cutoff to 0.2 and re-evaluate
y_pred_cutoff_02 = (y_prob >= 0.2).astype(int)
accuracy2, specificity2, sensitivity2, precision2 = compute_metrics(y_val, y_pred_cutoff_02)

print("\n--- Evaluation Metrics (cutoff = 0.2) ---")
print(f"Accuracy: {accuracy2:.2f}")
print(f"Specificity: {specificity2:.2f}")
print(f"Sensitivity: {sensitivity2:.2f}")
print(f"Precision: {precision2:.2f}")
