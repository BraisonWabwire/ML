import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt

# Sample Data (Replace with actual file path or data)
df = pd.read_excel("C:/Users/brais/OneDrive/Desktop/ML/STATISTICS2/data9.xlsx")


# a. Bin predictor variables x1, x2, and x3
# Use pandas qcut for equal count binning
bins = 3
df['x1_binned'] = pd.qcut(df['x1'], bins, labels=False)
df['x2_binned'] = pd.qcut(df['x2'], bins, labels=False)
df['x3_binned'] = pd.qcut(df['x3'], bins, labels=False)

# Display bin numbers for the first two observations
print("Bin numbers for the first two observations:")
print(df[['x1_binned', 'x2_binned', 'x3_binned']].iloc[:2])

# b. Partition the data into 60% training and 40% validation
X = df[['x1_binned', 'x2_binned', 'x3_binned']]
y = df['y']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=12345)

# Develop Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions on validation data
y_pred = model.predict(X_val)

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)
accuracy = accuracy_score(y_val, y_pred)
specificity = cm[0,0] / (cm[0,0] + cm[0,1])
sensitivity = recall_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)

# Report the metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"Sensitivity: {sensitivity:.2f}")
print(f"Precision: {precision:.2f}")

# c-1. Generate the ROC curve
fpr, tpr, thresholds = roc_curve(y_val, model.predict_proba(X_val)[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# c-2. ROC curve analysis statement
# The ROC curve indicates the trade-off between sensitivity and specificity across different cutoffs
roc_performance = "True" if roc_auc > 0.5 else "False"
print(f"Is the following statement true? 'The ROC curve shows that the naïve Bayes model performs better than the baseline model in terms of sensitivity and specificity across all possible cutoff values.'")
print(f"Answer: {roc_performance}")

# d. Change cutoff value to 0.2
y_pred_cutoff = (model.predict_proba(X_val)[:, 1] >= 0.2).astype(int)

# Confusion matrix and metrics for the new cutoff
cm_cutoff = confusion_matrix(y_val, y_pred_cutoff)
accuracy_cutoff = accuracy_score(y_val, y_pred_cutoff)
specificity_cutoff = cm_cutoff[0,0] / (cm_cutoff[0,0] + cm_cutoff[0,1])
sensitivity_cutoff = recall_score(y_val, y_pred_cutoff)
precision_cutoff = precision_score(y_val, y_pred_cutoff)

# Report the metrics for cutoff of 0.2
print(f"Accuracy (cutoff 0.2): {accuracy_cutoff:.2f}")
print(f"Specificity (cutoff 0.2): {specificity_cutoff:.2f}")
print(f"Sensitivity (cutoff 0.2): {sensitivity_cutoff:.2f}")
print(f"Precision (cutoff 0.2): {precision_cutoff:.2f}")
