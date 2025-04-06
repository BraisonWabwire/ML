import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# --------------------
# STEP 1: Load the data
# --------------------
# Replace "data.xlsx" with your actual Excel file name
df = pd.read_excel("C:/Users/brais/OneDrive/Desktop/ML/STATISTICS2/data6.xlsx")

# --------------------
# STEP 2: Bin x1 and x2
# --------------------
x1_bins = [0, 60, 400, 30000]
x2_bins = [0, 160, 400, 800]

df['x1_bin'] = pd.cut(df['x1'], bins=x1_bins, labels=[0, 1, 2], right=False)
df['x2_bin'] = pd.cut(df['x2'], bins=x2_bins, labels=[0, 1, 2], right=False)

# Show binned values for the first two observations
print("Bin numbers for the first two observations:")
print(df[['x1_bin', 'x2_bin']].head(2))

# --------------------
# STEP 3: Prepare data
# --------------------
# Drop rows with NaN from binning (if any)
df = df.dropna(subset=['x1_bin', 'x2_bin'])

# Convert to int type for Naive Bayes
X = df[['x1_bin', 'x2_bin']].astype(int)
y = df['y']

# Set seed
np.random.seed(1)

# Split into 60% training and 40% validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=1)

# --------------------
# STEP 4: Train Naive Bayes model
# --------------------
model = CategoricalNB()
model.fit(X_train, y_train)

# --------------------
# STEP 5: Make predictions
# --------------------
y_pred = model.predict(X_val)
y_proba = model.predict_proba(X_val)[:, 1]

# --------------------
# STEP 6: Evaluate model
# --------------------
accuracy = accuracy_score(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()

specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"\n--- Evaluation Metrics ---")
print(f"Accuracy: {accuracy:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"Sensitivity: {sensitivity:.2f}")

# --------------------
# STEP 7: ROC Curve and AUC
# --------------------
fpr, tpr, thresholds = roc_curve(y_val, y_proba)
auc_score = roc_auc_score(y_val, y_proba)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"Naive Bayes (AUC = {auc_score:.4f})")
plt.plot([0, 1], [0, 1], 'k--')  # Baseline
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"AUC: {auc_score:.4f}")

# --------------------
# STEP 8: Evaluate Statement
# --------------------
statement = "True" if auc_score > 0.5 else "False"
print(f"Statement: The Naive Bayes model performs better than the baseline model: {statement}")
