import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score
import matplotlib.pyplot as plt

df = pd.read_excel("C:/Users/brais/OneDrive/Desktop/ML/STATISTICS3/data1.xlsx")

# Define features and target
X = df[['x1', 'x2', 'x3', 'x4']]
y = df['y']

# Partition data: 50% train, 30% validation, 20% test, seed=12345
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.375, random_state=12345)  # 0.375 of 0.8 = 0.3

# Treat all predictors as numerical (no character data in this case)
# If any were categorical, we'd use pd.get_dummies()

# Train decision tree with cost-complexity pruning
clf = DecisionTreeClassifier(random_state=12345)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Evaluate trees on validation set
val_errors = []
decision_nodes = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=12345, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    val_pred = clf.predict(X_val)
    error = 1 - accuracy_score(y_val, val_pred)  # Misclassification error
    val_errors.append(error)
    decision_nodes.append(clf.tree_.node_count - clf.tree_.n_leaves)  # Decision nodes = total nodes - leaves

# a-1: Minimum error in validation data
min_error_idx = np.argmin(val_errors)
min_error = val_errors[min_error_idx]
print(f"a-1. Minimum error in validation data: {min_error:.4f}")

# a-2: Decision nodes for minimum error tree
min_error_decision_nodes = decision_nodes[min_error_idx]
print(f"a-2. Decision nodes for minimum error: {min_error_decision_nodes}")

# Fit the best-pruned tree (using the smallest ccp_alpha that gives good performance)
best_clf = DecisionTreeClassifier(random_state=12345, ccp_alpha=ccp_alphas[min_error_idx])
best_clf.fit(X_train, y_train)

# b: Leaf nodes in best-pruned and minimum error tree
best_leaf_nodes = best_clf.tree_.n_leaves
print(f"b. Leaf nodes in best-pruned tree: {best_leaf_nodes}, Minimum error tree: {best_leaf_nodes}")

# c: First split of best-pruned tree
feature_names = X.columns
tree_ = best_clf.tree_
first_feature_idx = tree_.feature[0]
first_split_value = tree_.threshold[0]
print(f"c. Predictor variable: {feature_names[first_feature_idx]}, Split value: {first_split_value:.1f}")

# d: Metrics on test data
test_pred = best_clf.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
accuracy = accuracy_score(y_test, test_pred)
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
precision = precision_score(y_test, test_pred)
print(f"d. Classification accuracy: {accuracy:.2f}, Specificity: {specificity:.2f}, Sensitivity: {sensitivity:.2f}, Precision: {precision:.2f}")

# e: Lift chart
test_prob = best_clf.predict_proba(X_test)[:, 1]
sorted_indices = np.argsort(test_prob)[::-1]
sorted_y_test = y_test.values[sorted_indices]
cum_positives = np.cumsum(sorted_y_test)
cum_total = np.arange(1, len(y_test) + 1)
lift = cum_positives / (cum_positives[-1] / len(y_test))  # Relative to baseline
baseline = np.ones(len(y_test))
plt.plot(cum_total, lift, label="Lift Curve")
plt.plot(cum_total, baseline, label="Baseline", linestyle="--")
plt.xlabel("Number of Instances")
plt.ylabel("Lift")
plt.legend()
plt.show()
entire_lift_above_baseline = np.all(lift >= baseline)
print(f"e. Entire lift curve above baseline? {'Yes' if entire_lift_above_baseline else 'No'}")

# f: ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, test_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
print(f"f. AUC value: {roc_auc:.4f}")