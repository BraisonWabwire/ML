import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Read the CSV file
data = pd.read_csv('C:/Users/brais/OneDrive/Desktop/ML/STATISTICS4/9.csv', header=None, names=['a', 'b', 'c', 'd', 'e', 'f', 'g'])

# Convert to one-hot encoded format (True if item is present, False if empty)
one_hot = data.notnull()

# Function to generate association rules and find top rule
def generate_rules(one_hot, min_support, min_confidence):
    # Generate frequent itemsets using Apriori
    frequent_itemsets = apriori(one_hot, min_support=min_support, use_colnames=True)
    
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    # Sort rules by lift, then confidence for ties
    rules = rules.sort_values(by=['lift', 'confidence'], ascending=False)
    
    return rules

# Part a-1 and a-2: min support = 10 transactions (10%), min confidence = 50%
min_support_a = 10 / 100  # 10 transactions out of 100
min_confidence_a = 0.5  # 50%
rules_a = generate_rules(one_hot, min_support_a, min_confidence_a)

# Number of rules
num_rules_a = len(rules_a)

# Top rule's lift ratio (rounded to 2 decimal places)
top_lift_a = round(rules_a['lift'].iloc[0], 2) if not rules_a.empty else None

# Top rule for multiple choice
top_rule_a = rules_a[['antecedents', 'consequents']].iloc[0] if not rules_a.empty else None
if top_rule_a is not None:
    top_rule_a_str = f"{set(top_rule_a['antecedents'])} => {set(top_rule_a['consequents'])}"
else:
    top_rule_a_str = "No rules generated"

# Part b-1 and b-2: min support = 20 transactions (20%), min confidence = 80%
min_support_b = 20 / 100  # 20 transactions out of 100
min_confidence_b = 0.8  # 80%
rules_b = generate_rules(one_hot, min_support_b, min_confidence_b)

# Number of rules
num_rules_b = len(rules_b)

# Top rule's lift ratio (rounded to 2 decimal places)
top_lift_b = round(rules_b['lift'].iloc[0], 2) if not rules_b.empty else None

# Top rule for multiple choice
top_rule_b = rules_b[['antecedents', 'consequents']].iloc[0] if not rules_b.empty else None
if top_rule_b is not None:
    top_rule_b_str = f"{set(top_rule_b['antecedents'])} => {set(top_rule_b['consequents'])}"
else:
    top_rule_b_str = "No rules generated"

# Print results
print("Part a-1:")
print(f"Number of rules generated: {num_rules_a}")
print(f"Lift ratio of top rule: {top_lift_a}")
print("\nPart a-2:")
print(f"Top rule: {top_rule_a_str}")

print("\nPart b-1:")
print(f"Number of rules generated: {num_rules_b}")
print(f"Lift ratio of top rule: {top_lift_b}")
print("\nPart b-2:")
print(f"Top rule: {top_rule_b_str}")