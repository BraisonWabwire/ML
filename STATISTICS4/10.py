import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Read the CSV file
data = pd.read_csv('C:/Users/brais/OneDrive/Desktop/ML/STATISTICS4/10.csv', header=None, names=['action', 'romance', 'drama', 'comedy', 'horror'])

# Convert to one-hot encoded format (True if genre is present, False if empty)
one_hot = data.notnull()

# Function to generate association rules and find top rule
def generate_rules(one_hot, min_support, min_confidence, part_label, options):
    # Generate frequent itemsets using Apriori
    frequent_itemsets = apriori(one_hot, min_support=min_support, use_colnames=True)
    
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    # Sort rules by lift, then confidence for ties
    rules = rules.sort_values(by=['lift', 'confidence'], ascending=False)
    
    # Top rule
    top_rule = rules[['antecedents', 'consequents']].iloc[0] if not rules.empty else None
    top_rule_str = (f"{set(top_rule['antecedents'])} => {set(top_rule['consequents'])}"
                    if top_rule is not None else "No rules generated")
    top_lift = round(rules['lift'].iloc[0], 2) if not rules.empty else None
    
    # Find the top rule among the provided options
    top_option = None
    for option in options:
        ante, cons = option.split(' => ')
        ante_set = set(ante.strip('{}').split(','))
        cons_set = set(cons.strip('{}').split(','))
        for idx, rule in rules.iterrows():
            if set(rule['antecedents']) == ante_set and set(rule['consequents']) == cons_set:
                top_option = f"{ante} => {cons}"
                break
        if top_option:
            break
    
    # Print top rule metrics for debugging
    if top_rule is not None:
        print(f"\n{part_label} Top Rule: {top_rule_str}")
        print(f"Lift: {top_lift}, Confidence: {rules['confidence'].iloc[0]:.3f}")
    
    return rules, top_option if top_option else top_rule_str

# Define multiple-choice options
options_a = [
    "{romance} => {action}",
    "{action} => {romance}",
    "{drama} => {horror}",
    "{horror} => {action}"
]
options_b = [
    "{action,comedy} => {drama,horror}",
    "{drama,horror} => {action,comedy}",
    "{romance,horror} => {action,comedy}",
    "{action,comedy} => {romance,horror}"
]

# Part a-1: min support = 9 transactions (9/88 ≈ 0.1023), min confidence = 50%
min_support_a = 9 / 88
rules_a, top_rule_a = generate_rules(one_hot, min_support_a, 0.5, "Part a-1", options_a)

# Part a-2: Interpret the lift of the top rule
lift_a = None
for idx, rule in rules_a.iterrows():
    if f"{set(rule['antecedents'])} => {set(rule['consequents'])}" == top_rule_a:
        lift_a = rule['lift']
        break
lift_percentage_a = round((lift_a - 1) * 100, 2) if lift_a else None
statement_a = f"Identifying someone who watches {top_rule_a.split(' => ')[0].strip('{}')} as one who is also going to watch {top_rule_a.split(' => ')[1].strip('{}')} is {lift_percentage_a}% better than just guessing that a random individual is going to watch {top_rule_a.split(' => ')[1].strip('{}')}."

# Part b-1: min support = 5 transactions (5/88 ≈ 0.0568), min confidence = 25%
min_support_b = 5 / 88
rules_b, top_rule_b = generate_rules(one_hot, min_support_b, 0.25, "Part b-1", options_b)

# Part b-2: Interpret the lift of the top rule
lift_b = None
for idx, rule in rules_b.iterrows():
    if f"{set(rule['antecedents'])} => {set(rule['consequents'])}" == top_rule_b:
        lift_b = rule['lift']
        break
lift_percentage_b = round((lift_b - 1) * 100, 2) if lift_b else None
statement_b = f"Identifying someone who watches {top_rule_b.split(' => ')[0].strip('{}')} as one who is also going to watch {top_rule_b.split(' => ')[1].strip('{}')} is {lift_percentage_b}% better than just guessing that a random individual is going to watch {top_rule_b.split(' => ')[1].strip('{}')}."

# Print results
print("\nFinal Results:")
print("Part a-1:")
print(f"Top rule: {top_rule_a}")
print("\nPart a-2:")
print(statement_a)
print("\nPart b-1:")
print(f"Top rule: {top_rule_b}")
print("\nPart b-2:")
print(statement_b)