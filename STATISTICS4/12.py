import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Read the CSV file (no header, assign column names)
csv_path = 'C:/Users/brais/OneDrive/Desktop/ML/STATISTICS4/12.csv'
try:
    data = pd.read_csv(csv_path, names=['Crime', 'Location'])
except FileNotFoundError:
    print(f"Error: '{csv_path}' not found. Please check the file path.")
    exit(1)
except pd.errors.EmptyDataError:
    print(f"Error: '{csv_path}' is empty. Please provide a valid CSV file.")
    exit(1)
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit(1)

# Convert to DataFrame
df = pd.DataFrame(data, columns=['Crime', 'Location'])

# Check for empty or missing data
if df['Crime'].isna().all() or df['Crime'].eq('').all():
    print("Error: 'Crime' column is empty or contains only missing values.")
    exit(1)
if df['Location'].isna().all() or df['Location'].eq('').all():
    print("Error: 'Location' column is empty or contains only missing values.")
    exit(1)

# Remove rows with missing or empty values
df = df.dropna(subset=['Crime', 'Location'])
df = df[(df['Crime'] != '') & (df['Location'] != '')]

# Verify data size
num_rows = len(df)
if num_rows != 2500:
    print(f"Warning: Expected 2500 rows, but loaded {num_rows} rows after cleaning.")
else:
    print(f"Loaded {num_rows} transactions from {csv_path}")

# (a) Explore data: Most frequent crime and location
crime_counts = df['Crime'].value_counts()
location_counts = df['Location'].value_counts()

if crime_counts.empty:
    print("(a) Most frequent crime: No crimes found")
else:
    print("(a) Most frequent crime:", crime_counts.idxmax(), f"(count: {crime_counts.max()})")
if location_counts.empty:
    print("    Most frequent location: No locations found")
else:
    print("    Most frequent location:", location_counts.idxmax(), f"(count: {location_counts.max()})")

# (b) Association Rule Mining
# Prepare transactions: Combine Crime and Location as items
transactions = df.apply(lambda row: [f"Crime_{row['Crime']}", f"Location_{row['Location']}"], axis=1).tolist()

# Convert to one-hot encoded format
te = TransactionEncoder()
te_ary = te.fit_transform(transactions)
transaction_df = pd.DataFrame(te_ary, columns=te.columns_)

# Generate frequent itemsets
min_support = 0.02  # 2% of 2500 = 50 transactions
frequent_itemsets = apriori(transaction_df, min_support=min_support, use_colnames=True)

# Generate association rules
min_confidence = 0.30  # 30%
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
rules = rules.sort_values("lift", ascending=False)

# (b-1) Number of rules and lift of top rule
print("\n(b-1) Number of rules generated:", len(rules))
print("      Lift ratio for top rule:", round(rules.iloc[0]["lift"], 2) if not rules.empty else "No rules generated")

# (b-2) Top rule
if not rules.empty:
    top_rule = rules.iloc[0]
    antecedents = ', '.join([str(x).replace('Crime_', '').replace('Location_', '') for x in top_rule['antecedents']])
    consequents = ', '.join([str(x).replace('Crime_', '').replace('Location_', '') for x in top_rule['consequents']])
    print("\n(b-2) Top rule:", f"{{{antecedents}}} => {{{consequents}}}")
else:
    print("\n(b-2) Top rule: No rules generated")

# (c) Most likely crimes by location
def most_likely_crime(location):
    loc_rules = rules[rules['consequents'].apply(lambda x: f'Location_{location}' in x)]
    if not loc_rules.empty:
        top_crime = loc_rules.iloc[0]['antecedents']
        return ', '.join([str(x).replace('Crime_', '') for x in top_crime])
    return "No strong rule"

locations = ['DEPARTMENT STORE', 'SIDEWALK', 'APARTMENT']
print("\n(c) Most likely crimes by location:")
for loc in locations:
    print(f"    {loc}: {most_likely_crime(loc)}")