# %%
import kagglehub

# Download latest version
path = kagglehub.dataset_download("architsharma01/loan-approval-prediction-dataset")

print("Path to dataset files:", path)

# %%
import pandas as pd
df=pd.read_csv('loan_approval_dataset.csv')
df.head(10)


# %%
# Dropping the column load_id
df.drop(columns=["loan_id"], inplace=True)

# %%
df.columns=df.columns.str.strip()

# %%
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df["education"] = encoder.fit_transform(df["education"])
df["self_employed"]=encoder.fit_transform(df["self_employed"])
df["loan_status"]=encoder.fit_transform(df["loan_status"])

df.head()

# %%
# Checking if there is null values
df.isnull().sum()

# %%
from sklearn.model_selection import train_test_split
X=df.drop(columns=['loan_status'])
y=df['loan_status']
X_train, X_test, y_train, y_test=train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=42)

# %%
# Training a decison tree
from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier(criterion='gini',max_depth=5,random_state=42)

# Train the model
model.fit(X_train,y_train)

# %%
from sklearn.metrics import accuracy_score, classification_report
y_pred=model.predict(X_test)

# Evaluate accuracy
print("accuracy:", accuracy_score(y_test,y_pred))
print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))


# %%
# Visualizing the decision tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(24,24))
plot_tree(model,feature_names=X.columns,class_names=['rejected','approved'],filled=True)
plt.show()

# %%
# Sample prediction
import numpy as np

def predict_loan_approval(model):
    # Collect user input
    no_of_dependents = int(input("Enter number of dependents: "))
    education = input("Enter education (Graduate/Not Graduate): ")
    self_employed = input("Are you self-employed? (Yes/No): ")
    income_annum = int(input("Enter annual income: "))
    loan_amount = int(input("Enter loan amount: "))
    loan_term = int(input("Enter loan term (in months): "))
    cibil_score = int(input("Enter CIBIL score: "))
    residential_assets_value = int(input("Enter residential assets value: "))
    commercial_assets_value = int(input("Enter commercial assets value: "))
    luxury_assets_value = int(input("Enter luxury assets value: "))
    bank_asset_value = int(input("Enter bank asset value: "))

    # Encode categorical variables
    education = 1 if education.lower() == "graduate" else 0
    self_employed = 1 if self_employed.lower() == "yes" else 0

    # Create an input array
    user_data = np.array([[no_of_dependents, education, self_employed, income_annum, 
                           loan_amount, loan_term, cibil_score, residential_assets_value, 
                           commercial_assets_value, luxury_assets_value, bank_asset_value]])

    # Make prediction
    prediction = model.predict(user_data)
    
    # Output result
    if prediction[0] == 1:
        print("✅ Loan Approved!")
    else:
        print("❌ Loan Rejected.")

# Call the function
predict_loan_approval(model)


# %%



