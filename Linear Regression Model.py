# %%
import pandas as pd
df=pd.read_csv('C:/Users/brais/OneDrive/Desktop/ML/SalaryPrediction.csv')
df.head()

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

# %%
# Split the features (X) and target variable (Y)
x=df[['Experience Years']]
y=df[['Salary']]

# %%
# Split the data into training and testing set(80% train and 20% test)
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2, random_state=42)

# %%
# train the linear regression model
model=LinearRegression()
model.fit(X_train,Y_train)

# %%
# Model evaluation
y_pred=model.predict(X_test)

# %%
# Printing the performance metrics
print("/nModel Performance")
print(f"MAE:{mean_absolute_error(Y_test,y_pred):.2f}")
print(f"MSE:{mean_squared_error(Y_test,y_pred):.2f}")
print(f"R squared score:{r2_score(Y_test,y_pred):.2f}")


# %%
# Visualizing the regression Line
plt.scatter(X_train,Y_train, color='blue', label='Training Data')
plt.scatter(X_test,Y_test, color='red',label='Test Data')
plt.plot(X_test,y_pred,color='green', linewidth=2, label='Regression Line')
plt.xlabel("years of experience")
plt.ylabel("salary")
plt.title("SalaryPrediction using linear regression")
plt.legend()
plt.show()

# %%
try:
    experience = float(input("\nEnter years of experience: "))
    predicted_salary = model.predict(np.array([[experience]]).reshape(-1, 1))
    print(f"\nPredicted Salary for {experience} years of experience: ${predicted_salary.item():,.2f}")

except ValueError:
    print("\nInvalid input! Please enter a numeric value for years of experience.")


