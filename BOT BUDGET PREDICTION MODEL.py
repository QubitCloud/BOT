#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

# Load the data
data = pd.read_csv('expenditure.csv')
data.drop(0, inplace=True)
data.reset_index(drop=True, inplace=True)
data.columns = ['Year', 'Expenditure']

data['Year'] = data['Year'].apply(lambda x: int(x.split('/')[0]))
data['Expenditure'] = data['Expenditure'].str.replace(',', '').astype(float)

# Prepare the data for regression
X = data['Year'].values.reshape(-1, 1)
y = data['Expenditure'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with polynomial features and ridge regression
alpha = 0.01  # regularization strength
pipeline = Pipeline([
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('ridge_regression', Ridge(alpha=alpha))
])

# Perform cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Generate points for plotting
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_plot = pipeline.predict(X_plot)

# Calculate R-squared for training and test sets
r2_train = r2_score(y_train, pipeline.predict(X_train))
r2_test = r2_score(y_test, pipeline.predict(X_test))

# Plot the results
plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, color='blue', alpha=0.7, label='Training data')
plt.scatter(X_test, y_test, color='green', alpha=0.7, label='Test data')
plt.plot(X_plot, y_plot, color='red', label='Ridge regression (degree=2)')
plt.xlabel('Year')
plt.ylabel('Expenditure (Million TZS)')
plt.title(f'Polynomial Ridge Regression: Tanzania Government Expenditure\n'
          f'R-squared (train): {r2_train:.4f}, R-squared (test): {r2_test:.4f}\n'
          f'Cross-validation R-squared: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print the coefficients and intercept
print("Coefficients:", pipeline.named_steps['ridge_regression'].coef_)
print("Intercept:", pipeline.named_steps['ridge_regression'].intercept_)

# Function to predict expenditure for a given year
def predict_expenditure(year):
    prediction = pipeline.predict([[year]])
    return prediction[0]

# User input and prediction
while True:
    user_input = input("Enter a year between 1963 and 2040 to predict expenditure (or 'q' to quit): ")
    if user_input.lower() == 'q':
        break
    try:
        year = int(user_input)
        if year < 1963 or year > 2040:
            print("Error: Please enter a year between 1963 and 2040.")
            continue
        if year > data['Year'].max():
            print(f"Warning: The entered year is beyond our original data range (up to {data['Year'].max()}).")
            print("The prediction may be less reliable for future years.")
        predicted_expenditure = predict_expenditure(year)
        print(f"Predicted expenditure for {year}: {predicted_expenditure:.2f} Million TZS")
    except ValueError:
        print("Invalid input. Please enter a valid year.")

print("Thank you for using the expenditure prediction tool!")

