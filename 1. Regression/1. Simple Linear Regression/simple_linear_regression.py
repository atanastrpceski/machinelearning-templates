## Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the dataset

dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

## Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

## Training
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

## Predicting
y_pred = regressor.predict(x_test)

## Visualize the Training set

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title("Salary vs Experiance (Training set)")
plt.xlabel("Experiance")
plt.ylabel("Salery")
plt.show()

## Visualize the Test set

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test, y_pred, color = 'blue')
plt.title("Salary vs Experiance (Test set)")
plt.xlabel("Experiance")
plt.ylabel("Salery")
plt.show()

## Making a single prediction

print(regressor.predict([[12]]))

## Getting the final linear regression equation with the values of the coefficients

print(regressor.coef_)
print(regressor.intercept_)

# Salary = 9345.94 × YearsExperience + 26816.19