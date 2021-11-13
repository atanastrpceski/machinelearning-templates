## Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the dataset

dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

## Linear Regression (for comparing)
from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression()
lin_regressor.fit(x, y)

## Transform to Polynomial model
from sklearn.preprocessing import PolynomialFeatures
polynomial_regressor = PolynomialFeatures(degree = 4)
x_polynomial = polynomial_regressor.fit_transform(x)

## Linear Regression for the Polynomial model
from sklearn.linear_model import LinearRegression
lin_regressor_2 = LinearRegression()
lin_regressor_2.fit(x_polynomial, y)

## Visualize Linear Regression (you can see it's not a good fit)
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_regressor.predict(x), color = 'blue')
plt.title("Truth or Bluff (Linear)")
plt.xlabel("Position Level")
plt.ylabel("Salery")
plt.show()

## Visualize Polynomial Linear Regression (good fit)
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_regressor_2.predict(x_polynomial), color = 'blue')
plt.title("Truth or Bluff (Polynomial)")
plt.xlabel("Position Level")
plt.ylabel("Salery")
plt.show()

## Making a single prediction with the linear model (not good prediction)
print(lin_regressor.predict([[6.5]]))

## Making a single prediction with the polynomial model (good prediction)
print(lin_regressor_2.predict(polynomial_regressor.fit_transform([[6.5]])))
