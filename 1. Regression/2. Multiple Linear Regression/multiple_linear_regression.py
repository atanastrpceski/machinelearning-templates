## Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the dataset

dataset = pd.read_csv("50_Startups.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

## Encoding the Categorical Data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [-1])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))

## Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

## Training
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

## Predicting
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2);

## Compare predicated values vs real values
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis = 1))

## Making a single prediction
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

## Getting the final linear regression equation with the values of the coefficients
print(regressor.coef_)
print(regressor.intercept_)

# Profit= (86.6 × Dummy State 1) − (873 × Dummy State 2) + (786 × Dummy State 3) + (0.773 × R&D Spend) + (0.0329 × Administration) + (0.0366 × Marketing Spend) + (42467.53)
