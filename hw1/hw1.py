import numpy as np
import pandas as pd

cars = pd.read_csv('data.csv')

# Get the average price of BMW cars
bmw_mean = cars[cars['Make'] == 'BMW']['MSRP'].mean()

# Get the number of missing values after year 2015
cars[cars['Year'] >= 2015]['Engine HP'].isna().sum()

# Calculate engine horse power mean and fill the missing values
hp_mean = cars['Engine HP'].mean()
cars['Engine HP'].fillna(hp_mean, inplace=True)

# Calculate the inverse of inner product of X from specific columns of cars dataset
cars = cars[cars['Make'] == 'Rolls-Royce']
cars = cars[['Engine HP', 'Engine Cylinders', 'highway MPG']]
cars.drop_duplicates(inplace=True)

X = cars.to_numpy()
XTX = X.T.dot(X)
XTX_inv = np.linalg.inv(XTX)

# Normal equation
y = [1000, 1100, 900, 1200, 1000, 850, 1300]
w = XTX_inv.dot(X.T).dot(y)


