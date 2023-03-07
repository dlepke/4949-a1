"""
OLS regression model with at least two features alongside a back-shifted target variable.
"""

import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn import metrics

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv("./mini_temp.csv", parse_dates=['Date Time'], index_col='Date Time')

df['T (degC) t-1'] = df['T (degC)'].shift(1)
df['Tpot (K) t-1'] = df['Tpot (K)'].shift(1)
df['VPmax (mbar) t-1'] = df['VPmax (mbar)'].shift(1)
df['rho (g/m**3) t-1'] = df['rho (g/m**3)'].shift(1)

df = df.dropna()

df = df.reset_index()

X = df.copy()

X = X[['T (degC) t-1', 'Tpot (K) t-1', 'VPmax (mbar) t-1', 'rho (g/m**3) t-1']]
print(X.head())

y = df[['T (degC)']]
print(y.head())

X = sm.add_constant(X)

pd.get_dummies(X)

len_data = len(X)
test_days = 50

X_train = X.iloc[:, :len_data - test_days]
X_test = X.iloc[len_data - test_days:, :]

y_train = y.iloc[:, :len_data - test_days]
y_test = y.iloc[len_data - test_days:, :]

model = sm.OLS(y_train, X_train).fit()

predictions = model.predict(X_test)

print(model.summary())

print('\n\nRoot Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
