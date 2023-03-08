"""
A stacked model that uses output from at least four separate time series models:
	- an OLS regression model with at least two features other than a back-shifted target variable
	- a model that uses exponentially smoothed data as a feature
	- an ARIMA model with back-shifted features
	- another model such as random forest or XGBoost
	
Must try:
	- scaling
	- binning
	- imputing
	- train/val/test split
"""

import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn import metrics
from xgboost import XGBRegressor
import statsmodels.tsa.arima.model as sma
import matplotlib.pyplot as plt

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

X = sm.add_constant(X)

X = X[['T (degC) t-1', 'Tpot (K) t-1', 'VPmax (mbar) t-1', 'rho (g/m**3) t-1', 'const']]
print(X.head())

y = df[['T (degC)']]
print(y.head())

pd.get_dummies(X)

len_data = len(X)
val_days = 30
test_days = 20
len_train = len_data - test_days - val_days
train_start = 0
train_end = len_train
val_start = train_end
val_end = val_start + val_days
test_start = val_end
test_end = len_data
print(train_start, train_end, val_start, val_end, test_start, test_end)

print(len_data, val_days, test_days, len_train)

X_train = X.iloc[:, :train_end]
X_val = X.iloc[val_start:val_end]
X_test = X.iloc[test_start:, :]

y_train = y.iloc[:, :train_end]
y_val = y.iloc[val_start:val_end]
y_test = y.iloc[test_start:, :]

print(len(y_train), len(y_val), len(y_test))


""" OLS model """
print("***** OLS model *****")
# X_ols = X[['T (degC) t-1', 'Tpot (K) t-1', 'VPmax (mbar) t-1', 'rho (g/m**3) t-1', 'const']]

# X_train_ols = X_ols.iloc[:, :len_train]
# X_val_ols = X_ols.iloc[len_train:, len_data - test_days]

ols_model = sm.OLS(y_train, X_train).fit()

ols_predictions = ols_model.predict(X_val)

print(ols_model.summary())

print('\n\nRoot Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, ols_predictions)))

""" XGB model """
print("***** XGB model *****")
xgb_model = XGBRegressor().fit(X_train, y_train)

xgb_predictions = xgb_model.predict(X_val)

print('\n\nRoot Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, xgb_predictions)))


""" Exponentially smoothed data model """
print("***** Exponential smoothing model *****")
X_exp = X.copy()
X_exp['T (degC) t-1'] = df['T (degC) t-1'].ewm(span=20, adjust=False).mean()

X_train_exp = X.iloc[:, :train_end]
X_val_exp = X.iloc[val_start:val_end]

exp_model = sm.OLS(y_train, X_train_exp).fit()

exp_predictions = exp_model.predict(X_val_exp)

print(exp_model.summary())

print('\n\nRoot Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, exp_predictions)))

print(len(y_val))


""" ARIMA model """
print("***** ARIMA model *****")

arima_model = sma.ARIMA(df['T (degC) t-1'], order=(2, 0, 2)).fit()
print(arima_model.summary())
arima_predictions = arima_model.predict(start=len_train, end=len_data - test_days - 1)

print('\n\nRoot Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, arima_predictions)))


""" Stacked model """
print("***** Stacked model *****")

pred_df = pd.DataFrame()
pred_df[0] = ols_predictions
pred_df[1] = xgb_predictions
pred_df[2] = exp_predictions
pred_df[3] = arima_predictions

stacked_model = sm.OLS(y_val, pred_df).fit()

pred_test_df = pd.DataFrame()
pred_test_df[0] = ols_model.predict(X_test)
pred_test_df[1] = xgb_model.predict(X_test)
pred_test_df[2] = exp_model.predict(X_test)
pred_test_df[3] = arima_model.predict(start=test_start, end=test_end)

stacked_predictions = stacked_model.predict(pred_test_df)

print(stacked_model.summary())

print('\n\nRoot Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, stacked_predictions)))

plt.scatter(stacked_predictions, y_test)
plt.title("Stacked Model Predictions vs Actual")
plt.xlabel("Stacked Model Predictions")
plt.ylabel("Actual Values")
plt.show()
