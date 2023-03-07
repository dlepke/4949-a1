"""
ARIMA model.
"""

import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.arima.model as sma
import numpy as np
from sklearn import metrics

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv("./mini_temp.csv", parse_dates=['Date Time'])

df['T (degC) t-1'] = df['T (degC)'].shift(1)
df['Tpot (K) t-1'] = df['Tpot (K)'].shift(1)
df['VPmax (mbar) t-1'] = df['VPmax (mbar)'].shift(1)
df['rho (g/m**3) t-1'] = df['rho (g/m**3)'].shift(1)

df = df.dropna()

df = df.reset_index()

X = df.copy()

X = X[['T (degC) t-1', 'Tpot (K) t-1', 'VPmax (mbar) t-1', 'rho (g/m**3) t-1']]

y = df[['T (degC)']]
# print(y.head())

X = sm.add_constant(X)

pd.get_dummies(X)

len_data = len(X)
test_days = 50
len_train = len_data - test_days

X_train = X.iloc[:, :len_data - test_days]
X_test = X.iloc[len_data - test_days:, :]

y_train = y.iloc[:, :len_data - test_days]
y_test = y.iloc[len_data - test_days:, :]

model_stats = []
df_stats = []

# optimal: ar 2, ma 2
# for ar in range(0, 5):
#     for ma in range(0, 5):
#         model = sma.ARIMA(df['T (degC)'], order=(ar, 0, ma)).fit()

#
#         print(model.summary())
#         predictions = model.predict(start=len_train, end=len_data - 1)
#         mse = mean_squared_error(predictions, y_test)
#         rmse = np.sqrt(mse)
#         print('RMSE: ', str(rmse))
#
#         model_stats.append({
#             "ar": ar,
#             "ma": ma,
#             "rmse": rmse
#         })
#
#         df_stats.append({
#             "ar": ar,
#             "ma": ma,
#             "rmse": rmse
#         })


# df_stats = pd.DataFrame(df_stats)
# df_stats.sort_values(by='rmse')
# print(df_stats)

# not sure if I need to backshift ARIMA data - playing it safe
model = sma.ARIMA(df['T (degC) t-1'], order=(2, 0, 2)).fit()
print(model.summary())
predictions = model.predict(start=len_train, end=len_data - 1)

print('\n\nRoot Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
