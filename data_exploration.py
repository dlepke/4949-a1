"""
Exploratory data analysis.

Please provide a thorough exploratory data analysis. Ensure the report is informative,
suitable and easy-to-follow for a non-technical audience. Use visualizations.

Identify trends, cycles or seasonal movements if they exist.
Identify attribute time steps that appear to be correlated with the target variable.

At the start of the report identify and focus on the most important features in your model
and explain how they are related to the target variable with visualizations.

"""

import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.impute import KNNImputer
from statsmodels.tsa.seasonal import seasonal_decompose

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# df = pd.read_csv("./jena_climate_2009_2016.csv", skiprows=1, parse_dates=['Date Time'], index_col='Date Time',
#                  names=('Date Time', 'p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
#                         'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)',
#                         'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)'))

df = pd.read_csv("./mini_temp.csv", parse_dates=['Date Time'])

print(df.head(50))
print(df.describe())
temp_df = pd.DataFrame()
temp_df['T (degC)'] = df.set_index('Date Time').resample('M')['T (degC)'].mean()
print(temp_df.head())

temp_df = temp_df.reset_index()

plt.figure(figsize=(15, 10))

plt.subplot(3, 5, 1)
plt.scatter(df['p (mbar)'], df['T (degC)'])
plt.title("p (mbar)")

plt.subplot(3, 5, 2)
plt.scatter(df['Tpot (K)'], df['T (degC)'])
plt.title("Tpot (K)")

plt.subplot(3, 5, 3)
plt.scatter(df['Tdew (degC)'], df['T (degC)'])
plt.title("Tdew (degC)")

plt.subplot(3, 5, 4)
plt.scatter(df['rh (%)'], df['T (degC)'])
plt.title("rh (%)")

plt.subplot(3, 5, 5)
plt.scatter(df['VPmax (mbar)'], df['T (degC)'])
plt.title("VPmax (mbar)")

plt.subplot(3, 5, 6)
plt.scatter(df['VPact (mbar)'], df['T (degC)'])
plt.title("VPact (mbar)")

plt.subplot(3, 5, 7)
plt.scatter(df['VPdef (mbar)'], df['T (degC)'])
plt.title("VPdef (mbar)")

plt.subplot(3, 5, 8)
plt.scatter(df['sh (g/kg)'], df['T (degC)'])
plt.title("sh (g/kg)")

plt.subplot(3, 5, 9)
plt.scatter(df['H2OC (mmol/mol)'], df['T (degC)'])
plt.title("H2OC (mmol/mol)")

plt.subplot(3, 5, 10)
plt.scatter(df['rho (g/m**3)'], df['T (degC)'])
plt.title("rho (g/m**3)")

plt.subplot(3, 5, 11)
plt.scatter(df['wv (m/s)'], df['T (degC)'])
plt.title("wv (m/s)")

plt.subplot(3, 5, 12)
plt.scatter(df['max. wv (m/s)'], df['T (degC)'])
plt.title("max. wv (m/s)")

plt.subplot(3, 5, 13)
plt.scatter(df['wd (deg)'], df['T (degC)'])
plt.title("wd (deg)")

plt.figure(figsize=(25, 6))

plt.subplot(1, 1, 1)
plt.scatter(y=temp_df['T (degC)'], x=temp_df['Date Time'])
plt.title("Temp vs Time")

plt.show()
