
import quandl
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

quandl.ApiConfig.api_key = '4D8hkYAV4WEkcTmD9LMW'

import pandas as pd
from sklearn.tree import DecisionTreeRegressor # Our DEcision Tree classifier


stock_data = quandl.get("NSE/ICICIBANK", rows=10000, sort_order="desc")

dataset = pd.DataFrame(stock_data)

dataset.to_csv('temp.csv')
data = pd.read_csv('temp.csv')
data.dropna(inplace=True)

import seaborn as sns
plt.figure(1 , figsize = (17 , 8))
cor = sns.heatmap(data.corr(), annot = True)
print(data.isnull().sum())


x = data.loc[:,'High':'Turnover (Lacs)']
y = data.loc[:,'Open']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

Classifier = DecisionTreeRegressor()
Classifier.fit(X_train,y_train)
y_pred = Classifier.predict(X_test)

print(y_pred)
print(y_test)

x = np.arange(len(y_test))
plt.scatter(x, y_test, color='red')
plt.scatter(x, y_pred, color='blue')
plt.title('ICICI Close Predict')
plt.xlabel('index')
plt.ylabel('values')
plt.show()


#
# #plot
# plt.figure(figsize=(16,8))
# plt.plot(data['Close'], label='Close Price history')

# y = pd.DataFrame(data['Close'])
#
# X = data.drop(['Close'], axis=1)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
#
# model = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
#                 max_depth = 5, alpha = 10, n_estimators = 1000)
#
# model.fit(X_train, y_train)
#
# y_pred = model.predict(X_test)
# print(y_pred)
# x = X_test.index.values
# plt.scatter(x, y_test, color='red')
# plt.scatter(x, y_pred, color='blue')
# plt.title('ICICI Close Predict')
# plt.xlabel('index')
# plt.ylabel('values')
# plt.show()
