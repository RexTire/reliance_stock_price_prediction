import pandas as pd
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

df = pd.read_csv(r'reliance_max.csv', header=0, index_col='Date', parse_dates=True) #reading the file and indexing it to Date as it is needed when I define the logic to put data in newly created column Forcast according to the date


df.drop(['Close'], axis=1) # drop Close column
df['Close'] = df['Adj Close'] # assign Ajd. Close to new Close column
df = df.drop(['Adj Close'], axis=1) #drop Adj. Close Column

df['Daily Percent Change'] = (df['Close'] - df['Open']) / df['Open'] * 100 # made a new column which indicates daily percent change i.e. (close price - open price)/open price*100
df['H-L Percent Change'] = (df['High'] - df['Close']) / df['Close'] * 100 # made a new column which indicates High - Low percent change i.e. (high price - close price)/close price*100

df = df[['Close', 'H-L Percent Change', 'Daily Percent Change', 'Volume']] # filtering the useful columns

import math

df.dropna(inplace=True) # drop rows with NaN values here are 17 such rows

forcast_col = df['Close']
df.fillna(-99999, inplace=True) # fill NaN values
forcast_out = int(math.ceil(0.01*len(df))) # represent no. of days you want to forcast
df['label'] = forcast_col.shift(-forcast_out) # make a column called label and assigh it the values of forcast_col shifted upward by value of forcast_out i.e no of days 

import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import LinearRegression

X = np.array(df.drop(['label'], axis=1)) # X represent features. Here label column is not taken because it is not a feature. 
X = preprocessing.scale(X) # scaling X
X = X[:-forcast_out] # all values except last forcast_out number of values
X_lately = X[-forcast_out:] # last forcast_out number of values
df.dropna(inplace=True) # drop NaN values
y = np.array(df['label']) # y represent label. label column is label 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = LinearRegression() # choosing a classifier
#clf = svm.SVR()
clf.fit(X_train, y_train) # training
accuracy = clf.score(X_test, y_test) * 100 # testing 

forcast_set = clf.predict(X_lately) # forecasting values for next forcast_out days

df['Forcast'] = np.nan # creating a new column Forcast in df

# logic to put data in newly created column Forcast according to the date
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forcast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

# printing graph
df['Close'].plot()
df['Forcast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


print(forcast_set, accuracy, forcast_out) # printing forcast values, accuracy percentage and value of forcast_out
