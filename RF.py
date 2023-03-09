#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
import datetime as dt
import datetime


def RFresult(x1):
    data = pd.read_csv('file1.csv')

    #Train data processing
    data[['Date', 'Time']] = data['Datetime'].str.split(' ', 1, expand=True)
    data[['TimeS', 'TimeV']] = data['Time'].str.split('-', 1, expand=True)
    data['Date'] =  data['Date']+' '+data['TimeS']

    df = pd.DataFrame(columns = ['date', 'bat'])
    df = df.append({'date' : data['Date'], 'bat' : data['High']},ignore_index = True)

    x=pd.to_datetime(data['Date'], format='%Y-%m-%d %H:%M:%S')
    y=data['High'].values.reshape(-1, 1)
    z=data['Low'].values.reshape(-1, 1)

    #Predictive data processing
    #x1=['2021-12-31 15:15:00','2021-12-31 10:30:00','2021-12-31 14:00:00','2021-12-31 14:30:00']
    df1 = pd.DataFrame(columns = ['date'])
    df1= df1.append(pd.DataFrame(x1,columns=['date']),ignore_index = True)
    x2=pd.to_datetime(df1['date'], format='%Y-%m-%d %H:%M:%S')

    #ML Model

    lm = RandomForestRegressor(n_estimators=20, random_state=0)
    lm.fit(x.values.reshape(-1, 1),y)

    lmL = RandomForestRegressor(n_estimators=20, random_state=0)
    lmL.fit(x.values.reshape(-1, 1),z)

    #lm = svm.SVR()
    #lm.fit(x.values.reshape(-1, 1),y)

    #lmL = svm.SVR()
    #lmL.fit(x.values.reshape(-1, 1),z)

    #ML Predict
    predictions = lm.predict(x2.values.astype(float).reshape(-1, 1))
    predictions1 = lmL.predict(x2.values.astype(float).reshape(-1, 1))
    #print(predictions)

    return x,y,z,x2,predictions,predictions1
    #Ploat
    

    


# In[ ]:




